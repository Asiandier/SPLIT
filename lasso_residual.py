"""Residual-based weighted Lasso: original objective and full-marker KKT.

The quadratic proposal metric is built from a cheap genotype Gram, optionally
preconditioned by the existing REML low-rank inverse. Exact residual gradients,
certified line searches and full-marker KKT determine the accepted solution.
No candidate H^-1 Z matrix is needed for the path search.
"""
from __future__ import annotations
import logging
import json
import math
import time
from collections import OrderedDict
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

from . import sparse_fp64
from .lasso_recycle import RecycledInverse
from .lasso_cd import solve_lasso_cd_gram, make_lambda_sequence
from .pcg import pcg_solve
from .precond import scalar_diag_from_precond_conf, build_projected_core_runtime

logger = logging.getLogger(__name__)

def solve_adaptive_pcg(hv, rhs, M=None, tol=1e-8, maxiter=400, X0=None,
                       check_every=2, recycle=None, verified_hv=None):
    with jax.experimental.enable_x64():
        rhs = jnp.asarray(rhs, dtype=jnp.float64)
        value = jnp.zeros_like(rhs) if X0 is None else jnp.asarray(X0, dtype=jnp.float64)
        scale = jnp.linalg.norm(rhs, axis=0)
        value = jnp.where(scale > 0, value, 0.)
        scale = jnp.where(scale > 0, scale, 1.)
        used = 0
        previous = math.inf
        recycled_precond = M if recycle is None else recycle.preconditioner(M)
        coarse_residual = None
        if recycle is not None and recycle.rank:
            # The Galerkin minimizer over previous solution directions is
            # U U' rhs. It subsumes a warm start in span(U), without applying
            # the covariance to that warm start again.
            value, coarse_residual = recycle.coarse(rhs)
        while True:
            verified = coarse_residual is None
            if verified:
                product = (hv if verified_hv is None else verified_hv)(value)
                residual = rhs-product
            else:
                residual, coarse_residual = coarse_residual, None
            actual = float(jnp.max(jnp.linalg.norm(residual, axis=0)/scale))
            if not np.isfinite(actual):
                raise FloatingPointError('Nonfinite verified residual')
            if not verified and (actual <= tol or used >= maxiter or actual >= previous):
                continue  # No result is returned without a true FP64 residual.
            if actual <= tol:
                if recycle is not None:
                    recycle.observe(value, product)
                return value, actual, used
            if used >= maxiter or actual >= previous:
                return value, actual, used
            previous = actual
            # r_new = r - H delta. A correction relative residual eta bounds
            # the global relative residual by eta*actual. The factor 1/2
            # leaves room for FP32 correction and addition roundoff. Any
            # failure of this estimate is caught by the FP64 verification.
            eta = min(.5, max(np.finfo(np.float32).eps, .5*tol/actual))
            correction, _, count = pcg_solve(hv, residual.astype(jnp.float32),
                M=recycled_precond, tol=eta, maxiter=maxiter-used, check_every=check_every,
                stop_on_residual_gap=True)
            used += int(count)
            if not count:
                if not verified:
                    continue
                return value, actual, used
            value += correction.astype(jnp.float64)


def quadratic_l1_step(beta, direction, score, curvature, lam):
    """Minimize the exact convex loss on beta + t*direction, 0 <= t <= 1."""
    linear = -float(score @ direction)
    scale = max(1., abs(linear), abs(curvature), lam*np.sum(np.abs(direction)))
    if curvature < -128*np.finfo(float).eps*scale:
        raise ValueError('Negative measured target curvature')
    curvature = max(0., float(curvature))

    def derivative(t):
        point = beta+t*direction
        return linear+t*curvature+lam*float(np.sign(point)@direction)

    if derivative(1.) <= 0:
        return 1.
    lo, hi = 0., 1.
    for _ in range(60):
        mid = .5*(lo+hi)
        if derivative(mid) > 0:
            hi = mid
        else:
            lo = mid
    # Include exact zero crossings to avoid artificial tiny active coefficients.
    candidates = [0., 1., .5*(lo+hi)]
    moving = direction != 0
    roots = -beta[moving]/direction[moving]
    roots = roots[(roots >= lo-1e-14) & (roots <= hi+1e-14)]
    candidates.extend(roots.tolist())
    def change(t):
        return t*linear+.5*t*t*curvature+lam*(np.sum(np.abs(beta+t*direction))-np.sum(np.abs(beta)))
    return float(min(candidates, key=change))


def bfgs_update(metric, step, change):
    """Maintain positive curvature; skip pairs dominated by roundoff."""
    product = metric@step
    denom = float(step@product)
    target = float(step@change)
    eps = 128*np.finfo(float).eps
    if denom <= eps*np.linalg.norm(step)*np.linalg.norm(product):
        return None
    if target <= eps*np.linalg.norm(step)*np.linalg.norm(change):
        return None
    # BLAS rank-one updates avoid an extra k-by-k temporary.
    sla.blas.dger(-1./denom, product, product, a=metric, overwrite_a=1)
    sla.blas.dger(1./target, change, change, a=metric, overwrite_a=1)
    return product, denom, target


def expand_bfgs_metric(metric, cross, block, pairs, old_indices, new_indices):
    """Extend only new rows/columns; the old block already has all updates.

    Stored products are B_i s_i, before update i. New coordinates have s_i=0,
    so their product is cross_i.T @ s_i. Induction gives the same expanded
    BFGS matrix without replaying the old-by-old rank updates.
    """
    cross = np.asfortranarray(cross)
    block = np.asfortranarray((block+block.T)*.5)
    for pair in pairs:
        step = pair['step']
        new_product = cross[:len(step), :].T@step
        old_product = pair['product']
        old_change = pair['change'][old_indices]
        new_change = pair['change'][new_indices]
        for scale, left, right in [(-1/pair['denominator'], old_product, new_product),
                                   (1/pair['target'], old_change, new_change)]:
            sla.blas.dger(scale, left, right, a=cross, overwrite_a=1)
        sla.blas.dger(-1/pair['denominator'], new_product, new_product, a=block, overwrite_a=1)
        sla.blas.dger(1/pair['target'], new_change, new_change, a=block, overwrite_a=1)
        pair['product'] = np.concatenate([old_product, new_product])
    # np.block followed by asfortranarray temporarily duplicates the complete
    # k-by-k metric. Populate the final column-major buffer directly instead.
    old_size, new_size = metric.shape[0], block.shape[0]
    expanded = np.empty((old_size+new_size, old_size+new_size), order='F')
    expanded[:old_size, :old_size] = metric
    expanded[:old_size, old_size:] = cross
    expanded[old_size:, :old_size] = cross.T
    expanded[old_size:, old_size:] = block
    return expanded


def extrapolate(beta, score, indices, previous_beta, previous_score,
                lam, old_lam, previous_lam, error_bound, previous_error_bound):
    """Return a descending predictor and its conservative score-error bound.

    ``beta`` and ``previous_beta`` use the same append-only working-set order;
    the older vector may be shorter. ``score`` covers EVERY variant.
    """
    delta = float(old_lam-previous_lam)
    if delta >= 0 or lam >= old_lam:
        raise ValueError('Continuation needs three strictly decreasing lambdas')
    ratio = float((lam-old_lam)/delta)
    previous_beta = np.pad(previous_beta, (0, len(beta)-len(previous_beta)))
    direction = beta-previous_beta
    score_direction = score-previous_score
    proposal = beta+ratio*direction
    curvature = -float(direction@score_direction[indices])
    change = (-ratio*float(score[indices]@direction)
              + .5*ratio**2*curvature
              + lam*float(np.abs(proposal).sum()-np.abs(beta).sum()))
    # Account for inexact covariance solves in BOTH the slope and curvature.
    uncertainty = (ratio*error_bound*np.abs(direction).sum()
                   + .5*ratio**2*(error_bound+previous_error_bound)
                   * np.abs(direction).sum())
    accept = bool(np.isfinite(change) and change+uncertainty < 0.)
    return dict(accepted=accept, ratio=ratio, beta=proposal,
                score=score+ratio*score_direction,
                objective_change=change, objective_uncertainty=uncertainty,
                error_bound=(1.+ratio)*error_bound+ratio*previous_error_bound)



def affine_gradient_roundoff(beta0, beta1, score0, score1, weight0, weight1, gram_bound):
    """FP64 affine-update allowance using |(Z' P_C Z)_ij| <= gram_bound.

    Solve-error bounds are propagated separately. Eight rounding units cover
    the short affine expressions, including a subtract-then-add evaluation.
    The allowance affects certification only, never the fitted objective.
    """
    eps = np.finfo(np.float64).eps
    def l1_upper(value):
        size = np.size(value)
        total = float(np.sum(np.abs(value)))
        return np.nextafter(total/(1.-size*eps), np.inf)
    alpha_scale = abs(weight0)*l1_upper(beta0)+abs(weight1)*l1_upper(beta1)
    score_scale = (abs(weight0)*float(np.max(np.abs(score0)))
                   +abs(weight1)*float(np.max(np.abs(score1))))
    return float((8.*eps/(1.-8.*eps))*(gram_bound*alpha_scale+score_scale))


def check_certificate(score, indices, beta, lam, tolerance, error_bound):
    active = np.flatnonzero(beta != 0.)
    selected = indices[active]
    outside = np.abs(score).copy()
    outside[selected] = 0.
    active_error = (float(np.max(np.abs(score[selected]-lam*np.sign(beta[active]))))
                    if len(active) else 0.)
    inactive_error = max(0., float(np.max(outside))-lam)
    observed = max(active_error, inactive_error)
    certified = bool(observed+error_bound <= tolerance)
    # If the computed KKT already passes, changing beta is the wrong action
    # while its score uncertainty prevents certification. Refine the score.
    refine = bool(observed <= tolerance and not certified)
    budget = None
    if refine:
        margin = tolerance-observed
        budget = .25*min(error_bound, margin) if margin > 0 else .25*error_bound
    return dict(active_error=active_error, inactive_error=inactive_error,
        observed=observed, upper_bound=observed+error_bound,
        certified=certified, refine_score=refine, next_error_budget=budget)


class ProfiledInverseMetric:
    """M_C = M - MC(C'MC)^-1 C'M, M = d^-1 I + U E U'.

    All quantities are proposal geometry only. Acceptance and convergence must
    still use the exact weighted-Lasso loss and its full-marker KKT conditions.
    """
    def __init__(self, inverse_diagonal, inverse_core, covar_gram, u_covar):
        self.dinv = float(inverse_diagonal)
        self.e = np.asarray(inverse_core, dtype=np.float64)
        self.uc = np.asarray(u_covar, dtype=np.float64)
        self.ec = self.e @ self.uc
        cc = self.dinv*np.asarray(covar_gram) + self.uc.T@self.ec
        self.chol = sla.cho_factor((cc+cc.T)*.5, lower=True, check_finite=False)

    def covar_features(self, covar_z, u_z):
        return self.dinv*np.asarray(covar_z) + self.ec.T@u_z

    def block(self, gram, u_left, u_right, c_left, c_right):
        return (self.dinv*np.asarray(gram) + u_left.T@(self.e@u_right)
                - c_left.T@sla.cho_solve(self.chol, c_right, check_finite=False))


def _matmul(a, b, *, transpose_a=False, workspace_limit=0):
    if workspace_limit > 0 and next(iter(a.devices())).platform == 'gpu':
        return sparse_fp64.matmul(a, b, transpose_a=transpose_a,
            workspace_limit=workspace_limit)
    return (a.T if transpose_a else a)@b


class _DeviceColumnCache:
    """Keep decoded columns on the host and cache FP64 device copies on demand.

    Eviction changes storage only. Columns, their order, and every arithmetic
    operation remain the same. Pinned operands survive until their operation
    completes; scratch and operator workspaces are reserved independently.
    """
    def __init__(self, host_blocks, device, *, memory_limit=None, reserve_bytes=0):
        self.host_blocks = host_blocks
        self.device = device
        self.memory_limit = memory_limit
        self.reserve_bytes = int(reserve_bytes)
        self.arrays = OrderedDict()
        self.pins = {}
        self.evictions = self.uploads = self.uploaded_bytes = 0
        self.peak_resident_bytes = 0

    def _make_room(self, additional_bytes=0):
        stats = self.device.memory_stats()
        if stats is None:
            return
        capacity = int(stats['bytes_limit'])
        if self.memory_limit is not None and self.memory_limit > 0:
            capacity = min(capacity, int(self.memory_limit))
        required = int(additional_bytes)+self.reserve_bytes
        while int(stats['bytes_in_use'])+required > capacity:
            victim = next((i for i in self.arrays if not self.pins.get(i, 0)), None)
            if victim is None:
                raise MemoryError(
                    'Residual Lasso needs more device memory for two column blocks '
                    'and its operator workspace, even after evicting cached columns')
            value = self.arrays.pop(victim)
            value.block_until_ready()
            value.delete()
            self.evictions += 1
            stats = self.device.memory_stats()

    @contextmanager
    def hold(self, index):
        if index not in self.arrays:
            host = self.host_blocks[index]
            size = int(host.size)*np.dtype(np.float64).itemsize
            self._make_room(size)
            with jax.default_device(self.device):
                self.arrays[index] = jnp.asarray(host, dtype=jnp.float64)
            self.uploads += 1
            self.uploaded_bytes += size
            self.peak_resident_bytes = max(self.peak_resident_bytes,
                sum(int(a.size)*a.dtype.itemsize for a in self.arrays.values()))
        self.arrays.move_to_end(index)
        self.pins[index] = self.pins.get(index, 0)+1
        try:
            yield self.arrays[index]
        finally:
            self.pins[index] -= 1

    def clear(self):
        if any(self.pins.values()):
            raise RuntimeError('Cannot evict a live Lasso matrix operand')
        for value in self.arrays.values():
            value.block_until_ready()
            value.delete()
        self.arrays.clear()

    def metrics(self):
        return dict(evictions=self.evictions, uploads=self.uploads,
            uploaded_bytes=self.uploaded_bytes, peak_resident_bytes=self.peak_resident_bytes,
            reserve_bytes=self.reserve_bytes)


def fit_weighted_residual_path(*, fitter, grm_index, hv, precond, theta, y,
        covar, lam_max, path_cfg, pcg_tol, max_pcg_iters, pcg_ridge,
        marker_batch, max_iterations, workspace_bytes, validation_evaluator=None,
        stopping_rule=None, covariate_ridge=0., use_preconditioned_metric=True,
        validation_requires_training_design=True,
        previous_candidate=None, previous_beta_path=None,
        materialize_result_design=True, hv_with_score=None):
    """Fit a validation path or a frozen lambda ratio without inverse columns.

    ``stopping_rule`` and ``validation_evaluator`` are supplied by the supported
    sparse pipeline. They retain the original metric, grid and selection rule.
    The returned inverse-design cache is empty by design; selected-span
    information is computed separately after validation has selected a model.
    """
    with jax.experimental.enable_x64():
        return _fit_path(fitter=fitter, grm_index=grm_index, hv=hv,
            precond=precond, theta=theta, y=y, covar=covar, lam_max=lam_max,
            path_cfg=path_cfg, pcg_tol=pcg_tol, max_pcg_iters=max_pcg_iters,
            pcg_ridge=pcg_ridge, marker_batch=marker_batch,
            max_iterations=max_iterations, workspace_bytes=workspace_bytes,
            validation_evaluator=validation_evaluator, stopping_rule=stopping_rule,
            covariate_ridge=covariate_ridge,
            use_preconditioned_metric=use_preconditioned_metric,
            validation_requires_training_design=validation_requires_training_design,
            previous_candidate=previous_candidate, previous_beta_path=previous_beta_path,
            materialize_result_design=materialize_result_design,
            hv_with_score=hv_with_score)


def _fit_path(*, fitter, grm_index, hv, precond, theta, y, covar, lam_max,
        path_cfg, pcg_tol, max_pcg_iters, pcg_ridge, marker_batch,
        max_iterations, workspace_bytes, validation_evaluator, stopping_rule,
        covariate_ridge, use_preconditioned_metric, validation_requires_training_design,
        previous_candidate, previous_beta_path, materialize_result_design,
        hv_with_score):
    started = time.perf_counter()
    n = np.asarray(y).size
    fixed = path_cfg.fixed_lam_ratio is not None
    if len(fitter.streamers) != 1:
        raise ValueError('Residual Lasso requires one genotype source')
    if not fixed and (validation_evaluator is None or stopping_rule is None):
        raise ValueError('Residual validation path requires an evaluator and stopping rule')
    if fixed and not 0 < path_cfg.fixed_lam_ratio <= 1:
        raise ValueError('Fixed Lasso ratio must lie in (0, 1]')
    if not np.isfinite(covariate_ridge) or covariate_ridge < 0:
        raise ValueError('Residual Lasso requires a finite nonnegative covariate ridge')
    if not np.isfinite(lam_max) or lam_max <= 0:
        raise ValueError('Residual validation path requires a positive lambda maximum')
    theta = jnp.asarray(theta, dtype=jnp.float32)
    sigma = float(theta[-1])
    if sigma <= 0 or not np.all(np.asarray(theta[:-1]) >= 0):
        raise ValueError('Covariance must have a positive residual component')
    yj = jnp.asarray(np.asarray(y).reshape(-1, 1), dtype=jnp.float64)
    c = (np.empty((n, 0)) if covar is None else np.asarray(covar, dtype=np.float64))
    qc_host, rc = sla.qr(c, mode='economic', check_finite=False)
    if np.any(np.abs(np.diag(rc)) <= np.finfo(float).eps*np.linalg.norm(c)):
        raise ValueError('Covariates must have full column rank')
    qc = jnp.asarray(qc_host)
    qc32 = qc.astype(jnp.float32)
    hqc = hv(qc).block_until_ready() if c.shape[1] else qc
    stream = fitter.streamers[0]
    # Training columns use their empirical standard deviations. The rounding
    # allowance covers their stored FP32 means/scales; target KKT is unchanged.
    column_norm_bound = np.sqrt(n)*(1+8*np.finfo(np.float32).eps)

    def project(value):
        q = qc32 if value.dtype == jnp.float32 else qc
        return value-q@(q.T@value)

    transform = adjoint = project
    transform_norm = 1.
    ridged = covariate_ridge > 0 and c.shape[1] > 0
    if ridged:
        # With C=Q R and ridge rho, the profiled inverse is
        # (H + C C'/rho)^-1. Congruence by A=P+Q(sqrt(rho) R^-1)Q'
        # gives A H A' + QQ', avoiding the large 1/rho eigenvalues.
        small = math.sqrt(covariate_ridge)*sla.solve_triangular(
            rc, np.eye(rc.shape[0]), check_finite=False)
        transform_norm = max(1., float(np.linalg.norm(small, ord='fro')))
        small64 = jnp.asarray(small)
        small32 = small64.astype(jnp.float32)

        def apply_transform(value, *, transpose=False):
            q, d = (qc32, small32) if value.dtype == jnp.float32 else (qc, small64)
            coordinates = q.T@value
            return value+q@(((d.T if transpose else d)@coordinates)-coordinates)

        transform = apply_transform
        adjoint = lambda value: apply_transform(value, transpose=True)

    def projected_hv(value):
        w = adjoint(value)
        return transform(hv(w))+(value-(project(value) if ridged else w))

    def projected_pre(value):
        w = project(value)
        return project(precond(w) if precond is not None else w)+(value-w)

    geometry = None
    if use_preconditioned_metric and fitter.precond_conf is not None:
        d = scalar_diag_from_precond_conf(fitter.precond_conf, theta[:-1], theta[-1])
        runtime = build_projected_core_runtime(fitter.precond_conf, theta[:-1], d, eps=pcg_ridge)
        if runtime is not None:
            u_basis = runtime.U.astype(jnp.float64)
            u_cov = np.asarray(_matmul(u_basis, jnp.asarray(c),
                transpose_a=True, workspace_limit=workspace_bytes))
            e = sla.cho_solve((np.asarray(runtime.chol, dtype=np.float64), True),
                np.eye(runtime.total_rank))-float(runtime.d_inv)*np.eye(runtime.total_rank)
            geometry = ProfiledInverseMetric(float(runtime.d_inv), e, c.T@c, u_cov)
            u_features = np.empty((runtime.total_rank, 0))
            c_features = np.empty((c.shape[1], 0))
    indices = np.empty(0, dtype=np.int64)
    beta = np.empty(0)
    host_blocks = []
    # The old append-only device list could fill VRAM on later outer rounds.
    # Keep its exact host values and reuse available VRAM without constraining
    # the statistical working set. Reserve the existing operator/GEMM scratch
    # plus a conversion block and small products while two operands are live.
    scratch = (workspace_bytes if workspace_bytes > 0 else
        16*n*int(getattr(stream, 'call_width', 0)))
    max_block = 2*max(1, int(marker_batch))
    cache = _DeviceColumnCache(host_blocks, next(iter(yj.devices())),
        memory_limit=getattr(getattr(fitter, 'cfg', None), 'gpu_budget_bytes', None),
        reserve_bytes=scratch+8*n*max_block+8*max_block**2+64*n)
    block_of_marker = np.full(int(grm_index.m_total), -1, dtype=np.int32)
    column_of_marker = np.zeros(int(grm_index.m_total), dtype=np.int32)

    def cached_design(markers):
        # Working-set columns have already been decoded. Reuse their FP32
        # host copies for validation and the returned design, avoiding another
        # packed-genotype scan for every lambda and again at return.
        result = np.empty((n, len(markers)), dtype=np.float32, order='F')
        for output_column, marker in enumerate(markers):
            block_id = int(block_of_marker[marker])
            if block_id < 0:
                raise RuntimeError('Requested marker is absent from the Lasso design cache')
            result[:, output_column] = host_blocks[block_id][:, column_of_marker[marker]]
        return result
    metric = np.empty((0, 0))
    pairs = []
    rows, solutions, gammas, metrics, trace = [], [], [], [], []
    counters = dict(gradient_calls=0, residual_solve_seconds=0., metric_seconds=0.,
        subproblem_seconds=0., validation_seconds=0., max_pcg_residual=0.,
        bfgs_update_seconds=0.,
        fitted_seconds=0., covariance_solve_seconds=0., score_seconds=0.,
        gradient_events=[], certificate_reuses=0,
        max_pcg_iterations=0, validation_pcg_residual=0., validation_pcg_iterations=0)
    # Reuse only coefficients, never old gradients or curvature after H changes.
    # A previous outer path supplies an optional CD starting point at the same
    # lambda-grid index. Restrict it to this working set and accept it only if
    # it lowers the CURRENT convex quadratic subproblem. The unchanged inner
    # and full-marker KKT checks still determine every returned solution.
    old_column = np.full(int(grm_index.m_total), -1, dtype=np.int64)
    prior_path = None
    if previous_beta_path is not None:
        prior_indices = np.asarray(previous_candidate, dtype=np.int64)
        prior_path = np.asarray(previous_beta_path, dtype=np.float64)
        if (prior_indices.ndim != 1 or np.unique(prior_indices).size != prior_indices.size
                or np.any(prior_indices < 0) or np.any(prior_indices >= grm_index.m_total)
                or prior_path.ndim != 2 or prior_path.shape[1] != len(prior_indices)
                or not np.isfinite(prior_path).all()):
            raise ValueError('Previous Lasso coefficient path has an invalid marker layout')
        old_column[prior_indices] = np.arange(len(prior_indices))
    warm_rows_used = set()
    lam = float(lam_max)

    def tolerance_at(value):
        return max(path_cfg.kkt_abs_tol, path_cfg.kkt_rel_tol*max(1., value))

    def fitted(coef):
        value = jnp.zeros((n, 1), dtype=jnp.float64)
        start = 0
        for index, host in enumerate(host_blocks):
            end = start+host.shape[1]
            if np.any(coef[start:end] != 0.):
                with cache.hold(index) as block:
                    value = (value+block@jnp.asarray(coef[start:end, None])).block_until_ready()
            start = end
        return value

    recycling = RecycledInverse(n, workspace_bytes=workspace_bytes,
        memory_limit=getattr(getattr(fitter, "cfg", None), "gpu_budget_bytes", None))

    def evaluate(coef, warm, error_budget=None, reason="proposal"):
        tick = time.perf_counter()
        residual = yj-fitted(coef)
        rhs = transform(residual)
        initial = projected_pre(rhs) if warm is None else warm
        rhs.block_until_ready()
        fitted_seconds = time.perf_counter()-tick
        solve_started = time.perf_counter()
        bound_scale = column_norm_bound*transform_norm*float(jnp.linalg.norm(rhs))/min(sigma, 1.)
        budget = .1*tolerance_at(lam) if error_budget is None else float(error_budget)
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError('Score-error budget must be positive and finite')
        requested = min(float(pcg_tol), budget/max(bound_scale, 1.))
        retained = None
        def verified_product(value):
            nonlocal retained
            w = adjoint(value)
            product, score = hv_with_score(w)
            retained = (value, w, score)
            return transform(product)+(value-(project(value) if ridged else w))
        solved, error, count = solve_adaptive_pcg(projected_hv, rhs,
            M=projected_pre, tol=requested, maxiter=max_pcg_iters, X0=initial, recycle=recycling,
            verified_hv=verified_product if hv_with_score is not None else None)
        if error > requested:
            raise RuntimeError('Lasso residual solve failed its requested accuracy')
        reused = retained is not None and retained[0] is solved
        dual = retained[1] if reused else adjoint(solved)
        dual.block_until_ready()
        solve_seconds = time.perf_counter()-solve_started
        score_started = time.perf_counter()
        score = np.array(retained[2] if reused else stream.xtv(dual, normalize=False),
                         copy=True).reshape(-1)
        counters['transpose_products_reused'] = counters.get('transpose_products_reused', 0)+int(reused)
        score_seconds = time.perf_counter()-score_started
        seconds = time.perf_counter()-tick
        event = dict(reason=reason, lam=float(lam), support=int(np.count_nonzero(coef)),
            workset=len(coef), fitted_seconds=fitted_seconds, covariance_solve_seconds=solve_seconds,
            score_seconds=score_seconds,
            requested_tolerance=float(requested), true_residual=float(error), pcg_iterations=int(count))
        counters['gradient_events'].append(event)
        logger.debug('[residual Lasso solve] %s', json.dumps(event))
        for key in ['fitted_seconds','covariance_solve_seconds','score_seconds']:
            counters[key] += event[key]
        counters['gradient_calls'] += 1
        counters['residual_solve_seconds'] += seconds
        counters['max_pcg_residual'] = max(counters['max_pcg_residual'], float(error))
        counters['max_pcg_iterations'] = max(counters['max_pcg_iterations'], int(count))
        return score, solved if ridged else dual, dict(seconds=seconds, residual=float(error),
            iterations=int(count), gradient_error_bound=bound_scale*float(error))

    gradient, warm, diagnostic = evaluate(beta, None, reason="initial")
    if fixed:
        # Freeze the requested ratio against the full, profiled zero-model
        # score, with the same covariance and nuisance ridge as the target.
        lam_max = float(np.max(np.abs(gradient)))
        lambdas = np.unique([lam_max, lam_max*path_cfg.fixed_lam_ratio])[::-1]
    else:
        lambdas = make_lambda_sequence(lam_max, path_cfg.lam_min_ratio, path_cfg.n_lambda)
    previous_lam = lam
    previous_state = None
    stopping = {}
    for path_index, lam_value in enumerate(lambdas):
        lam = float(lam_value)
        tolerance = tolerance_at(lam)
        old_state = dict(beta=beta.copy(), score=gradient.copy(), warm=warm,
            lam=previous_lam, bound=diagnostic['gradient_error_bound'])
        prior_attempted = False
        if previous_state is not None:
            guess = extrapolate(beta, gradient, indices,
                previous_state['beta'], previous_state['score'], lam, previous_lam,
                previous_state['lam'], diagnostic['gradient_error_bound'], previous_state['bound'])
            if guess['accepted']:
                guess['error_bound'] += affine_gradient_roundoff(
                    beta, previous_state['beta'], gradient, previous_state['score'],
                    1.+guess['ratio'], -guess['ratio'], column_norm_bound**2/sigma)
                beta, gradient = guess['beta'], guess['score']
                warm = warm+guess['ratio']*(warm-previous_state['warm'])
                diagnostic = dict(diagnostic, gradient_error_bound=guess['error_bound'])
        for iteration in range(max_iterations+1):
            check = check_certificate(gradient, indices, beta, lam, tolerance,
                diagnostic['gradient_error_bound'])
            trace.append(dict(path_index=path_index, iteration=iteration,
                workset=len(indices), selected=int(np.count_nonzero(beta)),
                kkt_upper_bound=check['upper_bound'], kkt_tolerance=tolerance,
                elapsed_seconds=time.perf_counter()-started))
            if check['refine_score']:
                gradient, warm, diagnostic = evaluate(beta, warm,
                    error_budget=check['next_error_budget'], reason='refine_certificate')
                continue
            if check['certified']:
                # The full-marker score and its propagated error already
                # certify the current coefficients. Validation independently
                # verifies the associated dual residual before prediction.
                counters['certificate_reuses'] += 1
                break
            if iteration == max_iterations:
                raise RuntimeError(f'Residual Lasso failed KKT at path index {path_index}')
            outside = np.abs(gradient).copy()
            outside[indices] = -np.inf
            eligible = np.flatnonzero(outside > lam+max(0., tolerance-diagnostic['gradient_error_bound']))
            batch = (2 if not len(indices) else 1)*max(1, int(marker_batch))
            # Partition the top scores, then sort that prefix with the same
            # marker-index tie break as the former full stable sort.
            if len(eligible) > batch:
                weights = outside[eligible]
                cutoff = np.partition(weights, len(weights)-batch)[len(weights)-batch]
                above = eligible[weights > cutoff]
                ties = eligible[weights == cutoff][:batch-len(above)]
                eligible = np.concatenate((above, ties))
            fresh = eligible[np.lexsort((eligible, -outside[eligible]))]
            if len(fresh):
                tick = time.perf_counter()
                host_new = grm_index.extract_standardized_columns(fresh)
                block_id = len(host_blocks)
                host_blocks.append(host_new)
                with cache.hold(block_id) as new:
                    crosses = []
                    for previous in range(block_id):
                        with cache.hold(previous) as old:
                            crosses.append(np.asarray(_matmul(old, new,
                                transpose_a=True, workspace_limit=workspace_bytes)))
                    cross = np.concatenate(crosses, axis=0) if crosses else np.empty((0, len(fresh)))
                    block = np.asarray(_matmul(new, new, transpose_a=True,
                        workspace_limit=workspace_bytes))
                    if geometry is not None:
                        uz = np.asarray(_matmul(u_basis, new,
                            transpose_a=True, workspace_limit=workspace_bytes))
                        cz = geometry.covar_features(np.asarray(jnp.asarray(c).T@new), uz)
                if geometry is not None:
                    cross = geometry.block(cross, u_features, uz, c_features, cz)
                    block = geometry.block(block, uz, uz, cz, cz)
                    u_features = np.concatenate([u_features, uz], axis=1)
                    c_features = np.concatenate([c_features, cz], axis=1)
                else:
                    cross, block = cross/sigma, block/sigma
                metric = expand_bfgs_metric(metric, cross, block, pairs, indices, fresh)
                block_of_marker[fresh] = block_id
                column_of_marker[fresh] = np.arange(len(fresh), dtype=np.int32)
                indices = np.concatenate([indices, fresh])
                beta = np.pad(beta, (0, len(fresh)))
                counters['metric_seconds'] += time.perf_counter()-tick
            tick = time.perf_counter()
            cd_start = beta
            metric_beta = metric@beta
            cd_product = metric_beta
            if prior_path is not None and path_index < len(prior_path) and not prior_attempted:
                prior_attempted = True
                mapped = old_column[indices]
                known = mapped >= 0
                seed = np.zeros_like(beta)
                seed[known] = prior_path[path_index, mapped[known]]
                direction = seed-beta
                change = (.5*float(direction@(metric@direction))
                          -float(gradient[indices]@direction)
                          +lam*float(np.abs(seed).sum()-np.abs(beta).sum()))
                if np.isfinite(change) and change < 0:
                    cd_start = seed
                    cd_product = None
                    warm_rows_used.add(path_index)
            proposal, _, _, passed = solve_lasso_cd_gram(metric,
                metric_beta+gradient[indices], lam, beta0=cd_start, _Qb0=cd_product,
                max_iter=path_cfg.max_cd_iter, tol=min(path_cfg.cd_tol, 1e-8),
                kkt_abs_tol=.1*path_cfg.kkt_abs_tol,
                kkt_rel_tol=.1*path_cfg.kkt_rel_tol)
            if not passed:
                raise RuntimeError('Residual Lasso quadratic subproblem failed KKT')
            counters['subproblem_seconds'] += time.perf_counter()-tick
            proposed_gradient, proposed_warm, proposed_diag = evaluate(proposal, warm)
            direction = proposal-beta
            change = gradient-proposed_gradient
            for _ in range(8):
                step_size = quadratic_l1_step(beta, direction, gradient[indices],
                    direction@change[indices], lam)
                delta = (-step_size*gradient[indices]@direction
                    +.5*step_size**2*(direction@change[indices])
                    +lam*(np.abs(beta+step_size*direction).sum()-np.abs(beta).sum()))
                uncertainty = np.abs(direction).sum()*(step_size*diagnostic['gradient_error_bound']
                    +.5*step_size**2*(diagnostic['gradient_error_bound']+proposed_diag['gradient_error_bound']))
                if step_size > 0 and delta+uncertainty < 0:
                    break
                accuracy = .1*min(diagnostic['gradient_error_bound'], proposed_diag['gradient_error_bound'])
                gradient, warm, diagnostic = evaluate(beta, warm, error_budget=accuracy, reason="line_search_current")
                proposed_gradient, proposed_warm, proposed_diag = evaluate(
                    proposal, proposed_warm, error_budget=accuracy, reason="line_search_proposal")
                change = gradient-proposed_gradient
            else:
                raise RuntimeError('Residual Lasso line search cannot certify descent')
            step = step_size*direction
            rounding_bound = 0.
            if step_size == 1.:
                beta, gradient, warm = proposal, proposed_gradient, proposed_warm
            else:
                rounding_bound = affine_gradient_roundoff(
                    beta, proposal, gradient, proposed_gradient,
                    1.-step_size, step_size, column_norm_bound**2/sigma)
                beta = (1.-step_size)*beta+step_size*proposal
                gradient = (1.-step_size)*gradient+step_size*proposed_gradient
                warm = (1.-step_size)*warm+step_size*proposed_warm
            update_started = time.perf_counter()
            update = bfgs_update(metric, step, step_size*change[indices])
            counters['bfgs_update_seconds'] += time.perf_counter()-update_started
            if update is not None:
                pairs.append(dict(step=step.copy(), change=step_size*change.copy(),
                    product=update[0], denominator=update[1], target=update[2]))
            diagnostic = dict(proposed_diag, gradient_error_bound=(1-step_size)*diagnostic['gradient_error_bound']
                +step_size*proposed_diag['gradient_error_bound']+rounding_bound)
        else:
            # Refining an uncertain score also consumes the iteration budget.
            # In particular, the last refinement must not fall through and
            # label an unchecked path point as converged.
            raise RuntimeError(f'Residual Lasso failed KKT at path index {path_index}')
        active = np.flatnonzero(beta != 0.)
        selected = indices[active]
        fixed_train = fitted(beta)
        dual = adjoint(warm) if ridged else warm
        gamma = sla.solve_triangular(rc, np.asarray(qc.T@(yj-fixed_train)-hqc.T@dual),
            check_finite=False).reshape(-1) if c.shape[1] else np.empty(0)
        row = dict(path_index=path_index, k=len(active), lam=lam,
            lam_ratio=(lam/lam_max if lam_max > 0 else path_cfg.fixed_lam_ratio),
            converged=True, kkt_passed=True, global_kkt_passed=True,
            kkt_tolerance=tolerance, max_active_kkt_error=check['active_error'],
            max_inactive_kkt_excess=check['inactive_error'],
            global_kkt_max_outside_score=lam+check['inactive_error'],
            score_error_bound=diagnostic['gradient_error_bound'], iterations=iteration)
        rows.append(row)
        solutions.append((selected.copy(), beta[active].copy()))
        gammas.append(gamma)
        logger.info('[residual Lasso timing] point=%s/%s workset=%s metric=%.2fs '
                    'subproblem=%.2fs bfgs=%.2fs covariance=%.2fs fitted=%.2fs',
                    path_index+1, len(lambdas), len(indices), counters['metric_seconds'],
                    counters['subproblem_seconds'], counters['bfgs_update_seconds'],
                    counters['covariance_solve_seconds'],
                    counters['fitted_seconds'])
        if fixed:
            logger.info('[residual Lasso] point=%s/%s support=%s KKT=%.3g/%.3g time=%.1fs',
                        path_index+1, len(lambdas), len(active), check['upper_bound'],
                        tolerance, time.perf_counter()-started)
            continue
        tick = time.perf_counter()
        prediction_reuse = ({} if validation_requires_training_design else
                            dict(training_fixed_score_path=np.asarray(fixed_train)))
        evaluated = validation_evaluator(candidate=selected,
            train_candidate=(cached_design(selected) if validation_requires_training_design else None),
            beta_cov_path=gamma[None, :], beta_snp_path=beta[active][None, :],
            hinv_residual_path=np.asarray(dual), training_score_path=gradient[:, None],
            **prediction_reuse)
        counters['validation_seconds'] += time.perf_counter()-tick
        counters['validation_pcg_residual'] = max(counters['validation_pcg_residual'], evaluated['pcg_rel_res'])
        counters['validation_pcg_iterations'] = max(counters['validation_pcg_iterations'], evaluated['pcg_iters'])
        metrics.append(dict(evaluated['metrics'][0], path_index=path_index))
        stopping = stopping_rule(metrics)
        label='AIC' if 'aic' in metrics[-1] else 'validation'
        value=metrics[-1]['aic'] if label=='AIC' else metrics[-1]['predictive_r2']
        logger.info('[residual Lasso] point=%s/%s support=%s KKT=%.3g/%.3g %s=%.8f time=%.1fs',
            path_index+1, len(lambdas), len(active), check['upper_bound'], tolerance,
            label, value, time.perf_counter()-started)
        if stopping['stopped']:
            break
        if path_index >= 1:
            previous_state = old_state
        previous_lam = lam
    final_candidate = np.unique(np.concatenate([item[0] for item in solutions]))
    beta_path = np.zeros((len(rows), len(final_candidate)))
    for i, (markers, coefficients) in enumerate(solutions):
        beta_path[i, np.searchsorted(final_candidate, markers)] = coefficients
    counters['recycling_rank'] = recycling.rank
    counters['recycling_dependent'] = recycling.dependent
    counters['recycling_memory_skips'] = recycling.memory_skips
    counters['device_column_cache'] = cache.metrics()
    cache.clear()
    counters['total_seconds'] = time.perf_counter()-started
    metadata = dict(algorithm='residual_proximal_quasi_newton_global_kkt', n_iterations=len(trace),
        base_marker_batch_size=marker_batch, lambda_block_size=1,
        ever_active_size=len(final_candidate), path_solve_seconds=counters['total_seconds']-counters['validation_seconds'],
        global_kkt_seconds=0., validation_seconds=counters['validation_seconds'],
        validation_trace=metrics, trace=trace, performance=counters)
    # Device blocks/curvature are local to this call and released before REML.
    return dict(lasso_path=dict(path=rows, beta_snp_path=beta_path,
        beta_cov_path=np.asarray(gammas), lam_max=lam_max, selected_index=None,
        selection_method=None, selected_lam_ratio=None,
        path_role=('frozen_ratio_target_only' if fixed else
                   'kkt_certified_training_aic_prefix_residual_lasso'
                   if metrics and 'aic' in metrics[-1]
                   else 'kkt_certified_validation_prefix_residual_lasso'),
        requested_n_lambda=len(lambdas), evaluated_n_lambda=len(rows),
        validation_metrics=metrics, validation_early_stopping=stopping,
        external_beta_path_warm_start_used=bool(warm_rows_used),
        external_beta_path_warm_start_rows_used=len(warm_rows_used),
        basil=metadata), candidate=final_candidate,
        geno=cached_design(final_candidate) if materialize_result_design else None,
        extract_geno=None if materialize_result_design else cached_design,
        hinv_geno=None,
        hinv_z_dict={}, kkt_trace=trace,
        candidate_pcg=dict(n_columns=0, n_batches=0, batch_size=0,
            max_true_relative_residual=counters['max_pcg_residual'],
            max_reported_relative_residual=counters['max_pcg_residual'],
            max_iterations=counters['max_pcg_iterations'], total_batch_iterations=0),
        global_kkt=dict(passed=True, n_path_points=len(rows), n_violating_path_points=0,
            n_union_violators=0, max_outside_excess_over_threshold=0.,
            score_backend='residual_pcg_full_marker_xtv'),
        validation=dict(metrics=metrics, pcg_max_rel_res=counters['validation_pcg_residual'],
            pcg_max_iters=counters['validation_pcg_iterations'], seconds=counters['validation_seconds'],
            early_stopping=stopping, trace=metrics))
