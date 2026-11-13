"""Selected-mean information correction, without sample-by-sample matrices.

The Lasso mean is an offset, NOT an additional profiled fixed effect. For a
fixed selected span Z, add log|Z' P_C Z| to the negative REML determinant.
Endpoint sparse variance subtracts tr((B' P_C B)^-1)/n. A projected
covariance trace estimator avoids one inverse solve per selected column.
"""
from __future__ import annotations

import logging
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

from .pcg import pcg_solve

logger = logging.getLogger(__name__)


def solve_sparse_pcg(hv, rhs, M=None, tol=1e-2, maxiter=200, X0=None,
                     check_every=2, *, log_diagnostics=True,
                     initial_operator_product=None):
    """Use FP32 PCG, with FP64 residual correction below its accuracy floor.

    Every correction solves H delta = b-Hx for the SAME H. The total PCG
    iteration budget and the final requested residual tolerance are unchanged.
    Ordinary REML and its PCG implementation do not use this helper.

    ``initial_operator_product`` may supply a previously evaluated FP64
    ``hv(X0)`` for a tight solve. The caller must own the matching immutable
    warm start and operator. It replaces only the first residual product;
    every corrected solution is verified with ``hv`` as usual.
    """
    inner_tol = max(float(tol), math.sqrt(float(tol)))
    tight = tol < math.sqrt(np.finfo(np.float32).eps)
    if initial_operator_product is not None and (
        not tight
        or getattr(X0, "dtype", None) != np.float64
        or getattr(initial_operator_product, "dtype", None) != np.float64
        or X0.shape != rhs.shape
        or initial_operator_product.shape != rhs.shape
    ):
        raise ValueError("Cached operator product requires a matching FP64 warm start and tight solve")
    if tight and getattr(X0, "dtype", None) == np.float64:
        # Accepted REML states already carry a precise solution. Do not round
        # away that work before checking an unchanged or nearby covariance.
        solution, used = X0, 0
    else:
        block = jnp.asarray(rhs, dtype=jnp.float32)
        initial = None if X0 is None else jnp.asarray(X0, dtype=jnp.float32)
        solution, relative, iterations = pcg_solve(
            hv, block, M=M, tol=inner_tol if tight else tol,
            maxiter=maxiter, X0=initial, check_every=check_every,
            stop_on_residual_gap=tight,
        )
        used = int(iterations)
        if not tight:
            return solution, float(relative), used
        # Verify the original system in FP64 immediately. Continuing to polish
        # the FP32 system cannot remove its operator/solution rounding error.
    with jax.experimental.enable_x64():
        target = jnp.asarray(rhs, dtype=jnp.float64)
        value = jnp.asarray(solution, dtype=jnp.float64)
        scale = jnp.linalg.norm(target, axis=0)
        nonzero = scale > 0
        value = jnp.where(nonzero, value, 0.)
        scale = jnp.where(nonzero, scale, 1.)
        previous = math.inf
        corrections = 0
        cached_product = initial_operator_product
        if cached_product is not None:
            # The solver zeros warm-start columns whose new RHS is zero.
            cached_product = jnp.where(nonzero, cached_product, 0.)
        while True:
            residual = target-(hv(value) if cached_product is None else cached_product)
            cached_product = None
            actual = float(jax.device_get(jnp.max(
                jnp.linalg.norm(residual, axis=0)/scale
            )))
            if not math.isfinite(actual):
                raise FloatingPointError("Non-finite FP64 information residual.")
            if actual <= tol or used >= maxiter or actual >= previous:
                if log_diagnostics:
                    logger.info(
                        "[information PCG] FP64 residual correction: columns=%s "
                        "corrections=%s iterations=%s/%s true_residual=%.3e tol=%.3e",
                        rhs.shape[1], corrections, used, maxiter, actual, tol,
                    )
                return value, actual, used
            previous = actual
            # Solving H delta = r to relative error eta gives the original
            # system a residual bounded by eta * actual in exact arithmetic.
            # The next FP64 check, not this prediction, controls acceptance.
            correction_tol = min(.5, max(np.finfo(np.float32).eps,
                                         .5 * tol / actual))
            correction, _, count = pcg_solve(
                hv, residual.astype(jnp.float32), M=M,
                tol=correction_tol, maxiter=maxiter-used, check_every=check_every,
                stop_on_residual_gap=True,
            )
            used += int(count)
            if not count:
                return value, actual, used
            value = value+correction.astype(jnp.float64)
            corrections += 1


def solve_columns(hv, rhs, precond, *, tol, maxiter, batch_size=128, warm=None):
    """Bound streamed operator temporaries and check the true solve residual."""
    rhs = np.asarray(rhs)
    if rhs.ndim != 2 or batch_size < 1:
        raise ValueError("Information solves require matrix RHS and positive batch size.")
    result = np.empty(rhs.shape, dtype=np.float64)
    for start in range(0, rhs.shape[1], batch_size):
        started = time.perf_counter()
        end = min(start + batch_size, rhs.shape[1])
        solution, actual, iterations = solve_sparse_pcg(
            hv, rhs[:, start:end], M=precond, tol=tol, maxiter=maxiter,
            X0=None if warm is None else warm[:, start:end],
        )
        if not np.isfinite(actual) or actual > tol * 1.05:
            raise FloatingPointError(
                f"Sparse information PCG failed: true residual={actual:.3e}, "
                f"tolerance={tol:.3e}, "
                f"iterations={int(iterations)}/{maxiter}."
            )
        result[:, start:end] = np.asarray(solution)
        logger.info("[information solve] columns=%s:%s iterations=%s residual=%.3e time=%.2fs",
                    start, end, iterations, actual, time.perf_counter()-started)
    return result


class SparseMeanInformation:
    """One immutable sample/design/support scope, with one covariance cache.

    A new object is required after an active-set or sample/transform change.
    Only a warm solution survives a theta change; no old information factor
    or trace does. Rank reduction removes numerical null directions only.
    """

    def __init__(self, active_x, covar, *, hinv_active=None, batch_size=128):
        started = time.perf_counter()
        x = np.asarray(active_x, dtype=np.float64)
        self.covar = (np.empty((x.shape[0], 0), dtype=np.float64) if covar is None
                      else np.array(covar, dtype=np.float64, copy=True))
        if self.covar.ndim == 1:
            self.covar = self.covar[:, None]
        if x.ndim != 2 or self.covar.shape[0] != x.shape[0]:
            raise ValueError("Sparse information design/sample mismatch.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(self.covar)):
            raise ValueError("Nonfinite sparse information design.")
        self.n = x.shape[0]
        self.batch_size = int(batch_size)
        self.hinv_warm = None
        if x.shape[1]:
            # QR first reduces the SVD to a support-sized matrix for n >> |S|.
            q, r = sla.qr(x, mode="economic", check_finite=False)
            u, singular, vt = sla.svd(r, full_matrices=False, check_finite=False)
            cutoff = max(x.shape) * np.finfo(np.float64).eps * singular[0]
            ranks = [int(np.sum(singular > a * cutoff)) for a in (0.1, 1., 10.)]
            if len(set(ranks)) != 1:
                raise ValueError("Numerically ambiguous sparse mean-space rank.")
            self.rank = ranks[1]
            self.basis = q @ u[:, :self.rank]
            if hinv_active is not None:
                transform = vt[:self.rank].T / singular[:self.rank]
                self.hinv_warm = np.asarray(hinv_active @ transform, dtype=np.float32)
        else:
            self.rank = 0
            self.basis = np.empty((self.n, 0), dtype=np.float64)
        if self.n - self.covar.shape[1] - self.rank <= 0:
            raise ValueError("No residual contrast dimension after selected mean fitting.")
        if self.rank and self.covar.shape[1]:
            qc, _ = sla.qr(self.covar, mode="economic", check_finite=False)
            overlap = sla.svdvals(qc.T @ self.basis, check_finite=False)[0]
            if overlap >= 1 - 1e-10:
                raise ValueError("Sparse mean span intersects ordinary covariates.")
        self._theta = None
        self._tol = None
        self._state = None
        self._endpoint_theta = None
        self._endpoint_state = None
        self._endpoint_tol = None
        logger.info("[information basis] selected=%s rank=%s time=%.2fs",
                    x.shape[1], self.rank, time.perf_counter()-started)

    def _cached(self, theta, tol):
        return (self._theta is not None and np.array_equal(theta, self._theta)
                and self._tol <= tol)

    def evaluate(self, theta, hv, precond, hinv_covar, *, tol, maxiter,
                 _solved_basis=None):
        """Return logdet, tr(Sigma_g), and W with P-P_S = W W'."""
        theta = np.asarray(theta, dtype=np.float64)
        if self._cached(theta, tol):
            return self._state
        hc = np.empty((self.n, 0), dtype=np.float64)
        cf = None
        if self.covar.shape[1]:
            hc = np.array(hinv_covar, dtype=np.float64, copy=True)
            if hc.shape != self.covar.shape or not np.all(np.isfinite(hc)):
                raise ValueError("Information covariance solve does not match C.")
            cg = self.covar.T @ hc
            cf = sla.cho_factor((cg+cg.T)*.5, lower=True, check_finite=False)
        if not self.rank:
            state = {"logdet": 0., "trace": 0., "w": np.empty((self.n, 0)),
                     "rank": 0}
        else:
            started = time.perf_counter()
            solved = _solved_basis
            if solved is None:
                solved = solve_columns(hv, self.basis, precond, tol=tol, maxiter=maxiter,
                                       batch_size=self.batch_size, warm=self.hinv_warm)
            solve_seconds = time.perf_counter()-started
            started = time.perf_counter()
            self.hinv_warm = solved
            pz = np.asarray(solved, dtype=np.float64)
            if self.covar.shape[1]:
                pz = pz - hc @ sla.cho_solve(cf, self.covar.T @ pz, check_finite=False)
            info = self.basis.T @ pz
            factor = sla.cholesky((info+info.T)*.5, lower=True, check_finite=False)
            inverse_factor = sla.solve_triangular(factor, np.eye(self.rank), lower=True,
                                                 check_finite=False)
            state = {
                "logdet": float(2*np.log(np.diag(factor)).sum()),
                "trace": float(np.sum(inverse_factor**2)),
                "w": sla.solve_triangular(factor, pz.T, lower=True,
                                          check_finite=False).T,
                "rank": self.rank,
            }
            logger.info("[information evaluation] rank=%s solves=%.2fs projection_and_factor=%.2fs",
                        self.rank, solve_seconds, time.perf_counter()-started)
        # Keep verified covariance solves available to optional endpoint SE.
        state.update(hinv_covar=hc, covar_factor=cf)
        self._theta, self._state, self._tol = theta.copy(), state, tol
        return state

    def trace_statistics(self, theta, hv, precond, *, tol, maxiter, exact=False):
        """Endpoint trace cache, separate from the REML likelihood cache."""
        from .sparse_trace import MIN_PROBES, estimate_trace
        theta = np.asarray(theta, dtype=float)
        if (theta.ndim != 1 or not np.isfinite(theta).all()
                or np.any(theta[:-1] < 0) or theta[-1] <= 0):
            raise ValueError('Selected-mean trace requires a PSD genetic covariance and positive noise')
        if self._cached(theta, tol):
            state = dict(trace=self._state['trace'], rank=self.rank,
                         endpoint_trace_correction='exact', trace_probes=0,
                         trace_mc_se=0., trace_solve_error_bound=None, trace_pcg_tolerance=tol)
        elif (not exact and self._endpoint_state is not None
              and np.array_equal(theta, self._endpoint_theta) and self._endpoint_tol <= tol):
            return self._endpoint_state
        else:
            state = None
            if self.rank > MIN_PROBES and not exact:
                state = estimate_trace(self.basis, self.covar, theta, hv, precond,
                    tol=tol, maxiter=maxiter, batch_size=self.batch_size)
            if state is None:
                hc = solve_columns(hv, self.covar, precond, tol=tol, maxiter=maxiter,
                                   batch_size=self.batch_size)
                direct = self.evaluate(theta, hv, precond, hc, tol=tol, maxiter=maxiter)
                state = dict(trace=direct['trace'], rank=self.rank,
                             endpoint_trace_correction='exact', trace_probes=0,
                             trace_mc_se=0., trace_solve_error_bound=None, trace_pcg_tolerance=tol)
        self._endpoint_theta, self._endpoint_state = theta.copy(), state
        self._endpoint_tol = tol
        return state

    def statistics(self, theta, hv, precond, residual, *, tol, maxiter, exact=False,
                   sparse_mean=None, background_variance=None, previous_h2=None,
                   h2_tolerance=None):
        """GLS profile and selected-mean correction at the same covariance.

        The stochastic endpoint trace never replaces the information
        determinant or score in evaluate(). The response profile is always
        recomputed: it changes after a Lasso update even if theta is unchanged.
        """
        started = time.perf_counter()
        profiled, gamma = self.profile_response(
            hv, precond, residual, tol=tol, maxiter=maxiter)
        state = self.trace_statistics(theta, hv, precond, tol=tol, maxiter=maxiter, exact=exact)
        if sparse_mean is not None and state['endpoint_trace_correction'] != 'exact':
            g = np.asarray(sparse_mean, dtype=float)
            q0 = float(g @ g+2*g @ profiled)/self.n
            uncertainty = (2*state['trace_mc_se']+state['trace_solve_error_bound'])/self.n
            q = q0-state['trace']/self.n
            total = q+float(background_variance)+float(theta[-1])
            sensitive = abs(q) <= uncertainty or total <= uncertainty
            if total > uncertainty:
                h2 = 1-float(theta[-1])/total
                h2_error = float(theta[-1])*uncertainty/(total*(total-uncertainty))
                if previous_h2 is not None and h2_tolerance is not None:
                    # Include uncertainty in both compared endpoints. This is
                    # an empirical MC guard, not a probabilistic guarantee.
                    sensitive |= abs(abs(h2-previous_h2)-h2_tolerance) <= 2*h2_error
            if sensitive:
                state = self.trace_statistics(theta, hv, precond, tol=tol, maxiter=maxiter, exact=True)
                state['exact_reason'] = 'near_variance_or_convergence_boundary'
        logger.info("[sparse endpoint] rank=%s trace=%g method=%s probes=%s mc_se=%.3g "
                    "solve_bound=%s time=%.2fs", self.rank, state['trace'],
                    state['endpoint_trace_correction'], state['trace_probes'],
                    state['trace_mc_se'], state['trace_solve_error_bound'], time.perf_counter()-started)
        return state, profiled, gamma

    def profile_response(self, hv, precond, residual, *, tol, maxiter):
        """Profile C with one contrast solve, without computing selected trace.

        For Pi=I-Q_C Q_C', P_C=Pi(Pi H Pi+Q_C Q_C')^-1 Pi.
        H P_C r = r-C gamma then recovers the same GLS covariate estimate.
        Intermediate, convergence and final-output profiles use this same
        GLS calculation. It does not request a selected-mean variance trace.
        """
        if not self.covar.shape[1]:
            return np.asarray(residual), np.empty(0)
        q, factor = sla.qr(self.covar, mode="economic", check_finite=False)
        response = np.asarray(residual, dtype=np.float32).reshape(-1, 1)
        with jax.experimental.enable_x64():
            q64 = jnp.asarray(q, dtype=jnp.float64)
            q32 = q64.astype(jnp.float32)
            def project(value):
                basis = q32 if value.dtype == jnp.float32 else q64
                return value-basis @ (basis.T @ value)
            def contrast(value):
                projected = project(value)
                return project(hv(projected))+(value-projected)
            def precondition(value):
                projected = project(value)
                return project(precond(projected))+(value-projected)
            target = jnp.asarray(response, dtype=jnp.float64)
            dual, actual, _ = solve_sparse_pcg(
                contrast, project(target), M=None if precond is None else precondition,
                tol=tol, maxiter=maxiter,
            )
            if not np.isfinite(actual) or actual > tol*1.05:
                raise FloatingPointError("Covariate contrast profile failed its true residual check")
            nuisance = target-hv(project(dual))
            gamma = sla.solve_triangular(
                factor, np.asarray(q64.T @ nuisance)[:, 0], lower=False,
                check_finite=False,
            )
        return np.asarray(residual)-self.covar @ gamma, gamma

    def score_trace_correction(self, state, kernel_mvs, residual_diags=None):
        """tr(W' D_j W), with bounded kernel products and float64 reductions."""
        cached = state.get("score_trace_cache")
        kernels = tuple(kernel_mvs)
        if (cached is not None and len(cached[0]) == len(kernels)
                and all(a is b for a, b in zip(cached[0], kernels, strict=True))
                and cached[1] is residual_diags):
            return cached[2].copy()
        n_residual = 1 if residual_diags is None else len(residual_diags)
        traces = np.zeros(len(kernel_mvs)+n_residual, dtype=np.float64)
        w = state["w"]
        started = time.perf_counter()
        for start in range(0, self.rank, self.batch_size):
            block = np.asarray(w[:, start:start+self.batch_size], dtype=np.float32)
            device_block = jnp.asarray(block)
            for k, mv in enumerate(kernel_mvs):
                product = np.asarray(mv(device_block), dtype=np.float64)
                traces[k] += np.einsum("ij,ij->", block, product)
            square = np.sum(np.asarray(block, dtype=np.float64)**2, axis=1)
            if residual_diags is None:
                traces[-1] += square.sum()
            else:
                traces[len(kernel_mvs):] += np.asarray(residual_diags) @ square
        state["score_trace_cache"] = (kernels, residual_diags, traces.copy())
        logger.info("[information trace] rank=%s components=%s time=%.2fs",
                    self.rank, len(kernels), time.perf_counter()-started)
        return traces
