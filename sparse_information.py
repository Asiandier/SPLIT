"""Selected-mean information correction, without sample-by-sample matrices.

The Lasso mean is an offset, NOT an additional profiled fixed effect. For a
fixed selected span Z, add log|Z' P_C Z| to the negative REML determinant and
subtract tr{Z (Z' P_C Z)^-1 Z'}/n from the calibrated sparse quadratic.
"""
from __future__ import annotations

import logging
import math

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

from .pcg import pcg_solve

logger = logging.getLogger(__name__)


def _solve_information_block(hv, rhs, precond, *, tol, maxiter, warm):
    """Use FP32 PCG, with FP64 residual correction below its accuracy floor.

    Every correction solves H delta = b-Hx for the SAME H. The total PCG
    iteration budget and the final requested residual tolerance are unchanged.
    Ordinary REML and its PCG implementation do not use this helper.
    """
    block = jnp.asarray(rhs, dtype=jnp.float32)
    initial = None if warm is None else jnp.asarray(warm, dtype=jnp.float32)
    inner_tol = max(float(tol), math.sqrt(float(tol)))
    tight = tol < math.sqrt(np.finfo(np.float32).eps)
    solution, relative, iterations = pcg_solve(
        hv, block, M=precond, tol=inner_tol if tight else tol,
        maxiter=maxiter, X0=initial,
    )
    used = int(iterations)
    if not tight or float(relative) <= tol:
        return np.asarray(solution), float(relative), used
    # In a well-conditioned system, tightening by the same log factor should
    # take about the same number of steps. Keep that cheap FP32 fast path;
    # avoid spending the entire budget repeatedly chasing a rounding floor.
    polish_budget = min(maxiter-used, max(1, used))
    if polish_budget:
        solution, relative, count = pcg_solve(
            hv, block, M=precond, tol=tol, maxiter=polish_budget, X0=solution,
        )
        used += int(count)
        if float(relative) <= tol:
            return np.asarray(solution), float(relative), used
    with jax.experimental.enable_x64():
        target = jnp.asarray(rhs, dtype=jnp.float64)
        value = jnp.asarray(solution, dtype=jnp.float64)
        scale = jnp.linalg.norm(target, axis=0)
        scale = jnp.where(scale > 0, scale, 1.)
        previous = math.inf
        corrections = 0
        while True:
            residual = target-hv(value)
            actual = float(jax.device_get(jnp.max(
                jnp.linalg.norm(residual, axis=0)/scale
            )))
            if not math.isfinite(actual):
                raise FloatingPointError("Non-finite FP64 information residual.")
            if actual <= tol or used >= maxiter or actual >= previous:
                logger.info(
                    "[information PCG] FP64 residual correction: columns=%s "
                    "corrections=%s iterations=%s/%s true_residual=%.3e tol=%.3e",
                    rhs.shape[1], corrections, used, maxiter, actual, tol,
                )
                return np.asarray(value), actual, used
            previous = actual
            correction, _, count = pcg_solve(
                hv, residual.astype(jnp.float32), M=precond,
                tol=inner_tol, maxiter=maxiter-used,
            )
            used += int(count)
            if not count:
                return np.asarray(value), actual, used
            value = value+correction.astype(jnp.float64)
            corrections += 1


def solve_columns(hv, rhs, precond, *, tol, maxiter, batch_size=128, warm=None):
    """Bound streamed operator temporaries and check the true solve residual."""
    rhs = np.asarray(rhs)
    if rhs.ndim != 2 or batch_size < 1:
        raise ValueError("Information solves require matrix RHS and positive batch size.")
    result = np.empty(rhs.shape, dtype=np.float64)
    for start in range(0, rhs.shape[1], batch_size):
        end = min(start + batch_size, rhs.shape[1])
        solution, actual, iterations = _solve_information_block(
            hv, rhs[:, start:end], precond, tol=tol, maxiter=maxiter,
            warm=None if warm is None else warm[:, start:end],
        )
        if not np.isfinite(actual) or actual > tol * 1.05:
            raise FloatingPointError(
                f"Sparse information PCG failed: true residual={actual:.3e}, "
                f"tolerance={tol:.3e}, "
                f"iterations={int(iterations)}/{maxiter}."
            )
        result[:, start:end] = np.asarray(solution)
    return result


class SparseMeanInformation:
    """One immutable sample/design/support scope, with one covariance cache.

    A new object is required after an active-set or sample/transform change.
    Only a warm solution survives a theta change; no old information factor
    or trace does. Rank reduction removes numerical null directions only.
    """

    def __init__(self, active_x, covar, *, hinv_active=None, batch_size=128):
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
        self._response_cache = None

    def _cached(self, theta, tol):
        return (self._theta is not None and np.array_equal(theta, self._theta)
                and self._tol <= tol)

    def evaluate(self, theta, hv, precond, hinv_covar, *, tol, maxiter):
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
            solved = solve_columns(hv, self.basis, precond, tol=tol, maxiter=maxiter,
                                   batch_size=self.batch_size, warm=self.hinv_warm)
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
        # The REML block already solved C at this theta. Keep that exact factor
        # for the subsequent paired h2 calculation, rather than solving C again.
        state.update(hinv_covar=hc, covar_factor=cf)
        self._theta, self._state, self._tol = theta.copy(), state, tol
        return state

    def statistics(self, theta, hv, precond, residual, *, tol, maxiter):
        """Profile C at this covariance and return the paired trace correction."""
        theta = np.asarray(theta, dtype=np.float64)
        response = np.asarray(residual, dtype=np.float32).reshape(-1)
        cached = self._response_cache
        reuse_response = (cached is not None and cached[1] <= tol
                          and np.array_equal(cached[0], theta)
                          and np.array_equal(cached[2], response))
        state = self._state if self._cached(theta, tol) else None
        response_tol = tol
        if state is None:
            rhs = np.column_stack([response, self.covar])
            solved = solve_columns(hv, rhs, precond, tol=tol, maxiter=maxiter,
                                   batch_size=self.batch_size)
            solved_response = solved[:, 0]
            state = self.evaluate(theta, hv, precond, solved[:, 1:], tol=tol, maxiter=maxiter)
        elif reuse_response:
            solved_response = cached[3]
            response_tol = cached[1]
        else:
            solved_response = solve_columns(
                hv, response[:, None], precond, tol=tol, maxiter=maxiter,
                batch_size=self.batch_size,
            )[:, 0]
        self._response_cache = (theta.copy(), response_tol, response.copy(), solved_response.copy())
        if self.covar.shape[1]:
            gamma = sla.cho_solve(state["covar_factor"], self.covar.T @ solved_response,
                                 check_finite=False)
        else:
            gamma = np.empty(0)
        return state, np.asarray(residual)-self.covar @ gamma, gamma

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
        return traces
