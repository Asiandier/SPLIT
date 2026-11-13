"""Selected-mean covariance trace from a small number of contrast solves.

For an orthonormal selected basis B, estimate tr((B' P_C B)^-1) using
the covariance Schur complement, without forming a sample covariance matrix
or solving V^-1 for every selected column. See docs/sparse_numerics.md.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla


TRACE_SEED = 20261112
MIN_PROBES = 16
MAX_PROBES = 128
TRACE_RTOL = 5e-3


def trace_probes(rank, count, seed=TRACE_SEED):
    """A reproducible prefix, independent of batching and covariance updates."""
    rng = np.random.default_rng(seed)
    return (2*rng.integers(0, 2, size=(count, rank), dtype=np.int8)-1).T.astype(float)


class ProjectedTrace:
    """Temporary device geometry for one immutable selected/nuisance span.

    B is the already verified orthonormal basis from SparseMeanInformation.
    No n-by-n matrix is stored. The caller releases this geometry after each
    endpoint so it cannot overlap the next REML likelihood's device buffers.
    """

    def __init__(self, basis, covar):
        if covar.shape[1]:
            qc, _ = sla.qr(covar, mode='economic', check_finite=False)
            w = basis.T @ qc
            f = qc-basis @ w
            factor = sla.cho_factor(f.T @ f, lower=True, check_finite=False)
            qf = sla.solve_triangular(factor[0], f.T, lower=True, check_finite=False).T
            self.lift_weights = sla.cho_solve(factor, w.T, check_finite=False)
        else:
            f = qf = np.empty((basis.shape[0], 0))
            self.lift_weights = np.empty((0, basis.shape[1]))
        with jax.experimental.enable_x64():
            self.b64 = jnp.asarray(basis, dtype=jnp.float64)
            self.b32 = self.b64.astype(jnp.float32)
            self.f64 = jnp.asarray(f, dtype=jnp.float64)
            self.q64 = jnp.asarray(qf, dtype=jnp.float64)
            self.q32 = self.q64.astype(jnp.float32)

    def project(self, x):
        b, q = (self.b32, self.q32) if x.dtype == jnp.float32 else (self.b64, self.q64)
        return x-b @ (b.T @ x)-q @ (q.T @ x)

    def quadratics(self, probes, hv, precond, residual_variance, *, tol, maxiter):
        """Return quadratic upper estimates and certified linear-solve errors.

        The error bound is deterministic in exact arithmetic; it does NOT
        bound Monte Carlo error. FP64 residual products verify coarse solves.
        """
        from .sparse_information import solve_sparse_pcg

        def apply(x):
            px = self.project(x)
            return self.project(hv(px))+(x-px)

        def apply_pre(x):
            px = self.project(x)
            return self.project(precond(px))+(x-px)

        with jax.experimental.enable_x64():
            u = self.b64 @ jnp.asarray(probes, dtype=jnp.float64)
            u -= self.f64 @ jnp.asarray(self.lift_weights @ probes, dtype=jnp.float64)
            vu = hv(u)
            rhs = self.project(vu)
            solution, _, iterations = solve_sparse_pcg(
                apply, rhs, M=None if precond is None else apply_pre,
                tol=tol, maxiter=maxiter, log_diagnostics=False)
            solution = jnp.asarray(solution, dtype=jnp.float64)
            ax = apply(solution)
            residual = rhs-ax
            upper = (jnp.sum(u*vu, axis=0)-2*jnp.sum(rhs*solution, axis=0)
                     + jnp.sum(solution*ax, axis=0))
            error = jnp.sum(residual*residual, axis=0)/min(float(residual_variance), 1.)
            norm = jnp.linalg.norm(rhs, axis=0)
            relative = jnp.linalg.norm(residual, axis=0)/jnp.where(norm > 0, norm, 1.)
            upper, error, relative = [np.asarray(a, dtype=float) for a in (upper, error, relative)]
        if not all(np.isfinite(a).all() for a in (upper, error, relative)):
            raise FloatingPointError('Nonfinite selected-mean trace quadratic')
        return upper, error, relative, int(iterations)


def estimate_trace(basis, covar, theta, hv, precond, *, tol, maxiter, batch_size):
    """Estimate T, refining probes/linear solves or requesting exact fallback.

    The numerical target is 0.5% of the correction (two empirical MC SEs
    plus the solve bound). It is not a confidence guarantee or a model prior.
    Small supports use the direct calculation in the caller. A failed
    precision check never silently disables the correction.
    """
    started = time.perf_counter()
    geometry = ProjectedTrace(basis, covar)
    rank = basis.shape[1]
    probes = trace_probes(rank, MAX_PROBES)
    # Four RHS fit the packed FP32/FP64 kernels; use wider native GEMM when
    # the packed backend is not active. This changes blocking only.
    from . import sparse_packed
    narrow = sparse_packed.supports(jax.ShapeDtypeStruct((basis.shape[0], 4), jnp.float32))
    width = max(1, min(int(batch_size), 4 if narrow else MIN_PROBES))
    uppers, errors, residuals = [], [], []
    iterations = 0
    count = 0
    for target in (MIN_PROBES, 32, 64, MAX_PROBES):
        for start in range(count, target, width):
            block = probes[:, start:min(start+width, target)]
            solve_tol = 1e-2
            while True:
                upper, error, relative, used = geometry.quadratics(
                    block, hv, precond, theta[-1], tol=solve_tol, maxiter=maxiter)
                iterations += used
                # Refine based on the quadratic error, not a loose PCG flag.
                scale = max(float(upper.mean()), rank*float(theta[-1]))
                if float(error.mean()) <= .1*TRACE_RTOL*scale:
                    break
                if solve_tol <= min(tol, 1e-5):
                    return None
                solve_tol = max(min(tol, 1e-5), solve_tol*.1)
            uppers.extend(upper.tolist())
            errors.extend(error.tolist())
            residuals.extend(relative.tolist())
        count = target
        value = float(np.mean(uppers))
        bound = float(np.mean(errors))
        mcse = float(np.std(uppers, ddof=1)/np.sqrt(count))
        if value > 0 and 2*mcse+bound <= TRACE_RTOL*value:
            return dict(trace=value, rank=rank, endpoint_trace_correction='projected_hutchinson',
                        trace_probes=count, trace_seed=TRACE_SEED, trace_mc_se=mcse,
                        trace_solve_error_bound=bound, trace_relative_target=TRACE_RTOL,
                        trace_max_residual=max(residuals), trace_batch_size=width,
                        trace_pcg_iterations=iterations, trace_seconds=time.perf_counter()-started)
    return None
