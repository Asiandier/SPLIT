"""Matrix-free covariance solves and Gaussian draws for sparse h2 uncertainty."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .sparse_information import solve_sparse_pcg
from .sparse_heritability_se import Backend
from .pcg import pcg_solve


class OperatorKernelBank:
    """Use exactly the fitted streamer's kernels, sample order and scaling.

    Covariance draws use X_g z_g / sqrt(m_eff,g), plus independent residual
    noise. Only small batches of marker weights are allocated; no n-by-n
    kernels, factorization or new phenotype fits are needed.
    """

    def __init__(self, fitter, grm_index, *, tol=1e-5, maxiter=400, batch_size=32):
        self.b = Backend(gpu=True)
        self.b.dtype = self.b.xp.float64
        self.index = grm_index
        self.streamer = grm_index.streamer
        self.n = int(self.streamer.n)
        self.fitter = fitter
        self.ops = fitter._assemble_reml_operators()
        self.k = self.ops.K_mvs
        self.tol, self.maxiter, self.batch_size = tol, maxiter, batch_size
        self.max_solve_residual = 0.
        self.solve_batches = 0
        self.pcg_iterations = 0
        self.warm_solve_batches = 0
        self.fp64_rescue_batches = 0
        self.fp64_rescue_iterations = 0
        if len(self.k) != grm_index.n_grm:
            raise ValueError("SE kernel count differs from fitted component partition")
        if getattr(self.streamer, "has_component_partition", False):
            effective = np.asarray(self.streamer._component_eff_m_host, dtype=float)
        else:
            effective = np.asarray([float(np.asarray(self.streamer._eff_m_const))])
        if effective.shape != (len(self.k),) or np.any(effective < 0):
            raise ValueError("Invalid effective SNP counts for SE covariance draws")
        self.effective = effective

    def mv(self, i, value):
        return self.k[i](value) if i < len(self.k) else value

    def factor(self, theta):
        self.theta = np.asarray(theta, dtype=float)
        precise_hv = self.fitter._make_hv(
            self.ops, jnp.asarray(theta[:-1], dtype=jnp.float64),
            jnp.asarray(theta[-1], dtype=jnp.float64))
        correction_hv = self.fitter._make_hv(
            self.ops, jnp.asarray(theta[:-1], dtype=jnp.float32),
            jnp.asarray(theta[-1], dtype=jnp.float32))
        # Match the accumulator precision to the current solve. Capturing
        # FP64 theta in the correction operator would also promote Hv to
        # FP64, even with an FP32 input and preconditioner.
        self.hv = lambda value: (correction_hv(value) if value.dtype == jnp.float32
                                 else precise_hv(value))
        # The preconditioner is approximate; keep its native FP32 factors.
        # FP64 factors would promote the FP32 PCG correction directions and
        # turn every subsequent genotype product into an FP64 calculation.
        # The covariance above and the final true-residual check remain FP64.
        approximate = self.fitter._make_effect_precond(
            self.ops, jnp.asarray(theta[:-1], dtype=jnp.float32),
            jnp.asarray(theta[-1], dtype=jnp.float32))
        self.precond = (None if approximate is None else
                        lambda value: approximate(value).astype(value.dtype))
        scales = np.sqrt(np.divide(self.theta[:-1], self.effective,
                                  out=np.zeros_like(self.effective), where=self.effective > 0))
        self.marker_scale = np.repeat(scales, self.index.m_per_grm)
        return None  # The covariance is represented by hv, not a dense factor.

    def solve(self, factor, rhs, *, initial=None):
        vector = rhs.ndim == 1
        target = rhs[:, None] if vector else rhs
        if initial is not None:
            initial = initial[:, None] if initial.ndim == 1 else initial
            if initial.shape != target.shape:
                raise ValueError("Cached SE solve has incompatible dimensions")
        parts = []
        for start in range(0, target.shape[1], self.batch_size):
            block = target[:, start:start+self.batch_size]
            value, residual, iterations = solve_sparse_pcg(
                self.hv, block, M=self.precond,
                tol=self.tol, maxiter=self.maxiter, log_diagnostics=False,
                X0=None if initial is None else initial[:, start:start+self.batch_size])
            if self.tol >= np.sqrt(np.finfo(np.float32).eps):
                # The shared helper permits FP32-only solves at loose tolerances.
                # SE still checks the original FP64 covariance in this case.
                value = jnp.asarray(value, dtype=jnp.float64)
                precise_block = jnp.asarray(block, dtype=jnp.float64)
                norm = jnp.linalg.norm(precise_block, axis=0)
                residual = float(jnp.max(jnp.linalg.norm(precise_block-self.hv(value), axis=0)
                                         / jnp.where(norm > 0, norm, 1.)))
            remaining = self.maxiter-int(iterations)
            if (np.isfinite(residual) and residual > self.tol*1.05 and remaining > 0):
                # A poorly conditioned FP32 correction can stagnate. Spend only
                # the remaining iteration budget on the same FP64 system.
                value, residual, extra = pcg_solve(
                    self.hv, jnp.asarray(block, dtype=jnp.float64), M=self.precond,
                    tol=self.tol, maxiter=remaining, X0=jnp.asarray(value, dtype=jnp.float64))
                iterations += int(extra)
                self.fp64_rescue_batches += 1
                self.fp64_rescue_iterations += int(extra)
            if not np.isfinite(residual) or residual > self.tol * 1.05:
                raise FloatingPointError(f"SE covariance solve failed: residual={residual:g}, tol={self.tol:g}")
            self.max_solve_residual = max(self.max_solve_residual, float(residual))
            self.solve_batches += 1
            self.pcg_iterations += int(iterations)
            self.warm_solve_batches += int(initial is not None)
            parts.append(jnp.asarray(value, dtype=self.b.dtype))
        result = jnp.concatenate(parts, axis=1) if parts else jnp.empty(target.shape, dtype=self.b.dtype)
        return result[:, 0] if vector else result

    def draw(self, factor, rng, count):
        noise = np.sqrt(self.theta[-1]) * rng.normal(size=(self.n, count))
        parts = []
        for start in range(0, count, self.batch_size):
            width = min(self.batch_size, count-start)
            weights = rng.standard_normal((self.index.m_total, width), dtype=np.float32)
            weights *= self.marker_scale[:, None].astype(np.float32)
            genetic = self.streamer.xv(jnp.asarray(weights))
            parts.append(jnp.asarray(noise[:, start:start+width]) + genetic)
        return jnp.concatenate(parts, axis=1)

    def inverse_draw(self, factor, rng, count):
        # Cov(H^-1 xi)=H^-1 for xi~N(0,H); projecting gives Cov(psi)=P_S.
        return self.solve(factor, self.draw(factor, rng, count))

    def resolve_anchor(self, py):
        scores = self.index.xtv_all(py, normalize=False)
        index = int(np.argmax(np.abs(scores)))
        anchor = self.index.extract_standardized_columns(np.asarray([index]))[:, 0]
        return anchor, float(np.sign(scores[index])), index
