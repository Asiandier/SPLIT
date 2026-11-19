"""Matrix-free covariance solves and Gaussian draws for sparse h2 uncertainty."""
from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from .sparse_information import solve_sparse_pcg
from .sparse_heritability_se import Backend
from .pcg import pcg_solve
from .sparse_memory import available_device_bytes, device_allocation_failed
from .sparse_trace import ProjectedTrace

logger = logging.getLogger(__name__)


class SelectedContrasts(ProjectedTrace):
    """Actions of the GLS projector and selected-mean inverse information."""

    def __init__(self, basis, covar, bank):
        super().__init__(basis, covar)
        self.bank = bank
        self.device_lift_weights = jnp.asarray(self.lift_weights, dtype=jnp.float64)

    def lift(self, value):
        return self.b64 @ value-self.f64 @ (self.device_lift_weights @ value)

    def coordinates(self, value):
        return self.b64.T @ value-self.device_lift_weights.T @ (self.f64.T @ value)

    def solve(self, value):
        return self.project(self.bank.solve(None, self.project(value), contrast=self.project))

    def condition_bound(self, covariance_norm, noise_variance):
        # B' P_C B is bounded below by B'(I-Q_C Q_C')B / ||H||.
        # Its smallest compression eigenvalue equals the smaller of one
        # and lambda_min(Q_C'(I-BB')Q_C), up to projector roundoff.
        gram = np.asarray(self.f64.T @ self.f64)
        gap = min(1., float(np.linalg.eigvalsh(gram)[0])) if gram.size else 1.
        return (max(1, self.b64.shape[1])*covariance_norm/(noise_variance*gap)
                if gap > 0 else float('inf'))

    def inverse_information(self, directions):
        """Return P_C B M^-1 directions and M^-1 directions."""
        u = self.lift(directions)
        w = u-self.solve(self.bank.hv(u))
        return w, self.coordinates(self.bank.hv(w))

    def unpenalized(self, value):
        dual = self.solve(value)
        residual = self.bank.hv(dual)
        return self.coordinates(value-residual), residual, dual


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
        self.solve_batch_size = batch_size
        self.allocation_retries = 0
        self.max_solve_residual = 0.
        self.solve_batches = 0
        self.rhs_columns = 0
        self.projected_rhs_columns = 0
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

    def selected_contrasts(self, basis, covar):
        return SelectedContrasts(basis, covar, self)

    def verify_selected_workspace(self, rank, *, projected):
        device = getattr(self.streamer, 'dev', None)
        budget = getattr(self.fitter.cfg, 'gpu_budget_bytes', None)
        if device is None or device.platform != 'gpu' or budget is None:
            return
        available = available_device_bytes(device, budget)
        # These arrays coexist even before including operator scratch.
        required = (12*self.n*rank if projected
                    else 24*self.n*rank+8*rank*rank)
        if available is not None and required > available:
            method = 'Projected' if projected else 'Inverse-column'
            raise MemoryError(f'{method} SE needs at least {required/1024**3:.2f} GiB '
                              f'for selected arrays, with {available/1024**3:.2f} GiB available')

    def _verified_solve(self, hv, precond, block, initial):
        value, residual, iterations = solve_sparse_pcg(
            hv, block, M=precond, tol=self.tol, maxiter=self.maxiter,
            log_diagnostics=False, X0=initial)
        if self.tol >= np.sqrt(np.finfo(np.float32).eps):
            # Loose tolerances still verify against the FP64 covariance.
            value = jnp.asarray(value, dtype=jnp.float64)
            precise_block = jnp.asarray(block, dtype=jnp.float64)
            norm = jnp.linalg.norm(precise_block, axis=0)
            residual = float(jnp.max(jnp.linalg.norm(precise_block-hv(value), axis=0)
                                     / jnp.where(norm > 0, norm, 1.)))
        remaining = self.maxiter-int(iterations)
        if (np.isfinite(residual) and residual > self.tol*1.05 and remaining > 0):
            value, residual, extra = pcg_solve(
                hv, jnp.asarray(block, dtype=jnp.float64), M=precond,
                tol=self.tol, maxiter=remaining, X0=jnp.asarray(value, dtype=jnp.float64))
            iterations += int(extra)
            self.fp64_rescue_batches += 1
            self.fp64_rescue_iterations += int(extra)
        if not np.isfinite(residual) or residual > self.tol * 1.05:
            raise FloatingPointError(f"SE covariance solve failed: residual={residual:g}, tol={self.tol:g}")
        return jnp.asarray(value, dtype=self.b.dtype), residual, iterations

    def solve(self, factor, rhs, *, initial=None, contrast=None):
        if contrast is None:
            hv, precond = self.hv, self.precond
        else:
            def hv(value):
                projected = contrast(value)
                return contrast(self.hv(projected))+(value-projected)

            if self.precond is not None:
                def precond(value):
                    projected = contrast(value)
                    return contrast(self.precond(projected))+(value-projected)
            else:
                precond = None

        vector = rhs.ndim == 1
        target = rhs[:, None] if vector else rhs
        if initial is not None:
            initial = initial[:, None] if initial.ndim == 1 else initial
            if initial.shape != target.shape:
                raise ValueError("Cached SE solve has incompatible dimensions")
        parts = []
        start = 0
        while start < target.shape[1]:
            width = min(self.solve_batch_size, target.shape[1]-start)
            stop = start+width
            try:
                value, residual, iterations = self._verified_solve(
                    hv, precond, target[:, start:stop],
                    None if initial is None else initial[:, start:stop])
            except (RuntimeError, MemoryError) as error:
                if width == 1 or not device_allocation_failed(error):
                    raise
                self.solve_batch_size = max(1, width//2)
                self.allocation_retries += 1
                logger.warning('[h2 SE solve] reduce solve width %s -> %s after allocation failure',
                               width, self.solve_batch_size)
                continue
            self.max_solve_residual = max(self.max_solve_residual, float(residual))
            self.solve_batches += 1
            self.rhs_columns += width
            if contrast is not None:
                self.projected_rhs_columns += width
            self.pcg_iterations += int(iterations)
            self.warm_solve_batches += int(initial is not None)
            parts.append(value)
            start = stop
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
