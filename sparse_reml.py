"""Equivalent projected likelihood for the sparse variance update.

For D=[C,B]=QR, Pi=I-QQ', the determinant identity is
  log|H| + log|C'H^-1 C| + log|B'P_C B| = log|Pi H Pi+QQ'| + log|D'D|.
Only the determinant uses D. The quadratic and AI retain P_C. Endpoint
heritability profiles ordinary covariates and separately estimates the
selected-mean covariance trace; that trace is not part of this determinant.
"""
from __future__ import annotations

import logging
import time
import weakref
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

from . import sparse_fp64
from .precond import (
    ProjectedCorePrecondConf, build_projected_core_runtime,
    projected_core_apply_invsqrt, projected_core_logdet,
    scalar_diag_from_precond_conf,
)
from .slq import logdet_value_and_grad, workspace_layout
from .sparse_memory import available_device_bytes
from .sparse_planner import projected_slq_workspace

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=("bounded_gemm",))
def _combined_gram_norm(u, q, *, bounded_gemm=False):
    """Infinity norm of [U,Q]'[U,Q], without a second copy of [U,Q]."""
    def gram(a, b):
        if bounded_gemm:
            return sparse_fp64.matmul(a, b, transpose_a=True)
        return a.T @ b

    cross = jnp.abs(gram(u, q))
    left = jnp.sum(jnp.abs(gram(u, u)), axis=1) + jnp.sum(cross, axis=1)
    right = jnp.sum(jnp.abs(gram(q, q)), axis=1) + jnp.sum(cross, axis=0)
    return jnp.maximum(jnp.max(left, initial=0), jnp.max(right, initial=0))


@jax.jit
def _cuda_span_product(q, value, scale):
    # Explicit GEMM transposes let cuBLAS use the same resident basis in
    # both directions without a layout copy or an autotuning workspace.
    vector = value.ndim == 1
    block = value[:, None] if vector else value
    coefficients = sparse_fp64.matmul(q, block, transpose_a=True)
    product = sparse_fp64.matmul(q, scale * coefficients)
    return product[:, 0] if vector else product


class SparseProjectedInformation:
    """One selected span; fit-only device caches are explicitly releasable."""

    def __init__(self, information, *, slq_workspace_bytes=None):
        self.exact = information
        self.slq_workspace_bytes = slq_workspace_bytes
        self.n = information.n
        self.rank = information.rank
        self.release_likelihood_cache()

    def __getattr__(self, name):
        return getattr(self.exact, name)

    def release_likelihood_cache(self):
        # Do not retain a REMLContext -> information -> REMLContext cycle, or
        # overlap the projection basis with wide endpoint information solves.
        self.q64 = self.q32 = None
        self._extra = None
        self._bounded_gemm = False
        self.covariate_q64 = self.covariate_q32 = None
        self._context_ref = None
        self.reference = None
        self.reference_logdet = 0.0
        self.inverse_lower = 1.0
        self.logdet_design = 0.0

    def _ensure_basis(self):
        if self.q32 is not None:
            return
        started = time.perf_counter()
        # B is already orthonormal. Only C's component outside span(B)
        # needs QR: [B,Q_perp] spans [C,B] and |D'D|=|C'(I-BB')C|.
        basis, covar = self.exact.basis, self.exact.covar
        remaining = covar-basis @ (basis.T @ covar)
        # Reorthogonalize the small covariate block, not the tall SNP basis.
        remaining -= basis @ (basis.T @ remaining)
        extra, r = sla.qr(remaining, mode="economic", check_finite=False)
        qc, _ = sla.qr(covar, mode="economic", check_finite=False)
        diagonal = np.abs(np.diag(r))
        if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
            raise ValueError("Sparse projected likelihood requires full-rank nuisance and selected spans")
        self.logdet_design = float(2.0 * np.log(diagonal).sum())
        self._extra = extra
        # The Lanczos and adjoint operations in REML use FP32. Preserve the
        # existing FP64 basis on the host for the precise reference projection.
        q = np.empty((self.n, self.rank+extra.shape[1]), dtype=np.float32)
        q[:, :self.rank] = basis
        q[:, self.rank:] = extra
        with jax.experimental.enable_x64():
            self.q32 = jnp.asarray(q).block_until_ready()
            self.covariate_q64 = jnp.asarray(qc, dtype=jnp.float64)
            self.covariate_q32 = self.covariate_q64.astype(jnp.float32)
        device = next(iter(self.q32.devices()))
        if device.platform == "gpu":
            with jax.default_device(device):
                self._bounded_gemm = sparse_fp64.initialize()
        logger.info("[projected information] contrasts=%s basis_setup=%.2fs",
                    q.shape[1], time.perf_counter()-started)

    def _host_span_product(self, value, scale):
        basis = self.exact.basis
        result = basis @ (scale * (basis.T @ value))
        result += self._extra @ (scale * (self._extra.T @ value))
        return result

    def _span_product(self, value, scale=None):
        self._ensure_basis()
        if value.dtype == jnp.float32:
            q = self.q32
            if self._bounded_gemm:
                factor = jnp.asarray(1. if scale is None else scale, dtype=value.dtype)
                return _cuda_span_product(q, value, factor)
        else:
            device = next(iter(self.q32.devices()))
            if device.platform == "gpu":
                factor = 1. if scale is None else float(scale)
                result = self._host_span_product(np.asarray(value), factor)
                with jax.experimental.enable_x64():
                    return jax.device_put(result, device)
            if self.q64 is None:
                with jax.experimental.enable_x64():
                    self.q64 = jax.device_put(
                        np.column_stack([self.exact.basis, self._extra]), device)
            q = self.q64
        projected = q.T @ value
        return q @ (projected if scale is None else scale * projected)

    def project(self, value):
        return value - self._span_product(value)

    def covariate_project(self, value):
        self._ensure_basis()
        q = self.covariate_q32 if value.dtype == jnp.float32 else self.covariate_q64
        return value - q @ (q.T @ value)

    def _prepare_reference(self, ctx, parameters):
        if self._context_ref is not None and self._context_ref() is ctx:
            return
        if ctx.E != 1 or ctx.residual_diag_stack is not None:
            raise ValueError("Sparse projected likelihood requires scalar residual variance")
        self._ensure_basis()
        started = time.perf_counter()
        conf = ctx.precond_conf
        self.reference = None
        self.reference_logdet = 0.0
        self.inverse_lower = 1.0
        if isinstance(conf, ProjectedCorePrecondConf) and conf.total_rank > 0:
            diagonal = scalar_diag_from_precond_conf(conf, parameters[:ctx.G], parameters[ctx.G:])
            core = jnp.tensordot(parameters[:ctx.G], conf.core_atoms, axes=1)
            if next(iter(conf.U.devices())).platform == "gpu":
                host = np.asarray(conf.U, dtype=np.float64)
                projected_u = host-self._host_span_product(host, 1.)
            else:
                projected_u = np.asarray(self.project(conf.U.astype(jnp.float64)))
            u, change = sla.qr(projected_u, mode="economic", check_finite=False)
            projected_core = change @ np.asarray(core) @ change.T
            projected_conf = ProjectedCorePrecondConf(
                U=jnp.asarray(u, dtype=jnp.float32),
                core_atoms=jnp.asarray(projected_core[None], dtype=jnp.float32),
                total_rank=u.shape[1], n_grm=1, diag_mode="scalar_identity",
            )
            self.reference = build_projected_core_runtime(
                projected_conf, jnp.ones(1, dtype=jnp.float32), diagonal, need_invsqrt=True,
            )
            self.reference_logdet = float(
                projected_core_logdet(self.reference, self.n)
                - self.q32.shape[1] * jnp.log(self.reference.d)
            )
            # A conservative spectral bound allowing imperfect stored-basis
            # orthogonality. A nonpositive bound disables quadrature early exit.
            norm_bound = 1.01 * float(_combined_gram_norm(
                self.reference.U, self.q32, bounded_gemm=self._bounded_gemm))
            base = float(self.reference.d) ** -0.5
            negative = min(0.0, float(jnp.max(self.reference.eigvals)) ** -0.5 - base, 1.0-base)
            self.inverse_lower = 0.9 * max(0.0, base + negative * norm_bound) ** 2
        # Freeze the reference for this entire fit; PCG may refresh separately.
        self._context_ref = weakref.ref(ctx)
        logger.info("[projected information] fixed reference setup=%.2fs",
                    time.perf_counter()-started)

    def whiten(self, value):
        if self.reference is None:
            return value
        return (projected_core_apply_invsqrt(self.reference, value)
                + self._span_product(value, scale=1.0-jnp.sqrt(self.reference.d_inv)))

    def joint_logdet_and_score(self, ctx, hv, parameters, key, *, nsamples, m,
                              accept_logdet=None):
        self._prepare_reference(ctx, parameters)
        started = time.perf_counter()

        def apply(value):
            left = self.whiten(value)
            projected = self.project(left)
            return self.whiten(self.project(hv(projected)) + (left-projected))

        def pullback(left, right):
            original_left = self.whiten(left)
            left = self.project(original_left)
            right = self.project(self.whiten(right))
            result = parameters[ctx.G] * left
            gradient = []
            for k, operator in enumerate(ctx.K_mvs):
                product = operator(left)
                result = result + parameters[k] * product
                gradient.append(jnp.sum(right * product))
            gradient.append(jnp.sum(right * left))
            result = self.project(result) + (original_left-left)
            return self.whiten(result), jnp.stack(gradient)

        constant = self.reference_logdet + self.logdet_design
        lower = 0.9 * min(float(parameters[ctx.G]), 1.0) * self.inverse_lower
        diagnostics = {}
        value, gradient = logdet_value_and_grad(
            apply, pullback, self.n, key, nsamples=nsamples, m=m,
            workspace_bytes=(self.slq_workspace_bytes or ctx.slq_workspace_bytes),
            dtype=parameters.dtype,
            accept_value=(None if accept_logdet is None else
                          lambda value: accept_logdet(value+constant)),
            lower_bound=lower if lower > 0 else None,
            quadrature_tolerance=ctx.slq_quadrature_tolerance, diagnostics=diagnostics,
            synchronize_batches=next(iter(self.q32.devices())).platform == "gpu",
        )
        logger.info("[projected information] joint determinant=%.6f time=%.2fs depths=%s",
                    float(value+constant), time.perf_counter()-started,
                    diagnostics.get("batch_depths"))
        return value+constant, gradient


def with_projected_likelihood(information, *, available_bytes=None, precond_rank=0,
                              slq_workspace_bytes=None, slq_samples=None, slq_depth=None,
                              device=None, gpu_budget_bytes=None):
    """Use the identity only when its additional basis fits the sparse plan."""
    q = information.rank + information.covar.shape[1]
    required = information.n * (4*q + 12*information.covar.shape[1] + 16*precond_rank)
    if precond_rank:
        # Include the simultaneous Gram block and reduction scratch at setup.
        required += 8 * (q+precond_rank)**2
    if device is not None and device.platform == "gpu":
        with jax.default_device(device):
            bounded_gemm = sparse_fp64.initialize()
        if not bounded_gemm:
            # XLA's native dot may use two additional full-sized layout or
            # tuning buffers even though the resident projector is FP32.
            required += 8 * information.n * q
        live = available_device_bytes(device, gpu_budget_bytes)
        if live is not None and slq_workspace_bytes is not None:
            slq_peak = workspace_layout(
                information.n, slq_samples, slq_depth, slq_workspace_bytes
            )["peak_bytes"]
            live_headroom = live-slq_peak
            available_bytes = (live_headroom if available_bytes is None
                               else min(available_bytes, live_headroom))
    if available_bytes is None or required <= available_bytes:
        return SparseProjectedInformation(information)
    if slq_workspace_bytes is not None:
        workspace = projected_slq_workspace(
            information.n, slq_samples, slq_depth, workspace_bytes=slq_workspace_bytes,
            available_bytes=available_bytes, basis_bytes=required,
        )
        if workspace is not None:
            logger.info("[projected information] resize SLQ workspace %.2f -> %.2f GiB "
                        "to retain the projection basis (%.2f GiB)",
                        slq_workspace_bytes/1024**3, workspace/1024**3, required/1024**3)
            return SparseProjectedInformation(information, slq_workspace_bytes=workspace)
    logger.info("[projected information] retain direct information solves: "
                "additional basis needs %.2f GiB; planned headroom %.2f GiB",
                required/1024**3, max(0, available_bytes)/1024**3)
    return information
