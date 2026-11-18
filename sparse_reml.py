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

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

from .precond import (
    ProjectedCorePrecondConf, build_projected_core_runtime,
    projected_core_apply_invsqrt, projected_core_logdet,
    scalar_diag_from_precond_conf,
)
from .slq import logdet_value_and_grad
from .sparse_planner import projected_slq_workspace

logger = logging.getLogger(__name__)


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
        self.covariate_q64 = self.covariate_q32 = None
        self._context_ref = None
        self.reference = None
        self.reference_logdet = 0.0
        self.inverse_lower = 1.0
        self.logdet_design = 0.0

    def _ensure_basis(self):
        if self.q64 is not None:
            return
        started = time.perf_counter()
        # B is already orthonormal. Only C's component outside span(B)
        # needs QR: [B,Q_perp] spans [C,B] and |D'D|=|C'(I-BB')C|.
        basis, covar = self.exact.basis, self.exact.covar
        remaining = covar-basis @ (basis.T @ covar)
        # Reorthogonalize the small covariate block, not the tall SNP basis.
        remaining -= basis @ (basis.T @ remaining)
        extra, r = sla.qr(remaining, mode="economic", check_finite=False)
        q = np.column_stack([basis, extra])
        qc, _ = sla.qr(covar, mode="economic", check_finite=False)
        diagonal = np.abs(np.diag(r))
        if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
            raise ValueError("Sparse projected likelihood requires full-rank nuisance and selected spans")
        self.logdet_design = float(2.0 * np.log(diagonal).sum())
        with jax.experimental.enable_x64():
            self.q64 = jnp.asarray(q, dtype=jnp.float64)
            self.q32 = self.q64.astype(jnp.float32)
            self.covariate_q64 = jnp.asarray(qc, dtype=jnp.float64)
            self.covariate_q32 = self.covariate_q64.astype(jnp.float32)
        logger.info("[projected information] contrasts=%s basis_setup=%.2fs",
                    q.shape[1], time.perf_counter()-started)

    def project(self, value):
        self._ensure_basis()
        q = self.q32 if value.dtype == jnp.float32 else self.q64
        return value - q @ (q.T @ value)

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
            u, change = sla.qr(np.asarray(self.project(conf.U.astype(jnp.float64))),
                               mode="economic", check_finite=False)
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
                - self.q64.shape[1] * jnp.log(self.reference.d)
            )
            # A conservative spectral bound allowing imperfect stored-basis
            # orthogonality. A nonpositive bound disables quadrature early exit.
            combined = jnp.concatenate([self.reference.U, self.q32], axis=1)
            gram = combined.T @ combined
            norm_bound = 1.01 * float(jnp.max(jnp.sum(jnp.abs(gram), axis=1)))
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
        q = self.q32 if value.dtype == jnp.float32 else self.q64
        return (projected_core_apply_invsqrt(self.reference, value)
                + q @ ((1.0-jnp.sqrt(self.reference.d_inv)) * (q.T @ value)))

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
            synchronize_batches=self.slq_workspace_bytes is not None,
        )
        logger.info("[projected information] joint determinant=%.6f time=%.2fs depths=%s",
                    float(value+constant), time.perf_counter()-started,
                    diagnostics.get("batch_depths"))
        return value+constant, gradient


def with_projected_likelihood(information, *, available_bytes=None, precond_rank=0,
                              slq_workspace_bytes=None, slq_samples=None, slq_depth=None):
    """Use the identity only when its additional basis fits the sparse plan."""
    required = information.n * (
        12 * (information.rank + 2 * information.covar.shape[1]) + 16 * precond_rank
    )
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
