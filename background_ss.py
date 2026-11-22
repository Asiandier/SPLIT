"""Constrained SS background refinement of finalized sparse prediction outputs.

The context owns reusable genotype products and trace quadrature. All supplied
phenotypes, designs and coefficients must be on the model's training scale.
No foreground, covariance component or noise parameter is re-estimated.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
import time
import numpy as np
import jax
import jax.numpy as jnp
from .grouped_ss_vamp import (
    GenotypeProducts, VAMPConfig, VAMPResult, VAMPConvergenceError, fit_grouped_ss_vamp,
)
from .ss_linear import MatrixFreeSampleSolver
from .sparse_memory import available_device_bytes, device_allocation_failed
from .sparse_packed_stream import SparsePackedStream

logger = logging.getLogger(__name__)


class StreamedGenotypeProducts:
    """Native forward/transpose/kernel operators in logical SNP order.

    Custom index implementations may supply standardized columns instead;
    that fallback retains only one memory-bounded genotype block.
    """
    def __init__(self, index, n, normalizers, *, block_bytes=64*1024**2,
                 device_cache_bytes=0, accelerate_packed=False,
                 gpu_budget_bytes=None, trace_workspace_bytes=0,
                 min_trace_workspace_bytes=None):
        self.index, self.n, self.p = index, int(n), int(index.m_total)
        self.streamer = getattr(index, 'streamer', None)
        self.normalizers = np.asarray(normalizers, float)
        self.offsets = np.r_[0, np.cumsum(index.m_per_grm)]
        self.chunk_size = min(4096, max(1, block_bytes//max(16*self.n, 1)))
        self.packed_calls = None
        self._device_packed = None
        self._prefetch = None
        self.device_cache_bytes = 0
        self.trace_workspace_bytes = trace_workspace_bytes
        self.sparse_packed_enabled = False
        if (self.streamer is not None and getattr(self.streamer, '_packed_buf', None) is not None
                and getattr(self.streamer, '_means_host', None) is not None
                and getattr(self.streamer, '_inv_sds_host', None) is not None):
            # SS uses FP64 products. Subdivide the existing packed cache so a
            # call planned for the FP32 REML stage cannot create an oversized
            # FP64 decode workspace. Source order and training statistics stay
            # exactly those of the native streamer.
            max_width = max(4, self.chunk_size//4*4)
            source_width = int(self.streamer._max_unpack_width)
            tiles_per_call = (source_width + max_width - 1)//max_width
            # Balance full source calls over the same number of tiles to
            # avoid padding every call's final tile up to max_width.
            width = 4*((source_width + 4*tiles_per_call - 1)//(4*tiles_per_call))
            self.chunk_size = width
            calls = []
            for call, (start, size) in enumerate(zip(self.streamer._call_snp_starts,
                                                    self.streamer._call_true_widths)):
                for offset in range(0, int(size), width):
                    actual = min(width, int(size)-offset)
                    calls.append((call, offset//4, (offset+actual+3)//4,
                                  int(start)+offset, actual))
            self.packed_calls = calls
            self.packed_width = width
            means = np.zeros((len(calls), self.packed_width), np.float32)
            inverse = np.zeros_like(means)
            for k, (_, _, _, start, actual) in enumerate(calls):
                means[k, :actual] = self.streamer._means_host[start:start+actual]
                inverse[k, :actual] = self.streamer._inv_sds_host[start:start+actual]
            self.means = jax.device_put(means, self.streamer.dev)
            self.inverse = jax.device_put(inverse, self.streamer.dev)
            self.starts = jax.device_put(np.array([c[3] for c in calls], np.int32), self.streamer.dev)
            self.widths = jax.device_put(np.array([c[4] for c in calls], np.int32), self.streamer.dev)
            self._all_calls = np.arange(len(calls), dtype=np.int32)
            self._missing = jnp.asarray(self.streamer._missing_val, jnp.uint8)
            if accelerate_packed and getattr(self.streamer.dev, 'platform', None) == 'gpu':
                from . import sparse_packed
                with jax.default_device(self.streamer.dev):
                    self.sparse_packed_enabled = sparse_packed.initialize()
            if self.streamer.dev.platform == 'gpu':
                shape = (self.n, self.packed_width//4)
                size = int(np.prod(shape))
                self._prefetch = SparsePackedStream(
                    self.streamer.dev, self._packed_padded, [size]*len(calls),
                    budget_bytes=gpu_budget_bytes,
                )
                # Reserve two decode tiles, trace storage and outputs before
                # retaining any packed blocks. Driver/allocator usage also
                # includes the remaining native fit state.
                reserve = block_bytes + 16*(self.n+self.p)
                self._cache_prefix(shape, size, device_cache_bytes, reserve,
                                   min_trace_workspace_bytes)
            self._parameters = tuple(zip(self.widths, self.means, self.inverse, self.starts))

    def _cache_prefix(self, shape, size, limit, reserve, min_trace_workspace_bytes):
        from .kv_impl import _store_packed_call
        device = self.streamer.dev
        available = available_device_bytes(device, self._prefetch.budget_bytes)
        if available is None:
            return
        usable = available - reserve - self._prefetch.prefetch_bytes
        preferred_trace = self.trace_workspace_bytes
        blocks = len(self.packed_calls)
        limit = int(limit)
        count = min(blocks, max(0, min(limit, usable-preferred_trace)//size))
        preferred_count = count
        full_bytes = blocks*size
        trace_with_full_cache = usable-full_bytes
        if (count < blocks and full_bytes <= limit
                and min_trace_workspace_bytes is not None
                and min_trace_workspace_bytes <= trace_with_full_cache):
            # Trace quadrature is built once in batches. A fully resident
            # genotype panel avoids transfers during every later PCG solve.
            count = blocks
            self.trace_workspace_bytes = min(preferred_trace, trace_with_full_cache)

        def fill(count):
            with jax.default_device(device):
                initial = jnp.zeros((count, *shape), jnp.uint8)
                return self._prefetch.reduce(range(count), initial,
                    lambda cache, k, block: _store_packed_call(cache, block, jnp.asarray(k, jnp.int32)))

        while count:
            try:
                self._device_packed = fill(count)
            except (RuntimeError, MemoryError) as exc:
                if not device_allocation_failed(exc):
                    raise
                count = min(count//2, preferred_count)
                self.trace_workspace_bytes = preferred_trace
            else:
                self.device_cache_bytes = count*size
                break

    def close(self):
        self._device_packed = None
        if getattr(self, '_prefetch', None) is not None:
            self._prefetch.close()
            self._prefetch = None

    def __del__(self):
        self.close()

    def _selection(self, active):
        if np.all(active):
            return self._all_calls, self.inverse
        mask = np.zeros((len(self.packed_calls), self.packed_width), bool)
        for k, (_, _, _, start, actual) in enumerate(self.packed_calls):
            mask[k, :actual] = active[start:start+actual]
        return np.flatnonzero(np.any(mask, axis=1)), self.inverse*jax.device_put(mask, self.streamer.dev)

    def _split_calls(self, selected):
        end = 0 if self._device_packed is None else int(self._device_packed.shape[0])
        count = int(np.searchsorted(selected, end))
        return selected[:count], selected[count:]

    def restricted(self, active):
        """Restrict the linear products to active markers in logical order."""
        from types import SimpleNamespace
        active = np.asarray(active)
        if active.shape != (self.p,) or active.dtype != np.dtype(bool):
            raise ValueError('Active SNP mask must be Boolean in logical SNP order')
        if np.all(active):
            return self
        if self.packed_calls is not None:
            selected, inverse = self._selection(active)
            _, streamed = self._split_calls(selected)
            # The mask and metadata are fixed for this linear operator.
            # Slice once, outside repeated VAMP forward/transpose scans.
            streamed_inverse = {int(k): inverse[int(k)] for k in streamed}
            return SimpleNamespace(
                forward=lambda effects: self._packed_forward(effects, selected, inverse, streamed_inverse),
                transpose=lambda value: self._packed_transpose(value, selected, inverse, streamed_inverse))
        mask = jnp.asarray(active)
        def masked(value):
            return jnp.where(mask if value.ndim == 1 else mask[:, None], value, 0.)
        return SimpleNamespace(
            forward=lambda effects: self.forward(masked(effects)),
            transpose=lambda value: masked(self.transpose(value)))

    def _packed(self, k):
        call, first, last, _, _ = self.packed_calls[k]
        return self.streamer._packed_block_host(call)[:, first:last]

    def _packed_padded(self, k):
        block = self._packed(k)
        pad = self.packed_width//4-block.shape[1]
        return np.pad(block, ((0, 0), (0, pad))) if pad else block

    def _reduce(self, calls, state, step):
        from .kv_impl import reduce_packed_blocks
        return reduce_packed_blocks(calls, state, step,
            pop_block=self._packed, dev=self.streamer.dev, packed_stream=self._prefetch)

    def _blocks(self, active=None):
        for start in range(0, self.p, self.chunk_size):
            stop = min(start+self.chunk_size, self.p)
            if active is not None and not np.any(active[start:stop]):
                continue
            block = self.index.extract_standardized_columns(np.arange(start, stop))
            yield start, stop, jnp.asarray(block, jnp.float64)

    def _packed_forward(self, effects, selected, inverse, streamed_inverse=None):
        from .kv_impl import _xv_call_accumulate, packed_cache_forward
        effects = jnp.asarray(effects)
        squeeze = effects.ndim == 1
        if squeeze:
            effects = effects[:, None]
        cached, streamed = self._split_calls(selected)
        if cached.size:
            total = packed_cache_forward(self._device_packed, self.starts,
                self.widths, self.means, inverse, jnp.asarray(cached, jnp.int32), effects,
                self._missing, sparse_packed_enabled=self.sparse_packed_enabled)
        else:
            total = jnp.zeros((self.n, effects.shape[1]), effects.dtype)
        if streamed.size:
            def step(total, k, block):
                width, mean, inv, start = self._parameters[k]
                return _xv_call_accumulate(total, block, width, mean,
                    inv if streamed_inverse is None else streamed_inverse[k], effects, start,
                    self._missing, sparse_packed_enabled=self.sparse_packed_enabled)
            total = self._reduce(streamed, total, step)
        return total[:, 0] if squeeze else total

    def forward(self, effects):
        effects = jnp.asarray(effects)
        if self.packed_calls is not None:
            return self._packed_forward(effects, self._all_calls, self.inverse)
        if self.streamer is not None and hasattr(self.streamer, 'xv'):
            return self.streamer.xv(effects)
        total = jnp.zeros((self.n, effects.shape[1]), effects.dtype)
        for start, stop, block in self._blocks():
            total += jnp.matmul(block, effects[start:stop], precision=jax.lax.Precision.HIGHEST)
        return total

    def _packed_transpose(self, value, selected, inverse, streamed_inverse=None):
        from .kv_impl import _xtv_one_call_jit, _xtv_scatter_jit, packed_cache_transpose
        value = jnp.asarray(value)
        squeeze = value.ndim == 1
        if squeeze:
            value = value[:, None]
        cached, streamed = self._split_calls(selected)
        if cached.size:
            total = packed_cache_transpose(self._device_packed, self.starts,
                self.widths, self.means, inverse, jnp.asarray(cached, jnp.int32), value,
                self._missing, p=self.p, sparse_packed_enabled=self.sparse_packed_enabled)
            if streamed.size:
                total = jnp.pad(total, ((0, self.packed_width), (0, 0)))
        else:
            total = jnp.zeros((self.p+self.packed_width, value.shape[1]), value.dtype)
        if streamed.size:
            def step(total, k, block):
                width, mean, inv, start = self._parameters[k]
                product = _xtv_one_call_jit(block, width, mean,
                    inv if streamed_inverse is None else streamed_inverse[k], value, self._missing,
                    sparse_packed_enabled=self.sparse_packed_enabled)
                return _xtv_scatter_jit(total, product[:int(self.packed_calls[k][-1])], start)
            total = self._reduce(streamed, total, step)
        result = total[:self.p]
        return result[:, 0] if squeeze else result

    def transpose(self, residual):
        residual = jnp.asarray(residual)
        if self.packed_calls is not None:
            return self._packed_transpose(residual, self._all_calls, self.inverse)
        if self.streamer is not None:
            return self.streamer.xtv(residual, normalize=False)
        total = jnp.zeros((self.p, residual.shape[1]), residual.dtype)
        for start, stop, block in self._blocks():
            values = jnp.matmul(block.T, residual, precision=jax.lax.Precision.HIGHEST)
            total = total.at[start:stop].set(values)
        return total

    def _packed_gram_operator(self, active, *, with_transpose=False):
        """Scan a device-resident prefix, then continue on prefetched blocks."""
        if self.packed_calls is None:
            return None
        from .kv_impl import (
            _kv_one_call_jit, _kv_xtv_one_call_jit, _xtv_scatter_jit,
            packed_cache_gram, packed_cache_gram_transpose,
        )
        selected, inverse = self._selection(np.asarray(active, bool))
        cached, streamed = self._split_calls(selected)
        cached = jnp.asarray(cached, jnp.int32)
        inv_params = None if inverse is self.inverse else {int(k): inverse[int(k)] for k in streamed}
        accelerated = self.sparse_packed_enabled

        def packed_operator(value):
            squeeze = value.ndim == 1
            if squeeze:
                value = value[:, None]
            total = jnp.zeros_like(value)
            projection = None
            if cached.size:
                if with_transpose:
                    total, projection = packed_cache_gram_transpose(
                        self._device_packed, self.starts, self.widths, self.means, inverse,
                        cached, value, self._missing, p=self.p, sparse_packed_enabled=accelerated)
                    if streamed.size:
                        projection = jnp.pad(projection, ((0, self.packed_width), (0, 0)))
                else:
                    total = packed_cache_gram(self._device_packed, self.widths, self.means,
                        inverse, cached, value, self._missing, sparse_packed_enabled=accelerated)
            elif with_transpose:
                projection = jnp.zeros((self.p+self.packed_width, value.shape[1]), value.dtype)

            def step(state, k, packed):
                total, projection = state
                width, mean, inv, start = self._parameters[k]
                if inv_params is not None:
                    inv = inv_params[k]
                if with_transpose:
                    total, block = _kv_xtv_one_call_jit(packed, width, mean, inv, value,
                        self._missing, total, sparse_packed_enabled=accelerated)
                    projection = _xtv_scatter_jit(projection,
                        block[:int(self.packed_calls[k][-1])], start)
                else:
                    total = _kv_one_call_jit(packed, width, mean, inv, value,
                        self._missing, total, sparse_packed_enabled=accelerated)
                return total, projection

            total, projection = self._reduce(streamed, (total, projection), step)
            if with_transpose:
                return (total[:, 0], projection[:self.p, 0]) if squeeze else (total, projection[:self.p])
            return total[:, 0] if squeeze else total

        return packed_operator

    def gram_transpose_operator(self, active):
        """Return a shared packed scan, or None for custom non-packed backends.

        Both outputs use the same active mask and training standardization.
        Callers without this capability keep their existing separate products.
        """
        return self._packed_gram_operator(active, with_transpose=True)

    def gram_operator(self, active):
        active = np.asarray(active, bool)
        packed_operator = self._packed_gram_operator(active)
        if packed_operator is not None:
            return packed_operator
        if self.streamer is not None:
            if np.all(active):
                return lambda value:self.streamer.kv(value, normalize=False)
            group_mask = active[self.offsets[:-1]]
            if (bool(getattr(self.streamer, 'has_component_partition', False))
                    and np.array_equal(active, np.repeat(group_mask, np.diff(self.offsets)))):
                weights = jnp.asarray(self.normalizers*active[self.offsets[:-1]])
                return lambda value:self.streamer.weighted_component_hv(weights, None, value)
        def operator(value):
            total = jnp.zeros_like(value)
            for start, stop, block in self._blocks(active):
                block = jnp.where(jnp.asarray(active[start:stop])[None, :], block, 0.)
                inner = jnp.matmul(block.T, value, precision=jax.lax.Precision.HIGHEST)
                total += jnp.matmul(block, inner, precision=jax.lax.Precision.HIGHEST)
            return total
        return operator

    def gram_diagonal(self, active):
        if self.packed_calls is not None and self.streamer.dev.platform == 'gpu':
            from .kv_impl import _packed_diagonal_step, packed_cache_diagonal
            selected, inverse = self._selection(np.asarray(active, bool))
            cached, streamed = self._split_calls(selected)
            initial = (packed_cache_diagonal(self._device_packed, self.widths, self.means,
                       inverse, jnp.asarray(cached, jnp.int32), self._missing)
                       if cached.size else jnp.zeros(self.n, jnp.float64))
            inv_params = None if inverse is self.inverse else {int(k): inverse[int(k)] for k in streamed}
            def step(total, k, block):
                width, mean, inv, _ = self._parameters[k]
                return _packed_diagonal_step(total, block, width, mean,
                    inv if inv_params is None else inv_params[k], self._missing)
            return self._reduce(streamed, initial, step)
        if self.streamer is not None and np.all(active) and hasattr(self.streamer, 'exact_diag'):
            return self.streamer.exact_diag().astype(jnp.float64)*self.streamer._eff_m_const
        total = jnp.zeros(self.n, jnp.float64)
        for start, stop, block in self._blocks(active):
            total += jnp.sum(jnp.where(jnp.asarray(active[start:stop])[None, :], block*block, 0.), axis=1)
        return total


@dataclass
class BackgroundSSResult:
    posterior: VAMPResult
    foreground: np.ndarray
    residual: np.ndarray
    background: np.ndarray
    groups: np.ndarray | None = None
    normalizers: np.ndarray | None = None
    numerical_attempts: list[dict] | None = None
    total_seconds: float | None = None
    setup_and_products_seconds: float | None = None
    context: GroupedSSBackgroundContext | None = None

    @property
    def fitted_mean(self):
        return self.foreground+self.background


class GroupedSSBackgroundContext:
    """Reuse genotype operators and fixed trace quadrature across native outputs."""
    def __init__(self, snp_major, groups, normalizers=None):
        self.products = (snp_major if hasattr(snp_major, 'gram_operator')
                         else GenotypeProducts(snp_major))
        self.groups = np.asarray(groups)
        if (not np.issubdtype(self.groups.dtype, np.integer)
                or self.groups.shape != (self.products.p,) or self.groups.min() < 0):
            raise ValueError('groups must identify every training SNP')
        self.n_groups = int(self.groups.max())+1
        counts = np.bincount(self.groups, minlength=self.n_groups)
        self.normalizers = np.asarray(counts if normalizers is None else normalizers, float)
        if (np.any(counts == 0) or self.normalizers.shape != (self.n_groups,)
                or np.any(self.normalizers < 0) or not np.all(np.isfinite(self.normalizers))):
            raise ValueError('Invalid GRM group normalizers')
        self.solvers = {}

    def linear_solver(self, active_groups, config):
        options = dict(pcg_tolerance=config.linear_pcg_tolerance,
            pcg_max_iterations=config.linear_pcg_max_iterations,
            probes=config.trace_probes, lanczos_steps=config.trace_lanczos_steps,
            workspace_bytes=config.trace_workspace_bytes, seed=config.trace_seed,
            exact_probe_limit=config.trace_exact_probe_limit,
            trace_tolerance=config.trace_tolerance)
        key = (tuple(active_groups), tuple(options.values()))
        if key not in self.solvers:
            self.solvers[key] = MatrixFreeSampleSolver(self.products,
                np.isin(self.groups, active_groups), **options)
        return self.solvers[key]


def _matrix(value, rows, batch, name):
    value = np.asarray(value, float)
    if value.ndim == 1:
        value = value[:, None]
    if value.shape != (rows, batch) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be finite with shape ({rows},{batch})')
    return value


def refine_lasso_background_ss(context, *, y, alpha, gamma, covariates,
                               blup, variance_components, config=None, callback=None):
    """Consume current prediction alpha, native gamma, BLUP and theta automatically.

    Supports one case or a batch with the same active GRM groups. Computes
    y-X alpha-C gamma itself, uses BLUP initialization, and fixes zero groups.
    The input designs are standardized using training statistics only.
    """
    config = config or VAMPConfig(max_iterations=2000)
    if not jax.config.jax_enable_x64:
        raise RuntimeError('Enable jax_enable_x64 before SS refinement')
    response = np.asarray(y, float)
    if response.ndim == 1:
        response = response[:, None]
    if response.ndim != 2:
        raise ValueError('y must be a vector or sample-by-case matrix')
    batch = response.shape[1]
    response = _matrix(response, context.products.n, batch, 'y')
    sparse = _matrix(alpha, context.products.p, batch, 'alpha')
    initial = _matrix(blup, context.products.p, batch, 'blup')
    theta = _matrix(variance_components, context.n_groups+1, batch, 'variance_components')
    if np.any(theta[:-1] < 0) or np.any(theta[-1] <= 0):
        raise ValueError('Nonnegative background and positive noise components required')
    if np.any((context.normalizers == 0) & np.any(theta[:-1] > 0, axis=1)):
        raise ValueError('Positive variance components require positive GRM normalizers')
    c = np.empty((context.products.n, 0)) if covariates is None else np.asarray(covariates, float)
    if c.ndim != 2 or c.shape[0] != context.products.n or not np.all(np.isfinite(c)):
        raise ValueError('Invalid training covariates')
    fixed = _matrix(gamma, c.shape[1], batch, 'gamma')
    foreground = np.asarray(context.products.forward(jax.device_put(sparse)))+c@fixed
    residual = response-foreground
    active = theta[:-1] > 0
    if not np.all(active == active[:, :1]):
        raise ValueError('A batch must have the same active group pattern')
    active_groups = np.flatnonzero(active[:, 0]).tolist()
    if active_groups:
        posterior = fit_grouped_ss_vamp(context.products, residual, context.groups,
            theta[:-1], context.normalizers, theta[-1], context.linear_solver(active_groups, config),
            config, callback, initial_mean=initial)
    else:
        zeros = np.zeros_like(initial)
        parameters = np.zeros((context.n_groups, batch))
        posterior = VAMPResult(zeros, zeros.copy(), zeros.copy(), parameters, parameters.copy(),
            ['converged']*batch, [], 0., 0, np.full(batch, context.products.n)/theta[-1],
            dict(method='all_zero_background', first_mean_relative_l2_error=[0.]*batch,
                 first_mean_max_abs_error=[0.]*batch))
    background = np.asarray(context.products.forward(jax.device_put(posterior.mean)))
    return BackgroundSSResult(posterior, foreground, residual, background, context.groups, context.normalizers)


def refine_sparse_fit_background(*, fitter, grm_index, y, covariates, support,
                                 beta_active, beta_cov, theta, background_effects,
                                 config=None, context=None, callback=None):
    """Automatic adapter for the final sparse pipeline's in-memory outputs.

    Reuses native streamed training products in logical GRM order, which is
    also the order of the native component BLUP vectors. Returns refined
    EffectEstimates directly usable by existing effect writers/predictors.
    Pass the preceding result's context to reuse packed products and trace
    quadrature across conditional rounds. Initialization remains the supplied
    native BLUP; it is independent of the reused linear solver.
    """
    # Keep the native FP32 pipeline's global precision setting intact.
    from jax.experimental import enable_x64
    with enable_x64():
        return _refine_sparse_fit_background(fitter=fitter, grm_index=grm_index,
            y=y, covariates=covariates, support=support, beta_active=beta_active,
            beta_cov=beta_cov, theta=theta, background_effects=background_effects,
            config=config, context=context, callback=callback)


def _refine_sparse_fit_background(*, fitter, grm_index, y, covariates, support,
                                  beta_active, beta_cov, theta, background_effects, config, context,
                                  callback=None):
    from dataclasses import asdict, replace
    began = time.monotonic()
    config = config or VAMPConfig(max_iterations=2000)
    counts = np.asarray(grm_index.m_per_grm, int)
    if len(background_effects.snp_effects) != len(counts):
        raise ValueError('Native BLUP components do not match the GRM partition')
    n, p = len(y), int(grm_index.m_total)
    groups = np.repeat(np.arange(len(counts)), counts)
    alpha = np.zeros(p); alpha[np.asarray(support, int)] = np.asarray(beta_active, float)
    initial = np.concatenate([np.asarray(v, float) for v in background_effects.snp_effects])
    if initial.shape != (p,):
        raise ValueError('Native BLUP vector lengths do not match the SNP groups')
    partitioned = getattr(fitter, '_partitioned_streamer', None)
    if partitioned is not None:
        normalizers = np.asarray(partitioned._component_eff_m_host, float)
    else:
        normalizers = np.array([float(np.asarray(streamer._eff_m_const)) for streamer in fitter.streamers])
    budget = getattr(getattr(fitter, 'cfg', None), 'gpu_budget_bytes', None)
    block_bytes = 512*1024**2
    if budget is not None:
        block_bytes = min(block_bytes, max(1, int(budget)//16))
        depth = min(config.trace_lanczos_steps, n)
        minimum_workspace = 8*(n*(2*depth+32)+8*(depth+1)**2)
        workspace = min(max(config.trace_workspace_bytes, minimum_workspace),
                        max(1, int(budget)//8))
        config = replace(config, trace_workspace_bytes=workspace)
    cache_bytes = getattr(getattr(fitter, 'cfg', None), 'packed_device_cache_bytes', None)
    cache_bytes = int(budget) if budget is not None else (cache_bytes or 0)
    if context is None:
        # REML is complete. Reuse its reservation for SS's smaller FP64 blocks
        # instead of holding two copies of the same compressed genotypes.
        for streamer in getattr(fitter, 'streamers', ()):
            release = getattr(streamer, 'release_packed_device_cache', None)
            if release is not None:
                release()
        products = StreamedGenotypeProducts(grm_index, n, normalizers, block_bytes=block_bytes,
            device_cache_bytes=cache_bytes, gpu_budget_bytes=budget,
            trace_workspace_bytes=config.trace_workspace_bytes)
        logger.info('[background SS] n=%s, SNPs=%s, packed GPU cache=%.3f GiB / %.3f GiB budget; '
                    'preparing linear solver and trace quadrature',
                    n, p, products.device_cache_bytes/2**30, cache_bytes/2**30)
        context = GroupedSSBackgroundContext(products, groups, normalizers)
    elif (context.products.n != n or context.products.p != p
            or getattr(context.products, 'index', None) is not grm_index
            or not np.array_equal(context.groups, groups)
            or not np.array_equal(context.normalizers, normalizers)):
        raise ValueError('Reused SS context must have the same training index and GRM partition')
    result = refine_lasso_background_ss(context, y=y, alpha=alpha, gamma=beta_cov,
        covariates=covariates, blup=initial, variance_components=theta, config=config, callback=callback)
    attempts = [dict(status=result.posterior.status, iterations=result.posterior.iterations,
                     seconds=result.posterior.seconds, config=asdict(config))]
    if result.posterior.status != ['converged']:
        # Weakly identified pi can amplify curvature below stopping accuracy.
        # Retain the model and tolerances, reuse trace quadrature, and restart from
        # the same BLUP with conservative numerical settings only.
        retry = replace(config, damping=min(config.damping, .1), em_damping=min(config.em_damping, .025),
                        em_acceleration_floor=max(config.em_acceleration_floor, config.em_tolerance))
        result = refine_lasso_background_ss(context, y=y, alpha=alpha, gamma=beta_cov,
            covariates=covariates, blup=initial, variance_components=theta, config=retry, callback=callback)
        attempts.append(dict(status=result.posterior.status, iterations=result.posterior.iterations,
                             seconds=result.posterior.seconds, config=asdict(retry)))
    result.numerical_attempts = attempts
    result.context = context
    if result.posterior.status != ['converged']:
        raise VAMPConvergenceError('background', result.posterior,
            elapsed_seconds=time.monotonic()-began, attempts=attempts)
    beta = result.posterior.mean[:, 0]
    # Batch components to bound the SNP-effect workspace even for large
    # partitions. The usual four groups still share one genotype scan.
    components = []
    for first in range(0, len(counts), 8):
        ids = jnp.arange(first, min(first+8, len(counts)))
        component_effects = jnp.where(jnp.asarray(groups)[:, None] == ids[None, :],
                                      jnp.asarray(beta)[:, None], 0.)
        predictions = context.products.forward(component_effects)
        predictions.block_until_ready()
        components.extend(predictions[:, g] for g in range(len(ids)))
    refined = replace(background_effects,
        random_effect=jax.device_put(result.background[:, 0]), random_effect_components=tuple(components),
        snp_effects=tuple(jax.device_put(beta[groups == g]) for g in range(len(counts))))
    result.total_seconds = time.monotonic()-began
    result.setup_and_products_seconds = result.total_seconds-sum(a['seconds'] for a in attempts)
    return refined, result, initial
