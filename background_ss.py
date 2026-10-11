"""Constrained SS background refinement of finalized sparse prediction outputs.

The context owns reusable genotype products and trace quadrature. All supplied
phenotypes, designs and coefficients must be on the model's training scale.
No foreground, covariance component or noise parameter is re-estimated.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import jax
import jax.numpy as jnp
from .grouped_ss_vamp import GenotypeProducts, VAMPConfig, VAMPResult, fit_grouped_ss_vamp
from .ss_linear import MatrixFreeSampleSolver


class StreamedGenotypeProducts:
    """Native forward/transpose/kernel operators in logical SNP order.

    Custom index implementations may supply standardized columns instead;
    that fallback retains only one memory-bounded genotype block.
    """
    def __init__(self, index, n, normalizers, *, block_bytes=64*1024**2,
                 device_cache_bytes=0):
        self.index, self.n, self.p = index, int(n), int(index.m_total)
        self.streamer = getattr(index, 'streamer', None)
        self.normalizers = np.asarray(normalizers, float)
        self.offsets = np.r_[0, np.cumsum(index.m_per_grm)]
        self.chunk_size = min(4096, max(1, block_bytes//max(16*self.n, 1)))
        self.packed_calls = None
        self._device_packed = None
        self.device_cache_bytes = 0
        if (self.streamer is not None and getattr(self.streamer, '_packed_buf', None) is not None
                and getattr(self.streamer, '_means_host', None) is not None
                and getattr(self.streamer, '_inv_sds_host', None) is not None):
            # SS uses FP64 products. Subdivide the existing packed cache so a
            # call planned for the FP32 REML stage cannot create an oversized
            # FP64 decode workspace. Source order and training statistics stay
            # exactly those of the native streamer.
            width = max(4, self.chunk_size//4*4)
            calls = []
            for call, (start, size) in enumerate(zip(self.streamer._call_snp_starts,
                                                    self.streamer._call_true_widths)):
                for offset in range(0, int(size), width):
                    actual = min(width, int(size)-offset)
                    calls.append((call, offset//4, (offset+actual+3)//4,
                                  int(start)+offset, actual))
            self.packed_calls = calls
            self.packed_width = min(width, int(self.streamer._max_unpack_width))
            means = np.zeros((len(calls), self.packed_width), np.float32)
            inverse = np.zeros_like(means)
            for k, (_, _, _, start, actual) in enumerate(calls):
                means[k, :actual] = self.streamer._means_host[start:start+actual]
                inverse[k, :actual] = self.streamer._inv_sds_host[start:start+actual]
            self.means = jax.device_put(means, self.streamer.dev)
            self.inverse = jax.device_put(inverse, self.streamer.dev)
            self.starts = jax.device_put(np.array([c[3] for c in calls], np.int32), self.streamer.dev)
            self.widths = jax.device_put(np.array([c[4] for c in calls], np.int32), self.streamer.dev)
            self._all_calls = jnp.arange(len(calls), dtype=jnp.int32)
            self._missing = jnp.asarray(self.streamer._missing_val, jnp.uint8)
            needed = len(calls)*self.n*(self.packed_width//4)
            if (getattr(self.streamer.dev, 'platform', None) == 'gpu'
                    and 0 < needed <= device_cache_bytes):
                from .kv_impl import _device_put_block, _store_packed_call
                with jax.default_device(self.streamer.dev):
                    cache = jnp.zeros((len(calls), self.n, self.packed_width//4), jnp.uint8)
                # One upload at a time bounds setup memory at cache + one block.
                for k in range(len(calls)):
                    block = self._packed(k)
                    if block.shape[1] < self.packed_width//4:
                        block = np.pad(block, ((0, 0), (0, self.packed_width//4-block.shape[1])))
                    cache = _store_packed_call(cache, _device_put_block(block, self.streamer.dev),
                                               jnp.asarray(k, jnp.int32))
                    cache.block_until_ready()
                self._device_packed = cache
                self.device_cache_bytes = needed

    def _active_calls(self, active):
        selected, masks = [], np.zeros((len(self.packed_calls), self.packed_width), bool)
        for k, (_, _, _, start, actual) in enumerate(self.packed_calls):
            masks[k, :actual] = active[start:start+actual]
            if np.any(masks[k]):
                selected.append(k)
        return jnp.asarray(selected, jnp.int32), self.inverse*jnp.asarray(masks)

    def restricted(self, active):
        """Restrict VAMP linear products; general foreground products stay unmasked."""
        if self._device_packed is None:
            return self
        from types import SimpleNamespace
        from .kv_impl import packed_cache_forward, packed_cache_transpose
        selected, inverse = self._active_calls(np.asarray(active, bool))
        return SimpleNamespace(
            forward=lambda effects: packed_cache_forward(self._device_packed, self.starts,
                self.widths, self.means, inverse, selected, effects, self._missing),
            transpose=lambda value: packed_cache_transpose(self._device_packed, self.starts,
                self.widths, self.means, inverse, selected, value, self._missing, p=self.p))

    def _packed(self, k):
        call, first, last, _, _ = self.packed_calls[k]
        return self.streamer._packed_block_host(call)[:, first:last]

    def _blocks(self, active=None):
        for start in range(0, self.p, self.chunk_size):
            stop = min(start+self.chunk_size, self.p)
            if active is not None and not np.any(active[start:stop]):
                continue
            block = self.index.extract_standardized_columns(np.arange(start, stop))
            yield start, stop, jnp.asarray(block, jnp.float64)

    def forward(self, effects):
        effects = jnp.asarray(effects)
        if self._device_packed is not None:
            from .kv_impl import packed_cache_forward
            return packed_cache_forward(self._device_packed, self.starts, self.widths,
                self.means, self.inverse, self._all_calls, effects, self._missing)
        if self.packed_calls is not None:
            from .kv_impl import xv_impl_streamed
            return xv_impl_streamed(effects, self.starts, self.widths, self.means, self.inverse,
                n=self.n, m=self.p, n_calls=len(self.packed_calls), pop_block=self._packed,
                missing_val=int(self.streamer._missing_val))
        if self.streamer is not None and hasattr(self.streamer, 'xv'):
            return self.streamer.xv(effects)
        total = jnp.zeros((self.n, effects.shape[1]), effects.dtype)
        for start, stop, block in self._blocks():
            total += jnp.matmul(block, effects[start:stop], precision=jax.lax.Precision.HIGHEST)
        return total

    def transpose(self, residual):
        residual = jnp.asarray(residual)
        if self._device_packed is not None:
            from .kv_impl import packed_cache_transpose
            return packed_cache_transpose(self._device_packed, self.starts, self.widths,
                self.means, self.inverse, self._all_calls, residual, self._missing, p=self.p)
        if self.packed_calls is not None:
            from .kv_impl import xtv_impl_streamed
            return xtv_impl_streamed(residual, self.p, self.starts, self.widths,
                self.means, self.inverse, n_calls=len(self.packed_calls),
                pop_block=self._packed, missing_val=int(self.streamer._missing_val))
        if self.streamer is not None:
            return self.streamer.xtv(residual, normalize=False)
        total = jnp.zeros((self.p, residual.shape[1]), residual.dtype)
        for start, stop, block in self._blocks():
            values = jnp.matmul(block.T, residual, precision=jax.lax.Precision.HIGHEST)
            total = total.at[start:stop].set(values)
        return total

    def gram_operator(self, active):
        active = np.asarray(active, bool)
        if self._device_packed is not None:
            from .kv_impl import packed_cache_gram
            selected, inverse = self._active_calls(active)
            return lambda value: packed_cache_gram(self._device_packed, self.widths,
                self.means, inverse, selected, value, self._missing)
        if self.packed_calls is not None:
            from .kv_impl import _device_put_block, _kv_one_call_jit
            selected, inverse = [], []
            for k, (_, _, _, start, actual) in enumerate(self.packed_calls):
                if np.any(active[start:start+actual]):
                    mask = np.zeros(self.packed_width, bool)
                    mask[:actual] = active[start:start+actual]
                    selected.append(k); inverse.append(self.inverse[k]*jnp.asarray(mask))
            missing = jnp.asarray(self.streamer._missing_val, jnp.uint8)
            def packed_operator(value):
                total = jnp.zeros_like(value)
                if selected:
                    pending = _device_put_block(self._packed(selected[0]), self.streamer.dev)
                    for position, k in enumerate(selected):
                        packed = pending
                        if position+1 < len(selected):
                            pending = _device_put_block(self._packed(selected[position+1]), self.streamer.dev)
                        total = _kv_one_call_jit(packed, self.widths[k], self.means[k],
                            inverse[position], value, missing, total)
                        del packed
                return total
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
        if self._device_packed is not None:
            from .kv_impl import packed_cache_diagonal
            selected, inverse = self._active_calls(np.asarray(active, bool))
            return packed_cache_diagonal(self._device_packed, self.widths, self.means,
                                          inverse, selected, self._missing)
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
                                 config=None):
    """Automatic adapter for the final sparse pipeline's in-memory outputs.

    Reuses native streamed training products in logical GRM order, which is
    also the order of the native component BLUP vectors. Returns refined
    EffectEstimates directly usable by existing effect writers/predictors.
    """
    # Keep the native FP32 pipeline's global precision setting intact.
    from jax.experimental import enable_x64
    with enable_x64():
        return _refine_sparse_fit_background(fitter=fitter, grm_index=grm_index,
            y=y, covariates=covariates, support=support, beta_active=beta_active,
            beta_cov=beta_cov, theta=theta, background_effects=background_effects, config=config)


def _refine_sparse_fit_background(*, fitter, grm_index, y, covariates, support,
                                  beta_active, beta_cov, theta, background_effects, config):
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
    from .runtime_env import packed_device_cache_budget
    cache_bytes = getattr(getattr(fitter, 'cfg', None), 'packed_device_cache_bytes', None)
    if cache_bytes is None:
        cache_bytes = packed_device_cache_budget(budget)
    # REML is complete. Reuse its reservation for SS's smaller FP64 blocks
    # instead of holding two copies of the same compressed genotypes.
    for streamer in getattr(fitter, 'streamers', ()):
        release = getattr(streamer, 'release_packed_device_cache', None)
        if release is not None:
            release()
    products = StreamedGenotypeProducts(grm_index, n, normalizers, block_bytes=block_bytes,
                                        device_cache_bytes=cache_bytes)
    context = GroupedSSBackgroundContext(products, groups, normalizers)
    result = refine_lasso_background_ss(context, y=y, alpha=alpha, gamma=beta_cov,
        covariates=covariates, blup=initial, variance_components=theta, config=config)
    attempts = [dict(status=result.posterior.status, iterations=result.posterior.iterations,
                     seconds=result.posterior.seconds, config=asdict(config))]
    if result.posterior.status != ['converged']:
        # Weakly identified pi can amplify curvature below stopping accuracy.
        # Retain the model and tolerances, reuse trace quadrature, and restart from
        # the same BLUP with conservative numerical settings only.
        retry = replace(config, damping=min(config.damping, .1), em_damping=min(config.em_damping, .025),
                        em_acceleration_floor=max(config.em_acceleration_floor, config.em_tolerance))
        result = refine_lasso_background_ss(context, y=y, alpha=alpha, gamma=beta_cov,
            covariates=covariates, blup=initial, variance_components=theta, config=retry)
        attempts.append(dict(status=result.posterior.status, iterations=result.posterior.iterations,
                             seconds=result.posterior.seconds, config=asdict(retry)))
    result.numerical_attempts = attempts
    if result.posterior.status != ['converged']:
        raise RuntimeError('SS background refinement did not converge; outputs rejected')
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
