"""Constrained SS background refinement of finalized native LASSO outputs.

The context owns reusable genotype products and eigensystems. All supplied
phenotypes, designs and coefficients must be on the model's training scale.
No foreground, covariance component or noise parameter is re-estimated.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import partial
import time
import numpy as np
import jax
import jax.numpy as jnp
from .grouped_ss_vamp import GenotypeProducts, SampleSpectrum, VAMPConfig, VAMPResult, fit_grouped_ss_vamp


@partial(jax.jit, donate_argnums=(0,))
def _add_sample_gram(total, block, active):
    """Accumulate one FP64 Gram block while reusing its large buffer."""
    z = jnp.where(active[:, None], block, 0.).astype(jnp.float64)
    return total+jnp.matmul(z.T, z, precision=jax.lax.Precision.HIGHEST)


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
    """Reuse one standardized SNP-major design across native model outputs."""
    def __init__(self, snp_major, groups, normalizers=None, spectra=None):
        self.products = snp_major if isinstance(snp_major, GenotypeProducts) else GenotypeProducts(snp_major)
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
        self.spectra = {} if spectra is None else dict(spectra)

    def spectrum(self, active_groups):
        key = tuple(active_groups)
        if key not in self.spectra:
            # Exact FP64 sample Gram from resident FP32 genotypes, without a
            # p*p LD matrix or a full FP64 genotype copy.
            n, p = self.products.n, self.products.p
            gram = jnp.zeros((n, n), jnp.float64)
            mask = jax.device_put(np.isin(self.groups, key))

            for start in range(0, p, self.products.chunk_size):
                end = min(start+self.products.chunk_size, p)
                gram = _add_sample_gram(gram, self.products.x[start:end], mask[start:end])
            eig, vectors = jnp.linalg.eigh((gram+gram.T)*.5)
            self.spectra[key] = SampleSpectrum(eig, vectors, int(np.isin(self.groups, key).sum()))
        spectrum = self.spectra[key]
        if spectrum.n != self.products.n or spectrum.p != int(np.isin(self.groups, key).sum()):
            raise ValueError('Cached spectrum does not match active training SNPs')
        return spectrum


def _matrix(value, rows, batch, name):
    value = np.asarray(value, float)
    if value.ndim == 1:
        value = value[:, None]
    if value.shape != (rows, batch) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be finite with shape ({rows},{batch})')
    return value


def refine_lasso_background_ss(context, *, y, alpha, gamma, covariates,
                               blup, variance_components, config=None, callback=None):
    """Consume final LASSO alpha/gamma, native BLUP and theta automatically.

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
            theta[:-1], context.normalizers, theta[-1], context.spectrum(active_groups),
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

    Materializes standardized training columns in logical GRM order, which is
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
    counts = np.asarray(grm_index.m_per_grm, int)
    if len(background_effects.snp_effects) != len(counts):
        raise ValueError('Native BLUP components do not match the GRM partition')
    n, p = len(y), int(grm_index.m_total)
    x = np.empty((p, n), np.float32)
    for offset in range(0, p, 4096):
        end = min(offset+4096, p)
        x[offset:end] = grm_index.extract_standardized_columns(np.arange(offset, end)).T
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
    context = GroupedSSBackgroundContext(x, groups, normalizers)
    config = config or VAMPConfig(max_iterations=2000)
    result = refine_lasso_background_ss(context, y=y, alpha=alpha, gamma=beta_cov,
        covariates=covariates, blup=initial, variance_components=theta, config=config)
    attempts = [dict(status=result.posterior.status, iterations=result.posterior.iterations,
                     seconds=result.posterior.seconds, config=asdict(config))]
    if result.posterior.status != ['converged']:
        # Weakly identified pi can amplify curvature below stopping accuracy.
        # Retain the model and tolerances, reuse the spectrum, and restart from
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
    # One genotype pass for all components, rather than one complete scan
    # per group. These columns use the same posterior means as prediction.
    component_effects = jnp.where(jnp.asarray(groups)[:, None] == jnp.arange(len(counts))[None, :],
                                  jnp.asarray(beta)[:, None], 0.)
    component_predictions = context.products.forward(component_effects)
    component_predictions.block_until_ready()
    components = tuple(component_predictions[:, g] for g in range(len(counts)))
    refined = replace(background_effects,
        random_effect=jax.device_put(result.background[:, 0]), random_effect_components=components,
        snp_effects=tuple(jax.device_put(beta[groups == g]) for g in range(len(counts))))
    result.total_seconds = time.monotonic()-began
    result.setup_and_products_seconds = result.total_seconds-sum(a['seconds'] for a in attempts)
    return refined, result, initial
