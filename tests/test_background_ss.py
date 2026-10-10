"""Native LASSO adapter checks against independent regression posteriors."""
import os
import time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import numpy as np
import pytest
from GPU_REML.background_ss import GroupedSSBackgroundContext, refine_lasso_background_ss
from GPU_REML.grouped_ss_vamp import VAMPConfig


@pytest.fixture(autouse=True)
def ss_precision():
    with jax.experimental.enable_x64():
        yield


def test_native_lasso_outputs_residual_and_gaussian_posterior():
    rng = np.random.default_rng(228)
    n, p = 35, 100
    x = rng.normal(size=(p, n)).astype('float32')
    groups = np.repeat(np.arange(4), 25)
    context = GroupedSSBackgroundContext(x, groups)
    y, cov = rng.normal(size=n), rng.normal(size=(n, 3))
    alpha = np.zeros(p); alpha[[3, 27, 52]] = [.2, -.1, .3]
    gamma = np.array([.15, -.07, .04])
    theta = np.array([0., .4, .2, .1, .5])
    residual = y-x.astype(float).T@alpha-cov@gamma
    d = (theta[:4]/25)[groups]
    oracle = d*(x.astype(float)@np.linalg.solve(x.astype(float).T@(d[:, None]*x.astype(float))+.5*np.eye(n), residual))
    initial = oracle.copy(); initial[groups == 0] = 5.
    alpha_before, gamma_before = alpha.copy(), gamma.copy()
    result = refine_lasso_background_ss(context, y=y, alpha=alpha, gamma=gamma,
        covariates=cov, blup=initial, variance_components=theta,
        config=VAMPConfig(initial_pi=1., learn_pi=False, damping=.5,
                          tolerance=1e-9, max_iterations=1000, min_iterations=5))
    np.testing.assert_allclose(result.residual[:, 0], residual, atol=1e-14)
    assert result.posterior.status == ['converged']
    np.testing.assert_allclose(result.posterior.mean[:, 0], oracle, atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(result.fitted_mean[:, 0], y-residual+x.astype(float).T@oracle, atol=1e-8)
    np.testing.assert_array_equal(alpha, alpha_before)
    np.testing.assert_array_equal(gamma, gamma_before)
    assert np.all(result.posterior.mean[groups == 0] == 0)


def test_all_zero_background_and_mixed_active_patterns():
    rng = np.random.default_rng(398)
    x = rng.normal(size=(12, 8)).astype('float32')
    groups = np.repeat(np.arange(4), 3)
    context = GroupedSSBackgroundContext(x, groups)
    result = refine_lasso_background_ss(context, y=rng.normal(size=8), alpha=np.zeros(12),
        gamma=np.empty(0), covariates=None, blup=np.ones(12),
        variance_components=np.array([0., 0., 0., 0., .5]))
    assert not context.spectra
    assert result.posterior.status == ['converged']
    assert np.all(result.background == 0) and np.all(result.posterior.mean == 0)
    theta = np.array([[.1, 0.], [.1, .1], [.1, .1], [.1, .1], [.5, .5]])
    with pytest.raises(ValueError, match='same active'):
        refine_lasso_background_ss(context, y=np.zeros((8, 2)), alpha=np.zeros((12, 2)),
            gamma=np.empty((0, 2)), covariates=None, blup=np.zeros((12, 2)), variance_components=theta)


def test_constant_only_group_is_inactive_with_zero_native_normalizer():
    rng = np.random.default_rng(299)
    x = rng.normal(size=(24, 8)).astype('float32')
    x[:12] = 0
    groups = np.repeat(np.arange(2), 12)
    context = GroupedSSBackgroundContext(x, groups, normalizers=np.array([0., 12.]))
    y = rng.normal(size=8)
    theta = np.array([0., .4, .6])
    z = x[12:].astype(float)
    expected = (.4/12)*z@np.linalg.solve((.4/12)*z.T@z+.6*np.eye(8), y)
    result = refine_lasso_background_ss(context, y=y, alpha=np.zeros(24),
        gamma=np.empty(0), covariates=None, blup=np.zeros(24), variance_components=theta,
        config=VAMPConfig(initial_pi=1., learn_pi=False, damping=.5, tolerance=1e-9))
    assert result.posterior.status == ['converged']
    np.testing.assert_array_equal(result.posterior.mean[:12], 0)
    np.testing.assert_allclose(result.posterior.mean[12:, 0], expected, atol=1e-9)
    with pytest.raises(ValueError, match='positive GRM normalizers'):
        refine_lasso_background_ss(context, y=y, alpha=np.zeros(24),
            gamma=np.empty(0), covariates=None, blup=np.zeros(24),
            variance_components=np.array([.1, .4, .6]))


def test_native_wrapper_retains_grm_normalizers_order_and_precision():
    from types import SimpleNamespace
    from GPU_REML.background_ss import refine_sparse_fit_background
    from GPU_REML.reml_model import EffectEstimates
    from jax.experimental import disable_x64
    rng = np.random.default_rng(441)
    n, p = 12, 48
    x = rng.normal(size=(p, n)).astype('float32')
    groups = np.repeat(np.arange(4), 12)
    x[[0, 12, 24, 36]] = 0  # Constant markers counted in P, excluded from native M.
    normalizers = np.full(4, 11.)
    index = SimpleNamespace(m_per_grm=[12]*4, m_total=p,
        extract_standardized_columns=lambda indices:x[indices].T)
    fitter = SimpleNamespace(_partitioned_streamer=SimpleNamespace(_component_eff_m_host=normalizers))
    y, cov = rng.normal(size=n), rng.normal(size=(n, 2))
    support, alpha, gamma = np.array([3, 25]), np.array([.3, -.2]), np.array([.1, .2])
    theta = np.array([.2, 0., .3, .1, .4])
    residual = y-x[support].astype(float).T@alpha-cov@gamma
    d = (theta[:-1]/normalizers)[groups]
    expected = d*(x.astype(float)@np.linalg.solve(x.astype(float).T@(d[:, None]*x.astype(float))+.4*np.eye(n), residual))
    initial = EffectEstimates(np.zeros(0), x.T@expected, tuple(x[groups == g].T@expected[groups == g] for g in range(4)),
        tuple(expected[groups == g] for g in range(4)), 1e-5, 7, .5, 1.2)
    began = time.monotonic()
    with disable_x64():
        refined, result, blup = refine_sparse_fit_background(fitter=fitter, grm_index=index,
            y=y, covariates=cov, support=support, beta_active=alpha, beta_cov=gamma, theta=theta,
            background_effects=initial, config=VAMPConfig(initial_pi=1., learn_pi=False,
                tolerance=1e-9, damping=.5, min_iterations=5))
        assert not jax.config.jax_enable_x64
    np.testing.assert_allclose(np.concatenate(refined.snp_effects), expected, atol=2e-8)
    np.testing.assert_allclose(np.sum(np.asarray(refined.random_effect_components), axis=0), refined.random_effect, atol=1e-12)
    np.testing.assert_array_equal(result.normalizers, normalizers)
    np.testing.assert_array_equal(result.groups, groups)
    np.testing.assert_array_equal(blup, expected)
    np.testing.assert_allclose(result.posterior.pi[:, 0]*result.posterior.slab_variance[:, 0], theta[:-1]/normalizers)
    assert refined.pcg_iters == initial.pcg_iters
    assert 0 <= result.setup_and_products_seconds <= result.total_seconds
    assert result.total_seconds <= time.monotonic()-began
