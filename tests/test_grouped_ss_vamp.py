"""Independent numerical references for constrained SS and VAMP."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import numpy as np
import pytest
from scipy.special import expit, logsumexp
from scipy.optimize import minimize_scalar, minimize, brentq
from GPU_REML.grouped_ss_vamp import (
    constrained_ss_m_step, unconstrained_ss_m_step, spike_slab_denoise, ss_message_from_mean, GenotypeProducts,
    SampleSpectrum, VAMPConfig, fit_grouped_ss_vamp,
)


@pytest.fixture(autouse=True)
def ss_precision():
    with jax.experimental.enable_x64():
        yield


def test_scalar_posterior_and_analytic_divergence():
    r = np.array([-.7, 0., .5, 2.])
    pi, v, gam = .17, .4, 3.2
    mean, var, pip, mu, s2, evidence = map(np.asarray, spike_slab_denoise(r, gam, pi, v))
    spike_log = np.log1p(-pi)-.5*np.log(2*np.pi/gam)-.5*gam*r*r
    slab_log = np.log(pi)-.5*np.log(2*np.pi*(v+1/gam))-.5*r*r/(v+1/gam)
    expected_pip = np.exp(slab_log-logsumexp([spike_log, slab_log], axis=0))
    np.testing.assert_allclose(pip, expected_pip, atol=1e-14)
    np.testing.assert_allclose(mean, pip*v*gam/(1+v*gam)*r, atol=1e-14)
    step = 1e-6
    difference = (np.asarray(spike_slab_denoise(r+step, gam, pi, v)[0])-
                  np.asarray(spike_slab_denoise(r-step, gam, pi, v)[0]))/(2*step)
    np.testing.assert_allclose(difference, gam*var, rtol=1e-8, atol=1e-9)
    assert np.all(var >= 0)


@pytest.mark.parametrize('a,b,count,d', [(20., 1.7, 100., .004),
    (1e-6, 1e-11, 158064., 1e-6), (100., .0002, 100., .001),
    (100., 4., 100., .001), (0., 0., 100., .001)])
def test_constrained_root_against_independent_optimizer(a, b, count, d):
    pi, v = map(float, constrained_ss_m_step(a, b, count, d))
    def q(p):
        bernoulli = 0. if count == a else (count-a)*np.log1p(-p)
        return 1.5*a*np.log(p)+bernoulli-b*p/(2*d)
    solution = minimize_scalar(lambda p: -q(p), bounds=(1e-8, 1-1e-12),
                               method='bounded', options={'xatol': 1e-14})
    assert q(pi) >= q(solution.x)-1e-7*max(1., abs(q(solution.x)))
    assert pi*v == pytest.approx(d, rel=1e-14)
    assert 1e-8 <= pi <= 1


def test_sample_space_solve_trace_and_chunked_products():
    rng = np.random.default_rng(73)
    x = rng.normal(size=(57, 13)).astype('float32')
    x -= x.mean(axis=1, keepdims=True)
    products = GenotypeProducts(x, chunk_size=11)
    coef, rhs = rng.normal(size=(57, 3)), rng.normal(size=(13, 3))
    np.testing.assert_allclose(products.forward(coef), x.astype(float).T@coef, atol=1e-12)
    np.testing.assert_allclose(products.transpose(rhs), x.astype(float)@rhs, atol=1e-12)
    spectrum = SampleSpectrum.from_genotype(x)
    shifts = np.array([.4, 2., 150.])
    solved, trace_fraction = map(np.asarray, spectrum.solve_and_trace(rhs, shifts))
    gram = x.astype(float).T@x.astype(float)
    for c, shift in enumerate(shifts):
        expected = np.linalg.solve(gram+shift*np.eye(13), rhs[:, c])
        np.testing.assert_allclose(solved[:, c], expected, atol=1e-11)
        inverse = np.linalg.inv(x.astype(float)@x.astype(float).T+shift*np.eye(57))
        alpha = shift*np.trace(inverse)/57
        assert 1-trace_fraction[c] == pytest.approx(alpha, rel=1e-11)
        assert np.linalg.norm((gram+shift*np.eye(13))@solved[:, c]-rhs[:, c])/np.linalg.norm(rhs[:, c]) < 1e-11


def test_invalid_design_and_nonfinite_spectrum_are_rejected():
    with pytest.raises(ValueError, match='nonempty'):
        GenotypeProducts(np.empty((0, 5)))
    with pytest.raises(ValueError, match='finite'):
        GenotypeProducts(np.full((4, 5), np.nan))
    with pytest.raises(ValueError, match='chunk size'):
        GenotypeProducts(np.ones((4, 5)), chunk_size=0)
    with pytest.raises(ValueError, match='finite'):
        SampleSpectrum(np.array([np.nan, 1.]), np.eye(2), 4)
    with pytest.raises(ValueError, match='finite'):
        SampleSpectrum(np.ones(2), np.full((2, 2), np.nan), 4)


@pytest.mark.parametrize('warm_start', [False, True])
def test_gaussian_limit_matches_exact_multigroup_blup_with_zero_group(warm_start):
    rng = np.random.default_rng(19)
    p, n = 120, 35
    x = rng.normal(size=(p, n)).astype('float32')
    x -= x.mean(axis=1, keepdims=True)
    groups = np.repeat(np.arange(4), 30)
    theta = np.array([0., .5, .15, .7])
    noise = .45
    counts = np.bincount(groups)
    d = theta/counts
    y = rng.normal(size=n)
    prior_var = d[groups]
    gram = x.astype(float).T@(prior_var[:, None]*x.astype(float))
    expected = prior_var*(x.astype(float)@np.linalg.solve(gram+noise*np.eye(n), y))
    supplied = expected.copy() if warm_start else None
    if warm_start:
        supplied[groups == 0] = 8.  # Inactive groups must ignore supplied effects.
    result = fit_grouped_ss_vamp(GenotypeProducts(x, 23), y, groups, theta, counts, [noise],
        SampleSpectrum.from_genotype(x, theta[groups] > 0),
        VAMPConfig(initial_pi=1., learn_pi=False, damping=.5, max_iterations=1000,
                   min_iterations=5, tolerance=1e-9, consecutive_converged=3), initial_mean=supplied)
    assert result.status == ['converged']
    np.testing.assert_allclose(result.mean[:, 0], expected, rtol=2e-7, atol=1e-9)
    assert np.all(result.mean[groups == 0] == 0)
    assert np.all(result.pip[groups == 0] == 0)
    np.testing.assert_allclose(result.pi[:, 0]*result.slab_variance[:, 0], d, atol=1e-15)
    if warm_start:
        assert result.initialization['inactive_initial_entries_discarded'] == [30]
        assert result.initialization['first_mean_relative_l2_error'][0] < 1e-13


def test_sparse_model_batch_and_constraint_em_fixed_point():
    rng = np.random.default_rng(83)
    p, n = 240, 80
    x = rng.normal(size=(p, n)).astype('float32')
    groups = np.repeat(np.arange(4), p//4)
    beta = np.zeros((p, 2)); beta[[13, 90, 131, 221]] = [.4, -.35]
    y = x.T@beta+rng.normal(scale=.5, size=(n, 2))
    theta = np.full((4, 2), .16)
    result = fit_grouped_ss_vamp(GenotypeProducts(x, 31), y, groups, theta, np.full(4, 60),
        [.25, .25], SampleSpectrum.from_genotype(x),
        VAMPConfig(max_iterations=800, tolerance=1e-5, em_tolerance=1e-5,
                   damping=.15, em_damping=.25, min_iterations=20))
    assert result.status == ['converged', 'converged']
    assert np.all(np.isfinite(result.mean))
    np.testing.assert_allclose(result.pi*result.slab_variance, theta/60, rtol=1e-14)
    assert np.max(result.trace[-1]['module_error']) < 1e-5


@pytest.mark.parametrize('warm_start', [False, True])
def test_non_gaussian_orthogonal_model_matches_exact_posterior(warm_start):
    rng = np.random.default_rng(63)
    p, n = 24, 48
    basis, _ = np.linalg.qr(rng.normal(size=(n, p)))
    x = np.sqrt(n)*basis.T  # float64 storage to preserve exact orthogonality
    groups = np.repeat(np.arange(4), 6)
    theta = np.array([.2, .7, .35, .1])
    noise = .8
    y = rng.normal(size=n)
    probability = .13
    d = (theta/6)[groups]
    blup = d/(noise+n*d)*(x@y)
    result = fit_grouped_ss_vamp(GenotypeProducts(x, 7), y, groups, theta, np.full(4, 6),
        [noise], SampleSpectrum.from_genotype(x),
        VAMPConfig(initial_pi=probability, learn_pi=False, damping=.5,
                   tolerance=1e-9, min_iterations=5, max_iterations=500),
        initial_mean=blup if warm_start else None)
    expected = spike_slab_denoise(x@y/n, n/noise, probability, (theta/6/probability)[groups])
    assert result.status == ['converged']
    for actual, reference in zip([result.mean[:, 0], result.variance[:, 0], result.pip[:, 0]], expected[:3]):
        np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=1e-10)


def test_initial_message_against_independent_scalar_root_and_batched_roundtrip():
    pi = np.array([0., 1., 1e-8, .005, .1, .9999, .5])[:, None]
    v = np.array([0., .01, 10., 1e-6, .1, 5., 1e-10])[:, None]
    precision = np.array([13000., 3.5])[None, :]
    target = np.array([[.2, -.3], [.1, -.2], [.1, -.03], [1e-4, -1e-7],
                       [.03, -.2], [.5, -.6], [1e-12, -1e-12]])
    message = np.asarray(ss_message_from_mean(target, precision, pi, v))
    expected = target.copy(); expected[0] = 0
    mean = np.asarray(spike_slab_denoise(message, precision, pi, v)[0])
    np.testing.assert_allclose(mean, expected, rtol=2e-12, atol=1e-20)
    assert np.all(message[0] == 0)
    # Independent SciPy root of mixture-density posterior, for moderate pi.
    for i in [3, 4, 5, 6]:
        for c in range(2):
            probability, variance, gam = pi[i, 0], v[i, 0], precision[0, c]
            gain = gam*variance/(1+gam*variance)
            def mean_at(r):
                logit = np.log(probability/(1-probability))-.5*np.log1p(gam*variance)+.5*gam*gain*r*r
                return expit(logit)*gain*r
            amplitude = abs(target[i, c])
            upper = amplitude/(gain*mean_at(1e-30)/(gain*1e-30))
            root = brentq(lambda r: mean_at(r)-amplitude, 0., upper, xtol=1e-14, rtol=1e-14)
            np.testing.assert_allclose(abs(message[i, c]), root, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize('bad_initial', [np.array([np.nan]*12), np.zeros(13)])
def test_invalid_initial_mean_is_rejected(bad_initial):
    rng = np.random.default_rng(902)
    x = rng.normal(size=(12, 8))
    groups = np.repeat(np.arange(4), 3)
    with pytest.raises(ValueError, match='initial_mean'):
        fit_grouped_ss_vamp(GenotypeProducts(x, 5), rng.normal(size=8), groups,
            np.full(4, .1), np.full(4, 3), [.5], SampleSpectrum.from_genotype(x),
            initial_mean=bad_initial)


def test_free_m_step_against_joint_numerical_optimizer():
    a, b, count = 20., .7, 100.
    pi, v = map(float, unconstrained_ss_m_step(a, b, count))
    def objective(z):
        probability, variance = expit(z[0]), np.exp(z[1])
        return -(a*np.log(probability)+(count-a)*np.log1p(-probability)
                 -.5*a*np.log(variance)-b/(2*variance))
    optimum = minimize(objective, [0., -2.], method='BFGS', options={'gtol': 1e-8})
    np.testing.assert_allclose([pi, v], [expit(optimum.x[0]), np.exp(optimum.x[1])], rtol=1e-6)
    empty_pi, empty_v = map(float, unconstrained_ss_m_step(0., 0., 100.))
    assert empty_pi == pytest.approx(1e-8) and empty_v == pytest.approx(1e-16)


def test_free_em_orthogonal_posterior_and_native_zero_group_is_open():
    rng = np.random.default_rng(541)
    n, p = 160, 80
    basis, _ = np.linalg.qr(rng.normal(size=(n, p)))
    x = np.sqrt(n)*basis.T
    groups = np.repeat(np.arange(4), 20)
    beta = np.zeros(p)
    beta[[1, 4, 12, 22, 27, 43, 46, 61, 67, 75]] = rng.normal(scale=.3, size=10)
    noise = .2
    y = x.T@beta+rng.normal(scale=np.sqrt(noise), size=n)
    theta = np.array([0., .3, .3, .3])
    result = fit_grouped_ss_vamp(GenotypeProducts(x, 13), y, groups, theta, np.full(4, 20),
        [noise], SampleSpectrum.from_genotype(x),
        VAMPConfig(variance_constraint=False, max_iterations=1500, min_iterations=20,
                   em_damping=.15, tolerance=1e-6, em_tolerance=1e-6))
    assert result.status == ['converged']
    assert np.any(result.mean[groups == 0, 0] != 0)
    expected = spike_slab_denoise(x@y/n, n/noise, result.pi[groups, 0], result.slab_variance[groups, 0])
    for actual, reference in zip([result.mean[:, 0], result.variance[:, 0], result.pip[:, 0]], expected[:3]):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-8)
    for g in range(4):
        mask = groups == g
        a = result.pip[mask, 0].sum()
        b = np.sum(result.variance[mask, 0]+result.mean[mask, 0]**2)
        np.testing.assert_allclose([result.pi[g, 0], result.slab_variance[g, 0]],
                                  [a/20, b/a], rtol=2e-6)
    assert np.max(np.abs(result.pi[:, 0]*result.slab_variance[:, 0]*20-theta)) > .1
