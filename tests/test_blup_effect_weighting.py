import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from GPU_REML.blup_effect_weighting import effect_power_variance
from GPU_REML.iterative_weighted_ridge import profiled_validation


@pytest.mark.parametrize('power', [1., 2., 4.])
def test_weights_against_unscaled_formula_and_group_units(power):
    rng = np.random.default_rng(81008)
    pilot = rng.normal(size=45)
    groups = np.arange(45) % 4
    c = np.array([.01, .03, 0., .002])[groups]
    pilot[c == 0] = 0.
    norms = rng.uniform(2, 40, len(c))
    d, _ = effect_power_variance(pilot, c, norms, groups, power)
    expected = np.zeros_like(c)
    for group in [0, 1, 3]:
        use = groups == group
        raw = c[use]*np.abs(pilot[use])**power
        expected[use] = raw*(c[use]@norms[use])/(raw@norms[use])
    np.testing.assert_allclose(d, expected, rtol=2e-14, atol=0)
    scales = np.array([1e-180, -1e170, 1., -4.])[groups]
    changed, _ = effect_power_variance(pilot*scales, c, norms, groups, power)
    np.testing.assert_allclose(changed, expected, rtol=2e-14, atol=0)


def test_zero_pilot_and_zero_budget_do_not_create_information():
    c = np.array([1., 1., 0., 0.])
    groups = np.array([0, 0, 1, 1])
    d, info = effect_power_variance(np.zeros(4), c, np.ones(4), groups)
    np.testing.assert_array_equal(d, c)
    assert info['groups'][0]['zero_score_fallback']
    assert not info['uses_inverse_quadratics']


def test_repeated_update_dense_primal_solution_and_validation_oracle():
    rng = np.random.default_rng(34)
    x = rng.normal(size=(27, 15)); x[:, 4] = x[:, 3]
    norms = np.sum(x*x, axis=0)
    groups = np.arange(15) % 3
    c = np.array([.01, .04, 0.])[groups]
    t = .6; r = rng.normal(size=len(x))
    original = c*(x.T@np.linalg.solve((x*c)@x.T+t*np.eye(len(x)), r))
    pilot = original
    for _ in range(3):
        d, _ = effect_power_variance(pilot, c, norms, groups, 2)
        dual = d*(x.T@np.linalg.solve((x*d)@x.T+t*np.eye(len(x)), r))
        active = d > 0
        primal = np.zeros_like(c)
        xx = x[:, active]
        primal[active] = np.linalg.solve(xx.T@xx/t+np.diag(1/d[active]), xx.T@r/t)
        np.testing.assert_allclose(dual, primal, atol=1e-13, rtol=2e-12)
        for g in range(3):
            use = groups == g
            assert d[use]@norms[use] == pytest.approx(c[use]@norms[use], rel=2e-14)
        pilot = dual
    xv = rng.normal(size=(80, 15))
    base = xv@original; direction = xv@(pilot-original)
    yv = base+.7*direction+rng.normal(size=80)*.001
    actual = profiled_validation(yv, base, direction)
    oracle = minimize_scalar(lambda rho: np.mean((yv-base-rho*direction)**2),
                              bounds=(0, 1), method='bounded')
    assert actual['rho'] == pytest.approx(oracle.x, abs=1e-6)


@pytest.mark.parametrize('power', [0., -1., np.nan, np.inf, True])
def test_invalid_exponents_fail(power):
    with pytest.raises(ValueError):
        effect_power_variance([.1, .2], [1., 1.], [3., 3.], np.array([0, 0]), power)


def test_effect_at_zero_prior_is_rejected():
    with pytest.raises(ValueError):
        effect_power_variance([1., 0.], [0., 1.], [3., 3.], np.array([0, 0]))
