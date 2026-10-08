"""Independent dense/joint-equation oracles for the matrix-free P workflow."""
import numpy as np
import pytest

from GPU_REML.profiled_prediction_refinement import (
    CovariateProjection, profiled_sparse_statistics, profiled_gaussian,
    project_inverse_quadratics, replace_exact_profiled_quadratics,
    fit_profiled_candidates)
from GPU_REML.matrix_free_sparse_statistics import marginal_statistics
class Dense:
    def __init__(self,x):self.x=x
    def prepare_shared(self,c,noise):
        self.v=(self.x*c)@self.x.T+noise*np.eye(len(self.x))
    def apply_exact(self,u):return self.v@u
    def solve_precise(self,rhs,c,noise):
        self.prepare_shared(c,noise)
        return np.linalg.solve(self.v,rhs),dict(original_system_verified=True,maximum_relative_residual=1e-15)
    def gaussian(self,rhs,c,noise):
        dual,rec=self.solve_precise(rhs,c,noise);return dual,c[:,None]*self.scores(dual),rec
    def scores(self,dual):return self.x.T@(dual[:,None] if dual.ndim==1 else dual)
    def predict(self,beta):return self.x@beta


def problem(k=3):
    rng = np.random.default_rng(107)
    x = rng.normal(size=(23, 17)); xv = rng.normal(size=(13, 17))
    c = np.column_stack((np.ones(len(x)), x[:, 0]+rng.normal(size=len(x))*.4,
                         rng.normal(size=len(x))))[:, :k]
    cv = np.column_stack((np.ones(len(xv)), xv[:, 0]+rng.normal(size=len(xv))*.4,
                          rng.normal(size=len(xv))))[:, :k]
    y = rng.normal(size=len(x)); variance = np.linspace(.025, .17, x.shape[1]); noise = .7
    v = (x*variance) @ x.T+noise*np.eye(len(x)); w = np.linalg.inv(v)
    a = c.T @ w @ c
    p = w-w @ c @ np.linalg.solve(a, c.T @ w) if k else w
    s = np.array([0, 3, 7, 9]); old = np.array([.03, -.07, .04, .11])
    gamma = np.linspace(-.3, .2, k)
    return x, xv, c, cv, y, variance, noise, w, p, s, old, gamma


def original(x, c, y, w, s, old, gamma):
    residual = y-c @ gamma-x[:, s] @ old
    stats = marginal_statistics(x[:, s], w @ x[:, s], residual, w @ residual, old)
    return dict(stats, old=old, dual=w @ residual, vinv_x=w @ x[:, s], gamma=gamma)


@pytest.mark.parametrize('k', [0, 1, 3])
def test_schur_statistics_match_dense_profiled_loss_and_are_gamma_invariant(k):
    x, xv, c, cv, y, d, noise, w, p, s, old, gamma = problem(k)
    projection = CovariateProjection(c, w @ c)
    out = profiled_sparse_statistics(original(x, c, y, w, s, old, gamma), projection)
    np.testing.assert_allclose(out['gram'], x[:, s].T @ p @ x[:, s], atol=1e-12)
    np.testing.assert_allclose(out['score'], x[:, s].T @ p @ y, atol=1e-12)
    np.testing.assert_allclose(out['response_ss'], y @ p @ y, atol=1e-12)
    np.testing.assert_allclose(out['projected_old_dual'], p @ (y-x[:, s] @ old), atol=1e-12)
    np.testing.assert_allclose(out['projected_x'], p @ x[:, s], atol=1e-12)
    other = profiled_sparse_statistics(original(x, c, y, w, s, old, gamma+4), projection)
    np.testing.assert_allclose(other['score'], out['score'], atol=1e-12)
    np.testing.assert_allclose(other['response_ss'], out['response_ss'], atol=1e-11)
    alpha = old*3
    residual = y-x[:, s] @ alpha
    fitted_gamma = projection.coefficients(w @ residual)
    assert residual @ p @ residual == pytest.approx(
        (residual-c @ fitted_gamma) @ w @ (residual-c @ fitted_gamma), abs=1e-12)
    assert residual @ p @ residual == pytest.approx(
        float(out['response_ss'])-2*out['score'] @ alpha+alpha @ out['gram'] @ alpha, abs=1e-12)


def test_profiled_gaussian_matches_joint_henderson_equations_and_covar_shift():
    x, xv, c, cv, y, d, noise, w, p, s, old, gamma = problem()
    device = Dense(x); fit = profiled_gaussian(device, y, c, d, noise)
    design = np.column_stack((c, x))
    precision = design.T @ design/noise
    precision[len(gamma):, len(gamma):] += np.diag(1/d)
    joint = np.linalg.solve(precision, design.T @ y/noise)
    np.testing.assert_allclose(fit['gamma'], joint[:len(gamma)], atol=1e-12)
    np.testing.assert_allclose(fit['effect'], joint[len(gamma):], atol=1e-12)
    np.testing.assert_allclose(fit['dual'], p @ y, atol=1e-12)
    np.testing.assert_allclose(c.T @ fit['dual'], 0, atol=1e-12)
    shifted = profiled_gaussian(device, y+c @ gamma, c, d, noise, projection=fit['projection'])
    np.testing.assert_allclose(shifted['gamma'], fit['gamma']+gamma, atol=1e-12)
    np.testing.assert_allclose(shifted['effect'], fit['effect'], atol=1e-12)
    covariance = np.linalg.inv(precision)[len(gamma):, len(gamma):]
    np.testing.assert_allclose(np.diag(covariance), d-d*d*np.sum(x*(p @ x), axis=0), atol=1e-12)


def test_rank_correction_preserves_probe_errors_and_exact_repairs_use_p():
    x, xv, c, cv, y, d, noise, w, p, s, old, gamma = problem()
    projection = CovariateProjection(c, w @ c)
    correction = projection.diagonal_correction(x.T @ w @ c)
    h = np.sum(x*(w @ x), axis=0); hp = np.sum(x*(p @ x), axis=0)
    np.testing.assert_allclose(h-correction, hp, atol=1e-12)
    error = np.linspace(-.03, .03, len(d))
    stats = dict(h=h.copy(), h_blocks=np.vstack((h+error, h-error)),
                 prior_variance=d, theta=np.array([noise]), exact_indices=np.array([], int))
    out = project_inverse_quadratics(stats, correction)
    np.testing.assert_allclose(out['h_blocks'], np.vstack((hp+error, hp-error)), atol=1e-12)
    replace_exact_profiled_quadratics(out, s, Dense(x), x.T, projection)
    np.testing.assert_allclose(out['h_blocks'][:, s], np.tile(hp[s], (2, 1)), atol=1e-12)
    np.testing.assert_array_equal(stats['h'], h)


def gates():
    return dict(training_half_rms_gate=1e-10, validation_half_rms_gate=1e-10,
                validation_half_mse_gate=1e-10)


def test_background_calibration_interpolates_nuisance_and_refit_freezes_rule():
    x, xv, c, cv, y, d, noise, w, p, s, old, gamma = problem()
    device = Dense(x)
    base = profiled_gaussian(device, y, c, d, noise)
    candidate = profiled_gaussian(device, y, c, d*.45, noise)
    predict = lambda f: xv @ f['effect']+cv @ f['gamma']
    expected_rho = .63
    target = predict(base)+expected_rho*(predict(candidate)-predict(base))
    models = [dict(A=d*.45, B=d*2) for _ in range(3)]
    fit, record = fit_profiled_candidates(device, y, c, base, models, noise, ['A', 'B'], gates(),
        validation_products=lambda b: xv @ b, validation_covariates=cv,
        validation_base=predict(base), validation_y=target)
    assert fit['numeric_accepted'] and fit['rule']['allocation'] == 'A'
    assert fit['rule']['background_rho'] == pytest.approx(expected_rho, abs=1e-12)
    np.testing.assert_allclose(predict(fit), target, atol=1e-12)
    np.testing.assert_allclose(fit['gamma'], base['gamma']+expected_rho*(candidate['gamma']-base['gamma']), atol=1e-12)
    assert np.max(abs(candidate['gamma']-base['gamma'])) > .001
    final, _ = fit_profiled_candidates(device, y, c, base, models, noise, ['A', 'B'], gates(), rule=fit['rule'])
    np.testing.assert_allclose(predict(final), target, atol=1e-12)
    with pytest.raises(ValueError, match='validation data'):
        fit_profiled_candidates(device, y, c, base, models, noise, ['A', 'B'], gates(),
                                rule=fit['rule'], validation_y=target)


def test_projection_rejects_rank_deficiency_and_unqualified_solves():
    with pytest.raises(ValueError, match='full rank'):
        CovariateProjection(np.ones((5, 2)), np.ones((5, 2)))
    x, xv, c, cv, y, d, noise, w, p, s, old, gamma = problem()
    class Bad(Dense):
        def solve_precise(self, *args):
            value, record = super().solve_precise(*args)
            return value, dict(record, maximum_relative_residual=1e-5)
    with pytest.raises(ArithmeticError, match='did not qualify'):
        profiled_gaussian(Bad(x), y, c, d, noise)


def test_cross_precision_audit_is_invariant_to_units_and_defined_at_orthogonality():
    from GPU_REML.profiled_prediction_refinement import covariate_cross_diagnostic
    a=np.diag([2.,5.]);j=np.diag([7.,11.,13.]);h=np.zeros((2,3))
    perturbation=np.array([[1.,-2.,3.],[4.,-5.,6.]])*1e-6
    base=covariate_cross_diagnostic(h,h+perturbation,a,j)
    rc=np.array([.01,100.]);rx=np.array([9.,.05,2.])
    rescaled=covariate_cross_diagnostic(h,(h+perturbation)*rc[:,None]*rx[None,:],
        a*rc[:,None]*rc[None,:],j*rx[:,None]*rx[None,:])
    assert base['max_information_normalized_error']==pytest.approx(rescaled['max_information_normalized_error'])
    assert base['max_information_normalized_error']<2e-6
