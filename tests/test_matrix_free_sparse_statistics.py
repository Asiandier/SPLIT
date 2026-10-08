import numpy as np
from numpy.testing import assert_allclose
from GPU_REML.matrix_free_sparse_statistics import build_sparse_statistics, conditional_gaussian


class DenseOracleBackend:
    """Only a toy oracle forms a covariance; production supplies chunked products."""
    def __init__(self, x): self.x = x
    def prepare_shared(self, variance, noise):
        self.v = (self.x * variance) @ self.x.T + noise * np.eye(len(self.x))
        self.shared_diagonal = np.diag(self.v)
    def apply_exact(self, value): return self.v @ value
    apply_fast = apply_exact
    def scores(self, dual):
        if dual.ndim == 1: dual = dual[:, None]
        return self.x.T @ dual
    def solve_precise(self, rhs, variance, noise):
        self.prepare_shared(variance, noise)
        answer = np.linalg.solve(self.v, rhs)
        return answer, dict(original_system_verified=True,
                            maximum_relative_residual=float(np.linalg.norm(self.v @ answer-rhs)/np.linalg.norm(rhs)) if np.linalg.norm(rhs) else 0.)


def test_integrated_likelihood_matches_direct_gaussian_density_and_gls():
    rng = np.random.default_rng(105)
    x = rng.normal(size=(31, 43)); x[:, 9] = x[:, 5]
    y = rng.normal(size=31); cov = np.column_stack((np.ones(31), rng.normal(size=31)))
    support = np.array([13, 5, 9, 19, 27]); old = rng.normal(size=5)
    variance = rng.uniform(0, .1, size=43); variance[::2] = 0
    backend = DenseOracleBackend(x)
    for fixed in [None, np.array([.3, -.2])]:
        data, record = build_sparse_statistics(x.T, support, old, y, cov, variance, .2,
            backend, covariate_effect=fixed, block_size=2)
        v = (x * variance) @ x.T + .2 * np.eye(31)
        inverse = np.linalg.inv(v)
        gamma = (np.linalg.solve(cov.T @ inverse @ cov, cov.T @ inverse @ (y-x[:, support]@old))
                 if fixed is None else fixed)
        assert_allclose(data['gamma'], gamma, rtol=3e-13, atol=3e-13)
        for _ in range(3):
            alpha = rng.normal(size=5)
            residual = y-cov@gamma-x[:, support]@alpha
            direct = residual @ inverse @ residual
            reduced = data['response_ss']-2*alpha@data['score']+alpha@data['gram']@alpha
            assert_allclose(reduced, direct, rtol=3e-13, atol=3e-13)
        assert not record['full_validation_operator_constructed']
        assert data['vinv_x'].shape == (31, 5)
        new = rng.normal(size=5)
        rhs = y-cov@gamma-x[:, support]@new
        initial = data['dual']-data['vinv_x']@(new-old)
        dual, effect, diagnostic = conditional_gaussian(backend, rhs, variance, .2,
                                                       initial=initial.astype(np.float32))
        expected = np.linalg.solve(v, rhs)
        assert_allclose(dual, expected, rtol=1e-9, atol=1e-9)
        assert_allclose(effect[:, 0], variance*(x.T@expected), rtol=1e-9, atol=1e-10)
        assert diagnostic['maximum_relative_residual'] <= 1e-10


def test_empty_support_and_zero_phenotype_are_well_defined():
    x = np.arange(35., dtype=float).reshape(7, 5)/35
    data, record = build_sparse_statistics(x.T, np.array([], dtype=int), np.zeros(0),
        np.zeros(7), np.empty((7, 0)), np.zeros(5), .7, DenseOracleBackend(x))
    assert data['gram'].shape == (0, 0) and data['score'].shape == (0,)
    assert_allclose(data['dual'], 0, atol=0)
    assert data['response_ss'] == 0 and record['numeric_accepted']
