import numpy as np

from GPU_REML.precision_refined_covariance import solve_refined_covariance


def test_refined_solve_matches_dense_reference_for_multiple_rhs():
    rng = np.random.default_rng(1028)
    matrix = rng.normal(size=(13, 13))
    covariance = matrix.T @ matrix + np.eye(13)
    rhs = rng.normal(size=(13, 3))

    solution, diagnostic = solve_refined_covariance(
        lambda value: covariance @ value,
        lambda value: covariance @ value,
        rhs,
        np.diag(covariance),
        tolerance=1e-10,
    )

    np.testing.assert_allclose(solution, np.linalg.solve(covariance, rhs), rtol=1e-9, atol=1e-9)
    assert diagnostic["original_system_verified"]
    assert diagnostic["maximum_relative_residual"] < 1e-10
