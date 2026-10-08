"""Sufficient statistics for sparse refinement under a frozen Gaussian background.

The backend applies V = X diag(c) X' + noise I and solves its systems without
constructing V. Only X_S and V^-1 X_S are retained. A validation matrix is not
built; validation predictions are evaluated after the sparse mean is known.
"""
from __future__ import annotations

import time
import numpy as np
def marginal_statistics(x, vinv_x, residual, dual, old_effect):
    """Likelihood of selected effects after integrating the Gaussian background.

    residual excludes ALL original sparse effects and fixed covariates, but
    retains the background; dual = V^-1 residual. The selected old effects
    are added back to construct the likelihood for their replacement.
    """
    x, vinv_x = np.asarray(x, float), np.asarray(vinv_x, float)
    residual, dual, old_effect = map(lambda a: np.asarray(a, float),
                                    (residual, dual, old_effect))
    raw = x.T @ vinv_x
    scale = max(float(np.linalg.norm(raw)), 1e-30)
    asymmetry = float(np.linalg.norm(raw - raw.T) / scale)
    if asymmetry > 2e-4:
        raise ValueError(f'Covariance solve asymmetry: {asymmetry}')
    g = (raw + raw.T) * .5
    eigen, vectors = np.linalg.eigh(g)
    maximum = max(float(eigen.max(initial=0.)), 1e-30)
    if eigen.min(initial=0.) < -2e-5 * maximum:
        raise ValueError('Indefinite marginal information beyond numerical tolerance')
    if np.any(eigen < 0.):
        g = (vectors * np.maximum(eigen, 0.)) @ vectors.T
    score = x.T @ dual
    h = score + g @ old_effect
    target = residual + x @ old_effect
    ss = float(target @ (dual + vinv_x @ old_effect))
    return dict(gram=g, score=h, correction_score=score, response_ss=ss,
                asymmetry=asymmetry, min_eigenvalue=float(eigen.min(initial=0.)),
                rank=int(np.sum(eigen > maximum * 1e-7)))

from .precision_refined_covariance import solve_refined_covariance


def conditional_gaussian(backend, residual, variance, noise, *, initial=None):
    """A cached start is corrected against the same original-system residual."""
    variance = np.asarray(variance, float)
    backend.prepare_shared(variance, noise)
    dual, diagnostic = solve_refined_covariance(
        backend.apply_exact, backend.apply_fast, residual, backend.shared_diagonal,
        initial=initial, tolerance=1e-10, correction_tolerance=1e-3,
        max_iter=500, max_refinements=8)
    effect = variance[:, None] * backend.scores(dual)
    return dual, effect, diagnostic


def build_sparse_statistics(x, indices, old_effect, y, covariates, variance, noise,
                            backend, *, covariate_effect=None, block_size=64, progress=None):
    """Build the integrated likelihood using fixed covariates or conditional GLS.

    x has SNPs on rows. Passing covariate_effect preserves an existing nuisance
    fit (the 10k stage); None reconstructs it by GLS (the 8k stage). Every solve
    retains the 1e-10 original-system residual criterion. Neither a full GRM nor
    a full marker Gram matrix is formed.
    """
    began = time.monotonic()
    indices = np.asarray(indices)
    old_effect = np.asarray(old_effect, float); y = np.asarray(y, float)
    covariates = np.asarray(covariates, float); variance = np.asarray(variance, float)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= x.shape[0])):
        raise ValueError('Sparse SNP indices must be unique valid integers')
    if (len(x.shape) != 2 or y.shape != (x.shape[1],) or old_effect.shape != indices.shape
            or covariates.ndim != 2 or covariates.shape[0] != len(y)
            or variance.shape != (len(x),) or np.any(variance < 0)
            or not np.isfinite(noise) or noise <= 0
            or not isinstance(block_size, (int, np.integer)) or block_size < 1):
        raise ValueError('Invalid or misaligned frozen model inputs')
    if not all(np.isfinite(a).all() for a in (y, old_effect, covariates, variance)):
        raise ValueError('Frozen model inputs must be finite')
    xs = np.asarray(x[indices], float).T
    raw = y - xs @ old_effect
    diagnostics = []
    if covariate_effect is None and covariates.shape[1]:
        solved, record = backend.solve_precise(np.column_stack((raw, covariates)), variance, noise)
        diagnostics.append(record)
        gamma = np.linalg.solve(covariates.T @ solved[:, 1:], covariates.T @ solved[:, 0])
        dual = solved[:, 0] - solved[:, 1:] @ gamma
    else:
        gamma = (np.zeros(covariates.shape[1]) if covariate_effect is None
                 else np.asarray(covariate_effect, float))
        if gamma.shape != (covariates.shape[1],) or not np.isfinite(gamma).all():
            raise ValueError('Covariate effects are not aligned')
        dual, record = backend.solve_precise(raw - covariates @ gamma, variance, noise)
        dual = np.asarray(dual, float).reshape(-1)
        diagnostics.append(record)
    residual = raw - covariates @ gamma
    # The linear combination of accurately solved GLS columns can lose relative
    # accuracy under cancellation. Correct it if needed; do not relax the gate.
    actual = np.asarray(backend.apply_exact(dual[:, None]), float)[:, 0]
    norm = float(np.linalg.norm(residual))
    relative = float(np.linalg.norm(actual-residual)/norm) if norm else float(np.linalg.norm(actual))
    if not np.isfinite(relative) or relative > 1e-10:
        backend.prepare_shared(variance, noise)
        dual, record = solve_refined_covariance(
            backend.apply_exact, backend.apply_fast, residual, backend.shared_diagonal,
            initial=dual, tolerance=1e-10, correction_tolerance=1e-3,
            max_iter=500, max_refinements=8)
        diagnostics.append(record)
    inverse = np.empty_like(xs)
    for lo in range(0, len(indices), block_size):
        hi = min(lo+block_size, len(indices))
        inverse[:, lo:hi], record = backend.solve_precise(xs[:, lo:hi], variance, noise)
        diagnostics.append(record)
        if progress: progress(dict(columns=hi, total=len(indices), diagnostic=record))
    if any(not d.get('original_system_verified', False)
           or d['maximum_relative_residual'] > 1e-10 for d in diagnostics):
        raise ArithmeticError('Sparse sufficient-statistic covariance solve did not qualify')
    statistics = marginal_statistics(xs, inverse, residual, dual, old_effect)
    data = dict(snps=indices.copy(), old=old_effect.copy(), gamma=gamma, dual=dual,
                vinv_x=inverse, gram=statistics['gram'], score=statistics['score'],
                response_ss=statistics['response_ss'])
    record = dict(numeric_accepted=True, covariance_solves=diagnostics,
                  information_asymmetry=statistics['asymmetry'],
                  seconds=time.monotonic()-began, explicit_grm=False,
                  full_validation_operator_constructed=False)
    return data, record
