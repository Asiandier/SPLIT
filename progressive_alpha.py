"""Joint OLS corrections of growing prefixes ranked by native LASSO effect.

Background SS and covariate effects are frozen. Validation selects only the
prefix count; the combined-sample refit uses its own support and ranking.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_solve, solve_triangular


@dataclass
class ProgressivePath:
    coefficients: np.ndarray
    order: np.ndarray
    metrics: np.ndarray
    diagnostics: dict


@dataclass
class ProgressiveAlphaResult:
    alpha: np.ndarray
    old_alpha: np.ndarray
    path: ProgressivePath
    selected_k: int
    training_sparse_score: np.ndarray
    training_full_score: np.ndarray
    summary: dict
    validation_corr2: np.ndarray | None = None
    validation_mse: np.ndarray | None = None


def fit_path(g, u, pilot, support=None, max_k=None, rank_tol=1e-10,
             normal_tol=1e-6):
    """Solve every requested prefix from FP64 Gram and response products.

    Dependent columns enter the prefix without adding a prediction direction.
    The correction has minimum Euclidean norm within the complete prefix.
    ``support`` supplies source SNP indices to break equal-magnitude ties.
    Metrics columns are k, rank, training MSE reduction, sparse squared term,
    relative normal-equation error, and the new Cholesky pivot.
    """
    g, u, pilot = np.asarray(g, float), np.asarray(u, float), np.asarray(pilot, float)
    if pilot.ndim != 1:
        raise ValueError('pilot must be a vector')
    s = len(pilot)
    support = np.arange(s) if support is None else np.asarray(support)
    if (g.shape != (s, s) or u.shape != (s,) or support.shape != (s,)
            or not np.issubdtype(support.dtype, np.integer)
            or np.any(support < 0) or len(np.unique(support)) != s
            or not all(np.all(np.isfinite(x)) for x in (g, u, pilot))):
        raise ValueError('Finite aligned Gram, response, effects and unique SNP indices required')
    if not np.allclose(g, g.T, rtol=1e-12, atol=1e-12):
        raise ValueError('Gram must be symmetric')
    if not 0 < rank_tol < 1 or not 0 < normal_tol < 1:
        raise ValueError('Invalid rank or normal-equation tolerance')
    if max_k is not None and (not isinstance(max_k, (int, np.integer)) or max_k < 0):
        raise ValueError('k must be a nonnegative integer')
    stop = s if max_k is None else min(int(max_k), s)
    order = np.lexsort((support, -abs(pilot)))
    gp = g[np.ix_(order, order)]
    gpilot = g@pilot
    b, gp_old = (u-gpilot)[order], gpilot[order]
    coefficients = np.tile(pilot, (stop+1, 1))
    lower = np.zeros((stop, stop))
    # X_prefix = X_independent C; H=(C C.T)^-1 gives the minimum-norm correction.
    # Full-rank prefixes need only their Cholesky factor. Allocate the two
    # additional quadratic buffers only when a dependent column enters.
    c = h = None
    basis, dependent = [], False
    qold = float(pilot@g@pilot)
    rows, min_pivot = [(0, 0, 0., qold, 0., 0.)], np.inf
    for k in range(1, stop+1):
        j, rank = k-1, len(basis)
        z = (solve_triangular(lower[:rank, :rank], gp[basis, j], lower=True,
                             check_finite=False) if rank else np.empty(0))
        pivot = float(gp[j, j]-z@z)
        threshold = rank_tol*max(float(gp[j, j]), 1e-30)
        if pivot > threshold:
            lower[rank, :rank], lower[rank, rank] = z, np.sqrt(pivot)
            basis.append(j)
            if dependent:
                c[rank, j], h[rank, rank] = 1., 1.
            rank += 1
            min_pivot = min(min_pivot, pivot)
        else:
            if pivot < -1e-8*max(float(gp[j, j]), 1e-30):
                raise RuntimeError('negative Cholesky pivot')
            if not dependent:
                c, h = np.zeros((stop, stop)), np.zeros((stop, stop))
                c[np.arange(rank), basis] = 1.
                h[:rank, :rank] = np.eye(rank)
                dependent = True
            if rank:
                d = solve_triangular(lower[:rank, :rank].T, z, lower=False, check_finite=False)
                c[:rank, j] = d
                hd = h[:rank, :rank]@d
                h[:rank, :rank] -= np.outer(hd, hd)/(1+d@hd)
        if rank:
            delta_basis = cho_solve((lower[:rank, :rank], True), b[basis], check_finite=False)
            delta = c[:rank, :k].T@(h[:rank, :rank]@delta_basis) if dependent else delta_basis
        else:
            delta = np.zeros(k)
        coefficients[k, order[:k]] += delta
        normal = float(np.max(abs(gp[:k, :k]@delta-b[:k]))/
                       max(float(np.max(abs(b[:k]))), 1e-12))
        if normal > normal_tol:
            raise RuntimeError(f'normal equations failed at k={k}: {normal}')
        reduction = float(delta@b[:k])
        q = qold+2*(delta@gp_old[:k])+reduction
        if reduction < -1e-10 or reduction < rows[-1][2]-1e-8:
            raise RuntimeError('non-monotone training projection')
        rows.append((k, rank, reduction, q, normal, pivot))
    metrics = np.asarray(rows)
    diagnostics = dict(status='solved', path_points=len(rows), final_active_count=stop,
        final_rank=len(basis), final_dependent_columns=stop-len(basis),
        maximum_normal_equation_error=float(metrics[:, 4].max()),
        minimum_independent_pivot=None if not np.isfinite(min_pivot) else float(min_pivot),
        minimum_training_reduction_increment=float(np.diff(metrics[:, 2]).min()) if stop else 0.,
        rank_pivot_relative_tolerance=rank_tol,
        minimum_norm='coefficient change within selected prefix')
    return ProgressivePath(coefficients, order, metrics, diagnostics)


def select_prefix(path, design, fixed_score, response, tie_tolerance=1e-12):
    """Select full-prediction validation corr²; ties prefer the smaller prefix."""
    design, fixed_score, response = (np.asarray(x, float) for x in (design, fixed_score, response))
    if (design.ndim != 2 or design.shape != (len(response), path.coefficients.shape[1])
            or fixed_score.shape != response.shape or response.ndim != 1 or len(response) < 2
            or not all(np.all(np.isfinite(x)) for x in (design, fixed_score, response))
            or not np.isfinite(tie_tolerance) or tie_tolerance < 0):
        raise ValueError('Aligned finite validation design, fixed score and response required')
    yc = response-response.mean()
    yenergy = float(yc@yc)
    if yenergy == 0:
        raise ValueError('Validation phenotype must have positive variance')
    prediction = fixed_score[:, None]+design@path.coefficients.T
    centered = prediction-prediction.mean(axis=0)
    denominator = yenergy*np.sum(centered*centered, axis=0)
    corr2 = np.divide((yc@centered)**2, denominator, out=np.zeros(len(denominator)), where=denominator > 0)
    corr2 = np.clip(corr2, 0., 1.)
    mse = np.mean((response[:, None]-prediction)**2, axis=0)
    k = int(np.flatnonzero(corr2 >= corr2.max()-tie_tolerance)[0])
    return k, corr2, mse


def refine_alpha(*, design, y, covariates, gamma, background_score, alpha,
                 source_indices, selected_k=None, validation_design=None,
                 validation_fixed_score=None, validation_response=None):
    """Refit alpha conditional on frozen SS and gamma, without a Q constraint.

    With no selected_k, validation is required and selects the prefix count.
    With a frozen count, no validation argument is accepted; the refit ranks
    the current sample set's native LASSO effects and caps k at its support size.
    """
    x, y, alpha = np.asarray(design, float), np.asarray(y, float), np.asarray(alpha, float)
    gamma, background = np.asarray(gamma, float), np.asarray(background_score, float)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError('Training phenotype must be a nonempty vector')
    c = np.empty((len(y), 0)) if covariates is None else np.asarray(covariates, float)
    if (alpha.ndim != 1 or x.shape != (len(y), len(alpha)) or c.ndim != 2
            or c.shape[0] != len(y) or gamma.shape != (c.shape[1],)
            or background.shape != y.shape
            or not all(np.all(np.isfinite(v)) for v in (x, y, alpha, c, gamma, background))):
        raise ValueError('Finite aligned training inputs required')
    validation = (validation_design, validation_fixed_score, validation_response)
    if selected_k is None and any(v is None for v in validation):
        raise ValueError('Validation inputs are required to select k')
    if selected_k is not None and any(v is not None for v in validation):
        raise ValueError('Frozen-count refit must not receive validation data')
    fixed = c@gamma+background
    r_alpha = y-fixed
    g, u = x.T@x/len(y), x.T@r_alpha/len(y)
    path = fit_path(g, u, alpha, support=source_indices, max_k=selected_k)
    corr2 = mse = None
    if selected_k is None:
        k, corr2, mse = select_prefix(path, *validation)
        mode = 'validation_selection'
    else:
        k, mode = min(int(selected_k), len(alpha)), 'frozen_count_refit'
    updated = path.coefficients[k].copy()
    sparse, old_sparse = x@updated, x@alpha
    full, old_full = fixed+sparse, fixed+old_sparse
    np.testing.assert_array_equal(updated[path.order[k:]], alpha[path.order[k:]])
    summary = dict(requested=True, status='solved', method='progressive_ols_alpha', mode=mode,
        selected_k=k, transferred_k=selected_k, selected_rank=int(path.metrics[k, 1]),
        support_size=len(alpha), background_fixed=True, gamma_fixed=True,
        variance_components_fixed=True, Q_constraint=False, test_labels_used=False,
        validation_labels_used_for_fitting=False,
        validation_labels_used_for_selection=selected_k is None,
        selection_metric='full_prediction_corr2', selection_tie_tolerance=1e-12,
        ranking='descending absolute native LASSO alpha; ties by source SNP index',
        unchanged_outside_refit_set=True,
        old_sparse_squared_term=float(np.mean(old_sparse**2)),
        new_sparse_squared_term=float(np.mean(sparse**2)),
        old_full_squared_term=float(np.mean(old_full**2)),
        new_full_squared_term=float(np.mean(full**2)),
        old_training_mse=float(np.mean((y-old_full)**2)),
        new_training_mse=float(np.mean((y-full)**2)), diagnostics=path.diagnostics)
    if corr2 is not None:
        summary.update(baseline_validation_corr2=float(corr2[0]), validation_corr2=float(corr2[k]))
    return ProgressiveAlphaResult(updated, alpha.copy(), path, k, sparse, full, summary, corr2, mse)
