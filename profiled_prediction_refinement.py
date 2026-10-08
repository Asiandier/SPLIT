"""Covariate profiling for sparse and Gaussian prediction refinement.

P = W - W C (C' W C)^-1 C' W, W = V^-1. Only solves and small
covariate matrices are needed. No sample covariance or P is constructed.
Covariates have a flat prior; V is fixed within each sparse EM fit.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def covariate_cross_diagnostic(direct, cached, covariate_information, marker_information):
    """Measure bilinear error on the Cauchy-Schwarz information scale.

    Dividing by the cross-product itself is invalid near orthogonality. This
    normalization is invariant to rescaling either a covariate or an SNP.
    It audits reuse of legacy V^-1 X caches; it is not a covariance solve gate.
    """
    direct, cached = np.asarray(direct, float), np.asarray(cached, float)
    a = np.diag(np.asarray(covariate_information, float))
    j = np.diag(np.asarray(marker_information, float))
    if direct.shape != cached.shape or direct.shape != (len(a), len(j)):
        raise ValueError('Aligned covariate-by-SNP information required')
    if np.any(a <= 0) or np.any(j <= 0) or not np.isfinite(direct).all() or not np.isfinite(cached).all():
        raise ValueError('Finite cross-products and positive diagonal information required')
    delta = abs(direct-cached)
    scale = np.sqrt(a[:, None]*j[None, :])
    return dict(max_absolute_error=float(np.max(delta, initial=0.)),
                max_information_normalized_error=float(np.max(delta/scale, initial=0.)))


class CovariateProjection:
    def __init__(self, covariates, inverse_covariates):
        self.c = np.asarray(covariates, float)
        self.wc = np.asarray(inverse_covariates, float)
        if (self.c.ndim != 2 or self.wc.shape != self.c.shape
                or not np.isfinite(self.c).all() or not np.isfinite(self.wc).all()):
            raise ValueError('Aligned finite covariates and V^-1 C required')
        a = self.c.T @ self.wc
        if np.linalg.norm(a-a.T) > 1e-9*max(float(np.linalg.norm(a)), 1e-30):
            raise ArithmeticError('Covariate information is not symmetric')
        if self.c.shape[1] and np.linalg.matrix_rank(self.c) != self.c.shape[1]:
            raise ValueError('Covariate information must have full rank')
        self.information = (a+a.T)*.5
        try:
            self.factor = cho_factor(self.information, lower=True) if a.size else None
        except np.linalg.LinAlgError as exc:
            raise ValueError('Covariate information must have full rank') from exc

    def solve(self, score):
        score = np.asarray(score, float)
        return cho_solve(self.factor, score) if self.factor is not None else score.copy()

    def coefficients(self, inverse_rhs):
        return self.solve(self.c.T @ np.asarray(inverse_rhs, float))

    def project(self, inverse_rhs):
        inverse_rhs = np.asarray(inverse_rhs, float)
        return inverse_rhs-self.wc @ self.coefficients(inverse_rhs)

    def diagonal_correction(self, marker_inverse_covariates):
        """diag(X' W C A^-1 C' W X), without a marker-by-marker product."""
        h = np.asarray(marker_inverse_covariates, float)
        if h.ndim != 2 or h.shape[1] != self.c.shape[1] or not np.isfinite(h).all():
            raise ValueError('Finite SNP-by-covariate scores required')
        if self.factor is None:
            return np.zeros(len(h))
        whitened = solve_triangular(self.factor[0], h.T, lower=True)
        return np.sum(whitened*whitened, axis=0)


def profiled_sparse_statistics(original, projection):
    """Convert the original fixed-gamma sufficient statistics by a Schur complement.

    original['dual'] = W(y-C gamma_old-X_S alpha_old). Gamma need not be
    an exact GLS estimate: its entire contribution is removed algebraically.
    """
    wx = np.asarray(original['vinv_x'], float)
    dual = np.asarray(original['dual'], float)
    old = np.asarray(original['old'], float)
    if wx.shape != (len(dual), len(old)) or len(dual) != len(projection.c):
        raise ValueError('Aligned original sparse sufficient statistics required')
    h = projection.c.T @ wx
    wy_centered = dual+wx @ old
    cov_score = projection.c.T @ wy_centered
    correction = h.T @ projection.solve(h)
    gram = np.asarray(original['gram'], float)-correction
    gram = (gram+gram.T)*.5
    score = np.asarray(original['score'], float)-h.T @ projection.solve(cov_score)
    return dict(gram=gram, score=score,
                response_ss=np.asarray(float(original['response_ss'])-cov_score @ projection.solve(cov_score)),
                projected_old_dual=projection.project(dual), projected_x=projection.project(wx),
                information_correction=correction, covariate_cross=h)


def profiled_gaussian(device, response, covariates, variance, noise, *, projection=None):
    """GLS nuisance mean and posterior Gaussian background under one fixed D.

    The mean is D X' P response; the nuisance coefficient is A^-1 C' W response.
    Passing an existing projection reuses the accurately solved C columns.
    """
    response = np.asarray(response, float)
    covariates = np.asarray(covariates, float)
    variance = np.asarray(variance, float)
    if response.ndim != 1 or covariates.ndim != 2 or len(response) != len(covariates):
        raise ValueError('Aligned vector response and covariate matrix required')
    if projection is None:
        solution, diagnostic = device.solve_precise(np.column_stack((response, covariates)), variance, noise)
        projection = CovariateProjection(covariates, solution[:, 1:])
        inverse_response = solution[:, 0]
    else:
        np.testing.assert_array_equal(projection.c, covariates)
        solution, diagnostic = device.solve_precise(response[:, None], variance, noise)
        inverse_response = solution[:, 0]
    if (not diagnostic.get('original_system_verified', False)
            or diagnostic['maximum_relative_residual'] > 1e-10):
        raise ArithmeticError('Profiled covariance solve did not qualify')
    gamma = projection.coefficients(inverse_response)
    dual = inverse_response-projection.wc @ gamma
    scores = device.scores(dual)[:, 0]
    effect = variance*scores
    orthogonality = float(np.max(abs(covariates.T @ dual), initial=0.))
    return dict(gamma=gamma, dual=dual, effect=effect, score=scores,
                projection=projection, solver=diagnostic, covariate_residual_score=orthogonality)


def project_inverse_quadratics(statistics, correction):
    """The same exact rank-k correction applies to every existing probe block."""
    correction = np.asarray(correction, float)
    result = {k: v.copy() for k, v in statistics.items()}
    if correction.shape != np.shape(result['h']) or not np.isfinite(correction).all() or np.any(correction < 0):
        raise ValueError('Finite nonnegative inverse-quadratic correction required')
    result['h'] -= correction
    result['h_blocks'] -= correction[None, :]
    return result


def replace_exact_profiled_quadratics(statistics, indices, device, x, projection):
    """Repair/refine selected P quadratics using precise V solves and projection."""
    indices = np.asarray(indices, int)
    for lo in range(0, len(indices), 32):
        ids = indices[lo:lo+32]
        rhs = np.asarray(x[ids], float).T
        solved, diagnostic = device.solve_precise(rhs, statistics['prior_variance'], float(statistics['theta'][-1]))
        if (not diagnostic.get('original_system_verified', False)
                or diagnostic['maximum_relative_residual'] > 1e-10):
            raise ArithmeticError('Exact projected inverse quadratics did not qualify')
        h = np.sum(rhs*projection.project(solved), axis=0)
        if np.any(h < 0) or np.any(h*statistics['prior_variance'][ids] >= 1):
            raise ArithmeticError('Exact projected Gaussian cavity precision is invalid')
        statistics['h_blocks'][:, ids] = h
        statistics['h'][ids] = h
    statistics['exact_indices'] = np.union1d(statistics.get('exact_indices', np.empty(0, int)), indices)


def fit_profiled_candidates(device, response, covariates, reference, models, noise, order, gates, *,
                            validation_products=None, validation_covariates=None,
                            validation_base=None, validation_y=None, rule=None, solve=None):
    """Use the original allocation/rho rule, interpolating BOTH eta and gamma.

    Each candidate D has its own GLS nuisance estimate. The calibrated predictor
    is a convex combination of complete predictors, not a BLUP under a claimed
    interpolated covariance. Half-probe gates include nuisance predictions.
    """
    training = validation_products is not None
    if training == (rule is not None) or len(models) not in (1, 3):
        raise ValueError('Validation inputs or a frozen rule, and one/three inputs required')
    if not training and any(v is not None for v in (validation_covariates, validation_base, validation_y)):
        raise ValueError('Refit must not receive validation data')
    if (not order or len(set(order)) != len(order) or not np.isfinite(noise) or noise <= 0):
        raise ValueError('Positive noise and an ordered candidate set required')
    if rule is not None and (rule['allocation'] not in order or rule.get('shape', 0.) != 0.
                            or not 0 <= rule['background_rho'] <= 1):
        raise ValueError('Frozen Gaussian allocation and interpolation rule required')
    solve = solve or (lambda d: profiled_gaussian(device, response, covariates, d, noise))
    eta0 = np.asarray(reference['effect'], float); gamma0 = np.asarray(reference['gamma'], float)
    c = np.asarray(covariates, float)
    if training:
        yv = np.asarray(validation_y, float); base = np.asarray(validation_base, float)
        cv = np.asarray(validation_covariates, float)
        if (yv.ndim != 1 or base.shape != yv.shape or cv.shape != (len(yv), len(gamma0))
                or not all(np.isfinite(a).all() for a in (yv, base, cv))):
            raise ValueError('Aligned finite validation response, covariates and base required')
    parts = len(models); parents = {}; rows = []; accepted = True
    uncertainty = np.zeros_like(eta0)
    for key in (order if training else [rule['allocation']]):
        variance = [np.asarray(m[key], float) for m in models]
        if any(d.shape != eta0.shape or not np.isfinite(d).all() or np.any(d < 0) for d in variance):
            raise ValueError('Aligned finite nonnegative candidate variances required')
        fits = [solve(d) for d in variance]
        effects = np.column_stack([f['effect'] for f in fits])
        gammas = np.column_stack([f['gamma'] for f in fits])
        parents[key] = (effects, gammas)
        prediction = device.predict(effects)+c @ gammas
        half = float(np.sqrt(np.mean((prediction[:, 1]-prediction[:, 2])**2))) if parts == 3 else 0.
        accepted &= bool(np.isfinite(prediction).all() and np.isfinite(effects).all()
                         and np.isfinite(gammas).all() and half <= gates['training_half_rms_gate'])
        if parts == 3:
            uncertainty += (effects[:, 1]-effects[:, 2])**2
        row = dict(allocation=key, training_half_rms=half,
                   maximum_covariate_residual_score=max(f['covariate_residual_score'] for f in fits))
        if training:
            change = validation_products(effects-eta0[:, None])+cv @ (gammas-gamma0[:, None])
            norm = np.sum(change**2, axis=0)
            rho = np.clip(np.divide((yv-base) @ change, norm, out=np.zeros(parts), where=norm > 1e-24), 0., 1.)
            predictions = base[:, None]+rho*change
            mse = np.mean((yv[:, None]-predictions)**2, axis=0)
            vh = float(np.sqrt(np.mean((predictions[:, 1]-predictions[:, 2])**2))) if parts == 3 else 0.
            md = float(abs(mse[1]-mse[2])) if parts == 3 else 0.
            accepted &= bool(np.isfinite(predictions).all() and np.isfinite(mse).all()
                             and vh <= gates['validation_half_rms_gate'] and md <= gates['validation_half_mse_gate'])
            row.update(rho=rho.tolist(), mse=mse.tolist(), validation_half_rms=vh, validation_half_mse=md)
        rows.append(row)
    winner = min(rows, key=lambda row: row['mse'][0]) if training else rows[0]
    frozen = (dict(allocation=winner['allocation'], shape=0., background_rho=winner['rho'][0])
              if training else dict(rule))
    eta, gamma = parents[frozen['allocation']]
    rho = frozen['background_rho']
    return dict(effect=eta0+rho*(eta[:, 0]-eta0), gamma=gamma0+rho*(gamma[:, 0]-gamma0),
                rule=frozen, numeric_accepted=bool(accepted)), dict(
        parents=rows, uncertainty=uncertainty, covariates_profiled_per_covariance=True,
        nuisance_interpolated_with_background=True)
