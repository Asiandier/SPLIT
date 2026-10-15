"""Prediction alpha refinement on the final refit's fixed LASSO support.

The single SS prior has no hyperpriors and satisfies pi*slab*tr(X'X)/n=Q_s.
Known-zero indicators outside the support enter EM counts and scalar evidence,
while VAMP moment matching uses only the modeled support coordinates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
import jax
import numpy as np

from .grouped_ss_vamp import GenotypeProducts, VAMPConfig, VAMPResult, fit_grouped_ss_vamp
from .ss_linear import SupportPrecisionSolver


@dataclass
class AlphaVAMPResult:
    alpha: np.ndarray
    posterior: VAMPResult
    training_sparse_score: np.ndarray
    residual: np.ndarray
    summary: dict


def refine_alpha_vamp(*, design, y, covariates, gamma, background_score,
                      alpha, total_snps, q_sparse, noise_variance, config=None,
                      callback=None, background_estimator='native_conditional_blup') -> AlphaVAMPResult:
    """Use training data only; hold the conditioning background, gamma and theta fixed.

    An unbiased sparse-variance estimate can be negative. Its nonnegative
    boundary gives the all-zero alpha prior; the raw estimate is recorded.
    Supplied alpha effects are retained for diagnostics, not initialization.
    """
    started = time.monotonic()
    if background_estimator not in ('native_conditional_blup', 'constrained_ss_em_vamp'):
        raise ValueError('Unknown conditioning background estimator')
    x = np.asarray(design)
    response, pilot, background = (np.asarray(value, float)
                                    for value in (y, alpha, background_score))
    if x.ndim != 2 or x.shape[0] < 1 or not np.all(np.isfinite(x)):
        raise ValueError('Support design must be a finite sample-by-SNP matrix')
    n, s = x.shape
    if (response.shape != (n,) or background.shape != (n,) or pilot.shape != (s,)
            or any(not np.all(np.isfinite(value)) for value in (response, background, pilot))):
        raise ValueError('Invalid alpha, response or conditioning background')
    if (not isinstance(total_snps, (int, np.integer)) or total_snps < max(1, s)
            or not np.isfinite(q_sparse) or not np.isfinite(noise_variance) or noise_variance <= 0):
        raise ValueError('Invalid total SNP count, sparse variance or noise variance')
    c = np.empty((n, 0)) if covariates is None else np.asarray(covariates, float)
    gamma = np.asarray(gamma, float)
    if (c.ndim != 2 or c.shape[0] != n or gamma.shape != (c.shape[1],)
            or not np.all(np.isfinite(c)) or not np.all(np.isfinite(gamma))):
        raise ValueError('Invalid covariates or gamma')
    nuisance = c@gamma
    residual = response-nuisance-background
    target = max(float(q_sparse), 0.)
    if s == 0 and target > 0:
        raise ValueError('Positive sparse variance requires a nonempty support')
    config = config or VAMPConfig(max_iterations=10000, damping=.2, em_damping=.2,
                                  tolerance=1e-4, em_tolerance=1e-4, log_every=25)
    with jax.experimental.enable_x64():
        if target == 0:
            zero = np.zeros((s, 1))
            posterior = VAMPResult(zero, zero.copy(), zero.copy(), np.zeros((1, 1)),
                np.zeros((1, 1)), ['converged'], [], 0., 0, np.array([n/noise_variance]),
                dict(method='all_zero_sparse_prior'))
            normalizer = float(np.einsum('ij,ij->', x, x, dtype=np.float64)/n)
        else:
            products = GenotypeProducts(x.T)
            solver = SupportPrecisionSolver(products)
            normalizer = solver.trace_normalizer
            if normalizer <= 0:
                raise ValueError('Positive sparse variance requires a nonconstant support design')
            config = replace(config, initial_pi=max(s/total_snps, config.pi_min))
            posterior = fit_grouped_ss_vamp(products, residual, np.zeros(s, int),
                np.array([target]), np.array([normalizer]), np.array([noise_variance]),
                solver, config, callback, em_counts=np.array([total_snps]))
            if (posterior.status != ['converged'] and config.max_iterations >= config.min_iterations
                    and (not config.learn_pi or config.max_iterations > config.em_warmup)):
                from .ss_fixed_point import polish_support_fixed_point
                posterior = polish_support_fixed_point(posterior, solver,
                    prior_second_moment=target/normalizer, total_snps=total_snps,
                    config=config, callback=callback)
                if posterior.status != ['converged']:
                    from .ss_ec import solve_support_ec
                    posterior = solve_support_ec(posterior, solver, products, residual,
                        prior_second_moment=target/normalizer, total_snps=total_snps,
                        config=config, callback=callback)
            if posterior.status != ['converged']:
                raise RuntimeError('Alpha prior-Q VAMP did not converge; outputs rejected')
    updated = posterior.mean[:, 0]
    score = x@updated
    old_score = x@pilot
    old_full, full = nuisance+background+old_score, nuisance+background+score
    summary = dict(requested=True, status='converged', method='prior_Q_ss_em_vamp_alpha',
        mode='final_refit', support_size=s, pi_denominator=int(total_snps), hyperpriors=None,
        constraint_kind='prior_expected_variance',
        variance_constraint='pi*slab_variance*trace(X_S.T@X_S)/n=Q_s',
        Q_s_raw=float(q_sparse), Q_s=target, trace_normalizer=normalizer,
        pi=float(posterior.pi[0, 0]), slab_variance=float(posterior.slab_variance[0, 0]),
        prior_implied_variance=float(posterior.pi[0, 0]*posterior.slab_variance[0, 0]*normalizer),
        sparse_squared_term=float(np.mean(score**2)),
        old_sparse_squared_term=float(np.mean(old_score**2)),
        old_full_squared_term=float(np.mean(old_full**2)), new_full_squared_term=float(np.mean(full**2)),
        new_training_mse=float(np.mean((response-full)**2)),
        gamma_fixed=True, background_fixed=True, variance_components_fixed=True, noise_fixed=True,
        alpha_residual=('y-C gamma-X beta_native_conditional_BLUP'
            if background_estimator == 'native_conditional_blup' else 'y-C gamma-X beta_SS_previous_round'),
        background_estimator=background_estimator,
        refinement_order='alpha_prior_Q_vamp_then_background_ss',
        validation_used=False, initialization=posterior.initialization,
        config=asdict(config),
        iterations=posterior.iterations, fit_seconds=posterior.seconds,
        total_refinement_seconds=time.monotonic()-started, linear_solver=posterior.linear_solver,
        final_diagnostics=posterior.trace[-1] if posterior.trace else None)
    return AlphaVAMPResult(updated, posterior, score, residual, summary)
