"""Fixed-budget square ridge with two validation stopping/calibration orders.

All genotype products and covariance solves are supplied by a backend. Alpha,
gamma, original group budgets and noise remain fixed throughout the path.
"""
from __future__ import annotations

import numpy as np
from .blup_effect_weighting import effect_power_variance
def profiled_validation(y, base, direction, direction_derivative=None):
    """Profile rho in [0,1] for base + rho * direction by validation MSE.

    With z=direction and z'=direction_derivative, the envelope derivative is
    -2 rho (y-base-rho*z)'z'/n, including regions where rho is 0 or 1. At the
    exact zero direction we return rho=0 and gradient=0 by convention. At
    kappa=0 this is NOT generally the right derivative of the profiled path:
    a search must evaluate positive kappa values rather than start there on
    the strength of that convention. If no derivative is supplied the
    gradient is None, except for the explicitly degenerate zero direction.
    """
    y, base, direction = (np.asarray(a, dtype=float) for a in (y, base, direction))
    arrays = (y, base, direction)
    if (y.ndim != 1 or not len(y)
            or any(a.shape != y.shape or not np.isfinite(a).all() for a in arrays)):
        raise ValueError('Aligned nonempty finite validation vectors required')
    if direction_derivative is not None:
        direction_derivative = np.asarray(direction_derivative, dtype=float)
        if (direction_derivative.shape != y.shape
                or not np.isfinite(direction_derivative).all()):
            raise ValueError('Direction derivative must be finite and aligned')
    residual = y.astype(np.longdouble) - base.astype(np.longdouble)
    direction_scale = float(np.max(np.abs(direction)))
    if direction_scale == 0:
        rho = 0.
    else:
        scaled = direction.astype(np.longdouble) / np.longdouble(direction_scale)
        numerator = np.sum(residual * scaled)
        denominator = np.longdouble(direction_scale) * np.sum(scaled * scaled)
        rho = float(np.clip(numerator / denominator, 0, 1))
    error = residual - np.longdouble(rho) * direction.astype(np.longdouble)
    mse = np.mean(error * error)
    if not np.isfinite(mse) or mse > np.finfo(float).max:
        raise ArithmeticError('Validation MSE exceeds floating-point range')
    if direction_scale == 0 or rho == 0:
        gradient = 0.
    elif direction_derivative is None:
        gradient = None
    else:
        gradient = -2 * np.longdouble(rho) * np.mean(
            error * direction_derivative.astype(np.longdouble))
        if not np.isfinite(gradient) or abs(gradient) > np.finfo(float).max:
            raise ArithmeticError('Validation derivative exceeds floating-point range')
        gradient = float(gradient)
    return dict(rho=rho, mse=float(mse), gradient=gradient)

from .profiled_prediction_refinement import CovariateProjection

POLICIES = ('select_then_calibrate', 'calibrate_then_select')


def _mse(y, prediction):
    error = y.astype(np.longdouble)-prediction.astype(np.longdouble)
    value = float(np.mean(error*error))
    if not np.isfinite(value):
        raise ArithmeticError('Nonfinite validation loss')
    return value


class ValidationStopper:
    """Stop on the first non-improvement; keep the earlier minimum on ties."""
    def __init__(self, y, base, policy, maximum_iterations=10, tolerance=1e-12):
        self.y, self.base = np.asarray(y, float), np.asarray(base, float)
        if (policy not in POLICIES or self.y.ndim != 1 or not len(self.y)
                or self.base.shape != self.y.shape
                or not np.isfinite(self.y).all() or not np.isfinite(self.base).all()
                or isinstance(maximum_iterations, (bool, np.bool_))
                or not isinstance(maximum_iterations, (int, np.integer)) or maximum_iterations < 1
                or not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError('Finite validation vectors and valid stopping controls required')
        self.policy = policy; self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance); self.best = None; self.history = []; self.active = True
        self.stop_reason = None

    def observe(self, iteration, direction):
        if not self.active or iteration != len(self.history)+1:
            raise ValueError('Observe consecutive iterations only while the rule is active')
        direction = np.asarray(direction, float)
        if direction.shape != self.y.shape or not np.isfinite(direction).all():
            raise ValueError('Finite validation direction required')
        raw = _mse(self.y, self.base+direction)
        rho = (profiled_validation(self.y, self.base, direction)['rho']
               if self.policy == POLICIES[1] else None)
        score = raw if rho is None else _mse(self.y, self.base+rho*direction)
        improved = self.best is None or score < self.best['selection_mse']-self.tolerance
        row = dict(iteration=int(iteration), raw_mse=raw, selection_mse=score,
                   rho_at_selection=rho, improved=bool(improved))
        self.history.append(row)
        if improved:
            self.best = dict(row, direction=direction.copy())
        else:
            self.active = False; self.stop_reason = 'first_non_improvement'
        if iteration >= self.maximum_iterations and self.active:
            self.active = False; self.stop_reason = 'maximum_iterations'
        return bool(improved)

    def rule(self):
        if self.active or self.best is None:
            raise ValueError('A completed validation path is required')
        rho = self.best['rho_at_selection']
        if rho is None:
            rho = profiled_validation(self.y, self.base, self.best['direction'])['rho']
        return dict(policy=self.policy, iteration=self.best['iteration'], rho=float(rho),
                    selection_mse=self.best['selection_mse'], raw_validation_mse=self.best['raw_mse'],
                    calibrated_validation_mse=_mse(self.y, self.base+rho*self.best['direction']),
                    evaluated_iterations=len(self.history), stop_reason=self.stop_reason,
                    maximum_iterations=self.maximum_iterations, improvement_tolerance=self.tolerance)


def validate_rules(rules, maximum_iterations):
    if not isinstance(rules, dict) or set(rules) != set(POLICIES):
        raise ValueError('Both frozen policy rules are required')
    for policy, rule in rules.items():
        k = rule.get('iteration'); rho = rule.get('rho')
        if (rule.get('policy') != policy or isinstance(k, (bool, np.bool_))
                or not isinstance(k, (int, np.integer)) or not 1 <= k <= maximum_iterations
                or not np.isscalar(rho) or not np.isfinite(rho) or not 0 <= rho <= 1
                or rule.get('maximum_iterations') != maximum_iterations):
            raise ValueError('Invalid transferred iteration/calibration rule')


def fit_iterative_background(gaussian, variance, norm_squared, groups, solve, *,
                             initial_dual=None, validation_y=None, validation_base=None,
                             validation_products=None, rules=None, maximum_iterations=10,
                             improvement_tolerance=1e-12, progress=None):
    """Share one uncalibrated path until both rules stop, or refit frozen counts.

    solve(d, initial_dual) -> (dual, effect, certified_solver_record).
    validation_products(effect_difference) -> validation prediction difference.
    Only selected effects, the current effect and small validation vectors are
    retained. Final refitting accepts no validation inputs.
    """
    eta0 = np.asarray(gaussian, float); c = np.asarray(variance, float)
    norms = np.asarray(norm_squared, float); groups = np.asarray(groups)
    # Reuse the established weight validation before calling an expensive backend.
    effect_power_variance(eta0, c, norms, groups, 2.)
    if (isinstance(maximum_iterations, (bool, np.bool_))
            or not isinstance(maximum_iterations, (int, np.integer)) or maximum_iterations < 1):
        raise ValueError('Positive integer iteration limit required')
    training = rules is None
    if training:
        if validation_products is None or validation_y is None or validation_base is None:
            raise ValueError('Validation inputs are required for training')
        stoppers = {p: ValidationStopper(validation_y, validation_base, p,
                    maximum_iterations, improvement_tolerance) for p in POLICIES}
    else:
        if any(v is not None for v in (validation_y, validation_base, validation_products)):
            raise ValueError('Final refitting must not receive validation data')
        validate_rules(rules, maximum_iterations)
        stop_at = max(r['iteration'] for r in rules.values())
    pilot = eta0.copy(); dual = initial_dual; selected = {}; history = []; directions = []
    for k in range(1, maximum_iterations+1):
        d, allocation = effect_power_variance(pilot, c, norms, groups, 2.)
        dual, effect, diagnostic = solve(d, dual)
        effect = np.asarray(effect, float)
        if (effect.shape != eta0.shape or not np.isfinite(effect).all()
                or np.any(effect[d == 0] != 0)
                or not diagnostic.get('original_system_verified', False)
                or diagnostic.get('maximum_relative_residual', np.inf) > 1e-10):
            raise ArithmeticError('Uncertified or invalid background endpoint')
        if training:
            direction = np.asarray(validation_products(effect-eta0), float).reshape(-1)
            directions.append(direction)
            for p, stopper in stoppers.items():
                if stopper.active and stopper.observe(k, direction):
                    selected[p] = effect.copy()
        else:
            for p, rule in rules.items():
                if k == rule['iteration']: selected[p] = effect.copy()
        row = dict(iteration=k, allocation=allocation, solver=diagnostic,
                   effect_relative_change=float(np.linalg.norm(effect-pilot)/max(np.linalg.norm(pilot),1e-300)),
                   numerically_zeroed_nonzero_pilot=int(np.count_nonzero((pilot!=0)&(c>0)&(norms>0)&(d==0))))
        history.append(row)
        if progress is not None: progress(k, dual, effect, d, row)
        pilot = effect
        if training and not any(s.active for s in stoppers.values()): break
        if not training and k >= stop_at: break
    if training:
        rules = {p:s.rule() for p,s in stoppers.items()}
    outputs = {p: eta0+rules[p]['rho']*(selected[p]-eta0) for p in POLICIES}
    result = dict(effects=outputs, rules=rules, history=history,
                  fitted_iterations=len(history), alpha_gamma_fixed=True)
    if training:
        result['validation_directions'] = np.column_stack(directions)
        result['stopping_history'] = {p:s.history for p,s in stoppers.items()}
    return result


def build_profiled_information(x, indices, old, y, covariates, variance, noise,
                               backend, *, block_size=64, progress=None):
    """Build J=X_S' P X_S, b=X_S' P y directly, without retaining V^-1 X_S.

    One solve for [y,C], then blocked support solves. Save only the small support
    information, covariate solves and phenotype solve. No background h_j probes.
    """
    indices = np.asarray(indices); old = np.asarray(old, float)
    y, covariates = np.asarray(y, float), np.asarray(covariates, float)
    c = np.asarray(variance, float)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices) or np.any(indices<0) or np.any(indices>=len(x))
            or old.shape != indices.shape or y.shape != (x.shape[1],)
            or covariates.ndim != 2 or len(covariates) != len(y)
            or c.shape != (len(x),) or not np.isfinite(c).all() or np.any(c<0)
            or not np.isfinite(noise) or noise<=0 or not isinstance(block_size,int) or block_size<1
            or not all(np.isfinite(a).all() for a in (old,y,covariates))):
        raise ValueError('Valid aligned sparse information inputs required')
    def checked(rhs):
        value, record = backend.solve_precise(rhs, c, noise)
        if (not record.get('original_system_verified', False)
                or record.get('maximum_relative_residual', np.inf)>1e-10):
            raise ArithmeticError('Uncertified support information solve')
        return np.asarray(value,float), record
    solved, record = checked(np.column_stack((y,covariates)))
    projection = CovariateProjection(covariates, solved[:,1:])
    xs = np.asarray(x[indices],float).T
    score = xs.T @ projection.project(solved[:,0]); gram = np.empty((len(indices),len(indices)))
    records = [record]
    for lo in range(0,len(indices),block_size):
        hi = min(lo+block_size,len(indices)); inverse, record = checked(xs[:,lo:hi])
        gram[:,lo:hi] = xs.T @ projection.project(inverse); records.append(record)
        if progress is not None: progress(dict(columns=hi,total=len(indices),solver=record))
    scale = max(float(np.linalg.norm(gram)),np.finfo(float).tiny)
    asymmetry = float(np.linalg.norm(gram-gram.T)/scale)
    if asymmetry>2e-5: raise ArithmeticError('Profiled information asymmetry exceeded gate')
    data = dict(snps=indices.copy(),old=old.copy(),gram=(gram+gram.T)*.5,score=score,
                inverse_covariates=solved[:,1:],inverse_y=solved[:,0])
    return data, dict(numeric_accepted=True,covariance_solves=records,
                      information_asymmetry=asymmetry,explicit_grm=False,retains_inverse_support=False)
