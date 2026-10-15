"""Double-loop EC fallback for the unchanged selected-support SS model.

Opper and Winther (JMLR 6, 2005) give the double-loop moment-matching
construction. Here the outer loop also alternates the existing constrained
EM update. No prior or likelihood is changed. Acceptance uses the original
VAMP kernels and tolerances, rather than an optimizer's success flag.
"""
from dataclasses import replace
import time
import warnings

import jax.numpy as jnp
import numpy as np
from scipy.linalg import LinAlgWarning, solve

from .grouped_ss_vamp import constrained_ss_m_step, spike_slab_denoise


def _relative(a, b):
    return float(np.sqrt(np.sum((a-b)**2)/max(np.sum(a*a), np.sum(b*b), 1e-30)))


class _SupportEC:
    """Small CPU problem in units beta=sqrt(d)*z, so pi*slab_z=1."""

    def __init__(self, eigenvalues, eigenvectors, b, total_snps, pi_min):
        self.eig = np.asarray(eigenvalues, float)
        self.vec = np.asarray(eigenvectors, float)
        self.b = np.asarray(b, float)
        self.s = len(self.b)
        self.total_snps, self.pi_min = total_snps, pi_min

    @staticmethod
    def prior(h, gamma, pi):
        precision = pi+gamma
        if precision <= 0 or not 0 < pi <= 1:
            raise ArithmeticError('Invalid EC prior tilt')
        mu, v = h/precision, 1/precision
        log_spike = np.log1p(-pi) if pi < 1 else -np.inf
        log_slab = np.log(pi)+.5*(np.log(pi)-np.log(precision))+.5*h*mu
        logs = np.logaddexp(log_spike, log_slab)
        pip = np.exp(log_slab-logs)
        return dict(mean=pip*mu, variance=pip*v+pip*(1-pip)*mu*mu,
            second=pip*(v+mu*mu), pip=pip, logz=float(logs.sum()),
            cov_x_x2=pip*mu*((1-pip)*mu*mu+(3-pip)*v),
            var_x2=pip*((1-pip)*mu**4+(6-2*pip)*mu*mu*v+(3-pip)*v*v))

    def linear(self, h, gamma, covariance=False):
        denominator = self.eig+gamma
        if np.min(denominator) <= 0:
            raise ArithmeticError('Non-positive EC linear precision')
        projected = self.vec.T@(self.b+h)
        mean = self.vec@(projected/denominator)
        result = dict(mean=mean, variance=float(np.mean(1/denominator)),
            second=float(mean@mean+np.sum(1/denominator)),
            logz=float(.5*(self.s*np.log(2*np.pi)-np.log(denominator).sum()
                          +projected@(projected/denominator))))
        if covariance:
            result['covariance'] = (self.vec/denominator)@self.vec.T
        return result

    def m_step(self, prior):
        # The same constrained M step as constrained_ss_m_step, with d=1.
        a, k = float(prior['pip'].sum()), float(prior['second'].sum())/2
        c = self.total_snps+.5*a+k
        return float(np.clip(3*a/(c+np.sqrt(max(c*c-6*a*k, 0.))), self.pi_min, 1.))

    def residuals(self, h, gamma, pi):
        if gamma <= 0:
            raise ArithmeticError('Non-positive VAMP message precision')
        p = self.prior(h, gamma, pi)
        eta = 1/float(np.mean(p['variance']))
        gamma2, h2 = eta-gamma, eta*p['mean']-h
        q = self.linear(h2, gamma2)
        denominator = self.eig+gamma2
        # Same cancellation-resistant algebra as SupportPrecisionSolver.
        proposed_gamma = float(np.mean(self.eig/denominator)/q['variance'])
        if proposed_gamma <= 0:
            raise ArithmeticError('Non-positive outgoing VAMP precision')
        proposed_h = self.vec@((self.vec.T@self.b/q['variance']
                       +(proposed_gamma-self.eig)*(self.vec.T@h2))/denominator)
        return dict(message_error=_relative(proposed_h/proposed_gamma, h/gamma),
            module_error=_relative(p['mean'], q['mean']),
            precision_error=abs(proposed_gamma/gamma-1),
            em_error=abs(self.m_step(p)/pi-1))

    def inner_terms(self, u, h0, eta, pi, *, curvature=True):
        scale = np.sqrt(eta)
        h, gamma = u[:-1]*scale, u[-1]*eta
        p = self.prior(h, gamma, pi)
        q = self.linear(h0-h, eta-gamma, covariance=curvature)
        gradient = np.r_[(p['mean']-q['mean'])*scale,
                         .5*(q['second']-p['second'].sum())*eta]
        if not curvature:
            return p['logz']+q['logz'], gradient, None, p, q
        cov = q['covariance']
        hessian = np.empty((self.s+1, self.s+1))
        hessian[:-1, :-1] = (np.diag(p['variance'])+cov)*eta
        hessian[:-1, -1] = (-.5*p['cov_x_x2']-cov@q['mean'])*eta*scale
        hessian[-1, :-1] = hessian[:-1, -1]
        hessian[-1, -1] = (.25*p['var_x2'].sum()+.5*np.sum(cov**2)
                           +q['mean']@cov@q['mean'])*eta**2
        return p['logz']+q['logz'], gradient, hessian, p, q

    def inner_solve(self, h0, eta, pi, h, gamma, tolerance):
        # Exact integrability domain, not extra restrictions on the model.
        lower, upper = -pi/eta, 1+self.eig[0]/eta
        fraction = gamma/eta
        if not lower < fraction < upper:
            fraction = (lower+upper)*.5
        u = np.r_[h/np.sqrt(eta), fraction]
        for iteration in range(100):
            value, gradient, hessian, p, q = self.inner_terms(u, h0, eta, pi)
            mismatch = max(_relative(p['mean'], q['mean']),
                abs(q['second']-p['second'].sum())/
                max(q['second'], float(p['second'].sum()), 1e-300))
            if mismatch < tolerance or np.linalg.norm(gradient, np.inf) < min(tolerance, 1e-12):
                return u[:-1]*np.sqrt(eta), u[-1]*eta, p, q, iteration
            with warnings.catch_warnings():
                warnings.simplefilter('error', LinAlgWarning)
                try:
                    direction = solve(hessian, -gradient, assume_a='pos')
                except (np.linalg.LinAlgError, LinAlgWarning):
                    direction = np.linalg.lstsq(hessian, -gradient, rcond=None)[0]
            derivative = float(gradient@direction)
            if not np.isfinite(derivative) or derivative >= 0:
                raise ArithmeticError('EC inner Newton direction is not descending')
            step = 1.
            if direction[-1] > 0:
                step = min(step, .99*(upper-u[-1])/direction[-1])
            elif direction[-1] < 0:
                step = min(step, .99*(lower-u[-1])/direction[-1])
            while step >= np.finfo(float).eps:
                trial = u+step*direction
                try:
                    # Backtracking needs only value/gradient: constructing the
                    # dense covariance and Hessian here would add O(s^3) work
                    # for every rejected trial, without affecting its decision.
                    new, new_gradient, _, _, _ = self.inner_terms(
                        trial, h0, eta, pi, curvature=False)
                    rounding = 16*np.finfo(float).eps*max(abs(value), 1.)
                    if (new <= value+1e-4*step*derivative or
                            (new <= value+rounding and
                             np.linalg.norm(new_gradient) < np.linalg.norm(gradient))):
                        break
                except ArithmeticError:
                    pass
                step *= .5
            else:
                raise ArithmeticError('EC inner line search stalled')
            u = trial
        raise ArithmeticError('EC inner iteration limit')


def _verify_native(solver, h, gamma, pi, d, total_snps, pi_change, config):
    """Original undamped equations on an EC iterate, in original effect units."""
    r = jnp.asarray((h/gamma*np.sqrt(d))[:, None])
    precision = jnp.asarray([gamma/d])
    mean, variance, pip, mu, v, _ = spike_slab_denoise(r, precision[None, :], pi, d/pi)
    eta, guarded = solver.stabilize_eta(1/jnp.mean(variance, axis=0), precision)
    m2, gp, rp = solver.natural_module(eta[None, :]*mean-precision[None, :]*r, eta-precision)
    next_pi, _ = constrained_ss_m_step(jnp.sum(pip), jnp.sum(pip*(v+mu*mu)),
                                     total_snps, d, config.pi_min)
    row = dict(message_error=_relative(np.asarray(rp), np.asarray(r)),
        module_error=_relative(np.asarray(m2), np.asarray(mean)),
        precision_error=float(np.asarray(jnp.abs(gp-precision)/precision)[0]),
        em_error=abs(float(next_pi)-pi)/max(pi, config.pi_min),
        pi_change=pi_change, guarded=bool(np.asarray(guarded)[0]))
    good = (all(np.isfinite(row[k]) for k in row if k != 'guarded') and
            not row['guarded'] and
            max(row[k] for k in ('message_error', 'module_error', 'precision_error')) < config.tolerance)
    if config.learn_pi:
        good &= max(row['em_error'], pi_change) < config.em_tolerance
    row['passed'] = bool(good)
    return row, (np.asarray(mean), np.asarray(variance), np.asarray(pip))


def solve_support_ec(posterior, solver, products, residual, *, prior_second_moment,
                     total_snps, config, callback=None):
    """Fallback after unresolved damped VAMP iterations; leave successes untouched.

    Restart from VAMP's recorded initial channel. EC's additional Gaussian
    cavity uses the prior second moment; no shared parameter is reinitialized.
    Five (or config.consecutive_converged) successive EC iterates must pass
    the native equations. Replaying the old iteration is not a certificate:
    its fixed point can be unstable even when the equations are satisfied.
    """
    if posterior.status == ['converged']:
        return posterior
    s, d = solver.p, float(prior_second_moment)
    if (s < 1 or posterior.mean.shape != (s, 1) or posterior.pi.shape != (1, 1)
            or not np.isfinite(d) or d <= 0 or total_snps < s):
        raise ValueError('EC fallback requires one positive-variance support model')
    initial = posterior.initialization or {}
    if initial.get('method') != 'zero':
        raise ValueError('Support EC requires the recorded zero-message VAMP initialization')
    pi = float(np.asarray(initial['initial_pi'])[0, 0])
    slab = float(np.asarray(initial['initial_slab_variance'])[0, 0])
    precision = float(np.asarray(initial['initial_message_precision'])[0])
    if (not np.all(np.isfinite([pi, slab, precision]))
            or not config.pi_min <= pi <= 1 or slab <= 0 or precision <= 0
            or not np.isclose(pi*slab, d, rtol=1e-12, atol=0.)):
        raise ValueError('Invalid recorded VAMP initialization for support EC')
    started = time.monotonic()
    report = dict(method='double_loop_support_EC', status='max_iterations',
        prior_and_likelihood_unchanged=True, max_outer_iterations=config.max_iterations,
        max_inner_iterations=100, initialization='same_as_vamp',
        initial_pi=pi, initial_slab_variance=slab, initial_message_precision=precision,
        verification='native_undamped_equations_on_consecutive_EC_iterates')
    rows, checks = [], []
    iteration = inner_iterations = em_updates = 0
    previous_energy = None
    max_energy_increase = 0.
    try:
        noise = float(np.asarray(solver.noise)[0])
        xy = np.asarray(products.transpose(jnp.asarray(residual)[:, None]))[:, 0]
        model = _SupportEC(np.asarray(solver.values)*d/noise, solver.vectors,
                           xy*np.sqrt(d)/noise, total_snps, config.pi_min)
        # beta=sqrt(d)*z: copy the original channel, not the failed iterate.
        # EC additionally needs a likelihood-side Gaussian cavity. Initialize
        # it at zero with variance d (precision 1 in z units). Then both
        # cavities integrate without clipping the shared VAMP precision.
        mean, variance, *_ = spike_slab_denoise(
            jnp.zeros((s, 1)), jnp.asarray([[precision]]), pi, slab)
        average_variance = float(jnp.mean(variance))
        h, gamma = np.zeros(s), d*precision
        eta = 1.+gamma
        h0 = eta*np.asarray(mean[:, 0])/np.sqrt(d)
        report.update(initial_mean=0., initial_nonlinear_variance=average_variance,
                      initial_linear_precision=1/d, initial_common_precision=eta/d)
        previous_pi = pi
        tolerance = .01*min(config.tolerance, config.em_tolerance) if config.learn_pi else .01*config.tolerance
        for iteration in range(1, config.max_iterations+1):
            h, gamma, p, q, inner_count = model.inner_solve(
                h0, eta, pi, h, gamma, min(1e-9, .01*tolerance))
            inner_iterations += inner_count
            outside = total_snps-s
            logz0 = .5*(s*np.log(2*np.pi)-s*np.log(eta)+h0@h0/eta)
            outside_evidence = outside*np.log1p(-pi) if outside else 0.
            energy = float(-p['logz']-q['logz']+logz0-outside_evidence)
            if not np.isfinite(energy):
                raise ArithmeticError('Nonfinite EC energy')
            if previous_energy is not None:
                max_energy_increase = max(max_energy_increase, energy-previous_energy)
            previous_energy = energy
            pi_change = abs(pi-previous_pi)/max(previous_pi, config.pi_min)
            slab_change = abs(previous_pi/pi-1)
            previous_pi = pi
            try:
                errors = model.residuals(h, gamma, pi)
                infer_ok = all(np.isfinite(v) for v in errors.values()) and max(
                    errors[k] for k in ('message_error', 'module_error', 'precision_error')) < tolerance
            except ArithmeticError:
                # Inner EC cavities need not yet define a valid VAMP step.
                errors = dict.fromkeys(('message_error', 'module_error', 'precision_error', 'em_error'))
                infer_ok = False
            eligible = iteration >= config.min_iterations and (not config.learn_pi or iteration > config.em_warmup)
            resolved = infer_ok and (not config.learn_pi or errors['em_error'] < tolerance)
            if eligible and resolved:
                check, arrays = _verify_native(solver, h, gamma, pi, d, total_snps, pi_change, config)
                checks = checks+[dict(check, ec_iteration=iteration)] if check['passed'] else []
            else:
                checks = []
            done = len(checks) >= config.consecutive_converged
            if iteration == 1 or iteration % config.log_every == 0 or done:
                row = dict(iteration=posterior.iterations+iteration, ec_iteration=iteration,
                    solver_stage='double_loop_support_EC', seconds=posterior.seconds+time.monotonic()-started,
                    pi=[[pi]], slab_variance=[[d/pi]], gamma1=[float(gamma/d)],
                    alpha1=[float(gamma*np.mean(p['variance']))],
                    **{key: [errors[key]] for key in errors}, pi_change=[pi_change],
                    slab_change=[slab_change], converged=[done], ec_energy=energy, em_updates=em_updates,
                    prior_variance_components=[[pi*(d/pi)*solver.trace_normalizer]],
                    prior_constraint_max_relative_error=float(abs(pi*(d/pi)-d)/d),
                    precision_safeguard_count=solver.precision_safeguard_count,
                    linear_solver=solver.last_diagnostics)
                if checks:
                    row.update({key: [checks[-1][key]] for key in errors})
                    row['precision_guarded'] = [checks[-1]['guarded']]
                rows.append(row)
                if callback:
                    callback(row)
            if done:
                report['status'] = 'converged'
                break
            # EC outer moment update, then the unchanged constrained M step
            # only when inference at the current pi has resolved.
            eta, h0 = 1/q['variance'], q['mean']/q['variance']
            if (config.learn_pi and iteration > config.em_warmup
                    and infer_ok and errors['em_error'] >= tolerance):
                pi = model.m_step(p)
                em_updates += 1
                previous_energy = None
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        report.update(status='failed', error=str(error))
    elapsed = time.monotonic()-started
    report.update(seconds=elapsed, outer_iterations=iteration, inner_iterations=inner_iterations,
        em_updates=em_updates, verification_checks=checks,
        verified_consecutive_iterations=len(checks),
        max_energy_increase_within_fixed_pi=max_energy_increase)
    diagnostics = dict(posterior.linear_solver or {}, **solver.diagnostics(), double_loop_ec=report)
    result = replace(posterior, seconds=posterior.seconds+elapsed,
        iterations=posterior.iterations+iteration, trace=posterior.trace+rows, linear_solver=diagnostics)
    if report['status'] == 'converged':
        result = replace(result, mean=arrays[0], variance=arrays[1], pip=arrays[2],
            pi=np.array([[pi]]), slab_variance=np.array([[d/pi]]), precision=np.array([gamma/d]),
            status=['converged'], initialization=dict(posterior.initialization or {},
                ec_fallback=dict(method='same_as_vamp', initial_pi=report['initial_pi'],
                    initial_slab_variance=slab, initial_message_precision=precision,
                    initial_nonlinear_variance=average_variance,
                    initial_linear_precision=1/d, prior_second_moment=d)))
    return result
