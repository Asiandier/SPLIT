"""Bounded Newton-Krylov refinement of selected-support EM-VAMP equations."""
from dataclasses import replace
import time
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root

from .grouped_ss_vamp import constrained_ss_m_step, spike_slab_denoise, ss_message_from_mean


def polish_support_fixed_point(posterior, solver, *, prior_second_moment,
                              total_snps, config, callback=None):
    """Continue an unconverged single-support fit at the same moment equations.

    No prior, likelihood, support, initialization or convergence tolerance is
    replaced. The small support-space Gaussian solver and scalar SS kernels
    are reused. Root candidates must pass the original undamped diagnostics
    for consecutive ordinary EM-VAMP steps before they can be published.
    """
    started = time.monotonic()
    s = solver.p
    d = float(prior_second_moment)
    if (posterior.mean.shape != (s, 1) or posterior.pi.shape != (1, 1)
            or not np.isfinite(d) or d <= 0 or total_snps < s):
        raise ValueError('Fixed-point refinement requires one positive-variance support model')
    warm_r = np.asarray(ss_message_from_mean(posterior.mean, posterior.precision[None, :],
                                             posterior.pi, posterior.slab_variance))[:, 0]
    scale = max(float(np.linalg.norm(warm_r)/np.sqrt(s)), np.sqrt(d), 1e-12)
    fixed_pi = float(posterior.pi[0, 0])
    state = np.r_[warm_r/scale, np.log(posterior.precision[0])]
    if config.learn_pi:
        state = np.r_[state, np.log(fixed_pi)]
    evaluations = 0

    def moments(r, gamma, pi):
        slab = d/pi
        values = spike_slab_denoise(r, gamma[None, :], pi, slab)
        mean, variance, pip, mu, conditional_variance = values[:5]
        updated_pi, _ = constrained_ss_m_step(jnp.sum(pip),
            jnp.sum(pip*(conditional_variance+mu*mu)), total_snps, d, config.pi_min)
        return mean, variance, pip, updated_pi

    def evaluate(value):
        nonlocal evaluations
        evaluations += 1
        r = jnp.asarray(value[:s, None]*scale)
        gamma = jnp.asarray([np.exp(np.clip(value[s], -60., 60.))])
        pi = (float(np.exp(np.clip(value[s+1], np.log(config.pi_min), 0.)))
              if config.learn_pi else fixed_pi)
        mean, variance, pip, next_pi = moments(r, gamma, pi)
        divergence = gamma*jnp.mean(variance, axis=0)
        if not np.all(np.isfinite(np.asarray(divergence))) or np.any(np.asarray(divergence) <= 0):
            raise ArithmeticError('Invalid SS variance at a root candidate')
        eta, guarded = solver.stabilize_eta(gamma/divergence, gamma)
        h2 = eta[None, :]*mean-gamma[None, :]*r
        m2, proposed_gamma, proposed_r = solver.natural_module(h2, eta-gamma)
        relative = lambda a, b: float(np.linalg.norm(np.asarray(a-b))/max(
            np.linalg.norm(np.asarray(a)), np.linalg.norm(np.asarray(b)), 1e-30))
        diagnostics = dict(message_error=relative(proposed_r,r), module_error=relative(m2,mean),
            precision_error=float(np.asarray(jnp.abs(proposed_gamma-gamma)/gamma)[0]),
            em_error=float(abs(float(next_pi)-pi)/max(pi, config.pi_min)),
            guarded=bool(np.asarray(guarded)[0]))
        residual = np.r_[np.asarray((proposed_r-r)/scale)[:, 0],
                         np.log(float(proposed_gamma[0]))-value[s]]
        if config.learn_pi:
            residual = np.r_[residual, np.log(float(next_pi))-value[s+1]]
        return residual, (r,gamma,pi,mean,variance,pip,proposed_r,proposed_gamma,diagnostics)

    report = dict(method='Newton_Krylov_support_fixed_point', max_root_iterations=100,
                  status='failed', prior_and_likelihood_unchanged=True)
    try:
        solved = root(lambda value:evaluate(value)[0], state, method='krylov',
                      options={'fatol':.01*min(config.tolerance,config.em_tolerance), 'maxiter':100})
        report.update(root_success=bool(solved.success), root_message=str(solved.message),
                      root_iterations=int(getattr(solved,'nit',0)))
        state = solved.x
        previous_pi = float(evaluate(state)[1][2])
        for check in range(config.consecutive_converged):
            _, current = evaluate(state)
            r,gamma,pi = current[:3]
            if config.learn_pi:
                # At a conditional fixed point the SQUAREM guard selects two
                # ordinary EM maps per inner step. Verify that same fallback.
                optimum = pi
                for _ in range(config.em_inner_steps):
                    one = float(moments(r,gamma,optimum)[3])
                    if config.em_acceleration:
                        resolution = max(config.em_tolerance,config.em_acceleration_floor)
                        if abs(one-optimum) > resolution*max(optimum,config.pi_min):
                            raise ArithmeticError('Root candidate is outside the ordinary-EM fallback region')
                        one = float(moments(r,gamma,one)[3])
                    optimum = one
                pi += config.em_damping*(optimum-pi)
                state[-1] = np.log(pi)
            _, current = evaluate(state)
            r,gamma,pi,mean,variance,pip,proposed_r,proposed_gamma,diag = current
            pi_change = abs(pi-previous_pi)/max(previous_pi,config.pi_min)
            slab_change = abs(previous_pi/pi-1.)
            good = (not diag['guarded'] and max(diag[k] for k in
                ['message_error','module_error','precision_error']) < config.tolerance)
            if config.learn_pi:
                good &= diag['em_error'] < config.em_tolerance and pi_change < config.em_tolerance
            if not good:
                raise ArithmeticError('Root candidate failed the original EM-VAMP convergence checks')
            if check+1 < config.consecutive_converged:
                next_gamma = (1-config.damping)*gamma+config.damping*proposed_gamma
                next_r = ((1-config.damping)*gamma[None,:]*r
                          +config.damping*proposed_gamma[None,:]*proposed_r)/next_gamma[None,:]
                state[:s] = np.asarray(next_r)[:,0]/scale
                state[s] = np.log(float(next_gamma[0]))
            previous_pi = pi
        elapsed = time.monotonic()-started
        report.update(status='converged', function_evaluations=evaluations, seconds=elapsed,
                      verified_consecutive_iterations=config.consecutive_converged)
        row = dict(posterior.trace[-1] if posterior.trace else {},
            iteration=posterior.iterations+config.consecutive_converged,
            seconds=posterior.seconds+elapsed, pi=[[pi]], slab_variance=[[d/pi]],
            gamma1=[float(gamma[0])], alpha1=[float(gamma[0]*jnp.mean(variance))],
            **{key:[diag[key]] for key in ['message_error','module_error','precision_error','em_error']},
            pi_change=[pi_change], slab_change=[slab_change], converged=[True], precision_guarded=[False],
            precision_safeguard_count=solver.precision_safeguard_count,
            prior_constraint_max_relative_error=float(abs(pi*(d/pi)-d)/d),
            fixed_point_refinement=report)
        result = replace(posterior, mean=np.asarray(mean), variance=np.asarray(variance), pip=np.asarray(pip),
            pi=np.array([[pi]]), slab_variance=np.array([[d/pi]]), precision=np.asarray(gamma),
            status=['converged'], trace=posterior.trace+[row], seconds=posterior.seconds+elapsed,
            iterations=row['iteration'], linear_solver=dict(solver.diagnostics(),fixed_point_refinement=report))
    except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:
        report.update(error=str(error), function_evaluations=evaluations, seconds=time.monotonic()-started)
        return replace(posterior, seconds=posterior.seconds+report['seconds'],
                       linear_solver=dict(solver.diagnostics(),fixed_point_refinement=report))
    if callback:
        callback(row)
    return result
