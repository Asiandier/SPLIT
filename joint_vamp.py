"""Joint sparse/background spike-and-slab EM-VAMP.

The genome-wide alpha prior has pi_a*v_a*tr(X.T@X)/n = Q_s. Its approximate
posterior is restricted to the fixed LASSO support; background priors obey
pi_g*v_g*M_g = theta_g. The four-state scalar posterior retains cross moments.
Only training data enter inference. See docs/joint_vamp.md for derivations.
"""

from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

from .grouped_ss_vamp import VAMPConfig, constrained_ss_m_step, validate_vamp_config
from .ss_linear import SupportPrecisionSolver


@jax.jit
def joint_denoise(r, precision, pi_a, var_a, pi_b, var_b):
    r, precision, pi_a, var_a, pi_b, var_b = jnp.broadcast_arrays(
        r, precision, pi_a, var_a, pi_b, var_b
    )
    za = jnp.array([0.0, 0.0, 1.0, 1.0])
    zb = jnp.array([0.0, 1.0, 0.0, 1.0])
    a = var_a[..., None] * za
    b = var_b[..., None] * zb
    prior_a = jnp.where(za > 0, pi_a[..., None], 1 - pi_a[..., None])
    prior_b = jnp.where(zb > 0, pi_b[..., None], 1 - pi_b[..., None])
    den = 1 / precision[..., None] + a + b
    log_weight = (
        jnp.log(prior_a)
        + jnp.log(prior_b)
        - 0.5 * (jnp.log(2 * jnp.pi * den) + r[..., None] ** 2 / den)
    )
    evidence = jax.scipy.special.logsumexp(log_weight, axis=-1)
    prob = jnp.exp(log_weight - evidence[..., None])
    ma = a / den * r[..., None]
    mb = b / den * r[..., None]
    va = a - a * a / den
    vb = b - b * b / den
    cab = -a * b / den
    mean_a = jnp.sum(prob * ma, axis=-1)
    mean_b = jnp.sum(prob * mb, axis=-1)
    second_a = jnp.sum(prob * (ma * ma + va), axis=-1)
    second_b = jnp.sum(prob * (mb * mb + vb), axis=-1)
    cross = jnp.sum(prob * (ma * mb + cab), axis=-1)
    total = ma + mb
    mean = mean_a + mean_b
    variance = jnp.maximum(
        jnp.sum(prob * (total * total + (a + b) / den / precision[..., None]), axis=-1)
        - mean * mean,
        0.0,
    )
    return dict(
        mean=mean,
        variance=variance,
        mean_alpha=mean_a,
        mean_background=mean_b,
        second_alpha=second_a,
        second_background=second_b,
        cross_moment=cross,
        pip_alpha=jnp.sum(prob * za, axis=-1),
        pip_background=jnp.sum(prob * zb, axis=-1),
        log_evidence=evidence,
        state_probabilities=prob,
    )


@jax.jit
def invert_mean(target, precision, pi_a, var_a, pi_b, var_b):
    # E[w|r] is odd and nondecreasing; its derivative is precision*Var[w|r].
    target, precision, pi_a, var_a, pi_b, var_b = jnp.broadcast_arrays(
        target, precision, pi_a, var_a, pi_b, var_b
    )
    active = (pi_a * var_a + pi_b * var_b) > 0
    goal = jnp.where(active, jnp.abs(target), 0.0)
    upper = jnp.maximum(goal, 1 / jnp.sqrt(precision))

    def expand(state):
        count, hi = state
        value = joint_denoise(hi, precision, pi_a, var_a, pi_b, var_b)["mean"]
        return count + 1, jnp.where(value < goal, hi * 2, hi)

    upper = jax.lax.while_loop(
        lambda st: (st[0] < 100)
        & jnp.any(
            joint_denoise(st[1], precision, pi_a, var_a, pi_b, var_b)["mean"] < goal
        ),
        expand,
        (0, upper),
    )[1]

    def bisect(_, state):
        lo, hi = state
        mid = (lo + hi) / 2
        value = joint_denoise(mid, precision, pi_a, var_a, pi_b, var_b)["mean"]
        return jnp.where(value < goal, mid, lo), jnp.where(value >= goal, mid, hi)

    lo, hi = jax.lax.fori_loop(0, 64, bisect, (jnp.zeros_like(upper), upper))
    return jnp.where(active, jnp.sign(target) * (lo + hi) / 2, 0.0)


def _make_em_update(em_map, evidence_at, config, active):
    resolution = max(config.em_acceleration_floor, config.em_tolerance)

    @jax.jit
    def update(r, gam, prob):
        def body(_, old):
            one = em_map(r, gam, old)
            if not config.em_acceleration:
                return one
            two = em_map(r, gam, one)
            delta = one - old
            curvature = two - 2 * one + old
            steps = -jnp.clip(
                jnp.abs(delta) / jnp.maximum(jnp.abs(curvature), 1e-30),
                1.0,
                config.em_max_step,
            )
            resolved = active & (
                jnp.abs(delta) > resolution * jnp.maximum(old, config.pi_min)
            )
            baseline = evidence_at(r, gam, two)

            def try_step(state):
                trial, steps, accepted, answer = state
                proposal = jnp.clip(
                    old - 2 * steps * delta + steps * steps * curvature,
                    jnp.maximum(config.pi_min, 0.8 * old),
                    jnp.minimum(1.0, 1.25 * old),
                )
                proposal = jnp.where(active, jnp.where(resolved, proposal, two), 0.0)
                polished = em_map(r, gam, proposal)
                direction = jnp.all(~resolved | ((polished - old) * delta >= 0))
                better = (evidence_at(r, gam, polished) >= baseline) & direction
                answer = jnp.where(better, polished, answer)
                return (
                    trial + 1,
                    -jnp.maximum(jnp.abs(steps) * 0.5, 1.0),
                    better | jnp.all(jnp.abs(steps) <= 1),
                    answer,
                )

            return jax.lax.while_loop(
                lambda st: (st[0] < 42) & ~st[2],
                try_step,
                (0, steps, ~jnp.any(resolved), two),
            )[3]

        optimum = jax.lax.fori_loop(0, config.em_inner_steps, body, prob)
        candidate = (1 - config.em_damping) * prob + config.em_damping * optimum
        ordinary = (1 - config.em_damping) * prob + config.em_damping * em_map(
            r, gam, prob
        )
        # No acceptance of a decrease caused by damping an accelerated move.
        return jnp.where(
            evidence_at(r, gam, candidate) >= evidence_at(r, gam, prob),
            candidate,
            ordinary,
        )

    return update


def fit_joint(
    products,
    response,
    groups,
    support,
    q_sparse,
    alpha_normalizer,
    theta,
    normalizers,
    noise,
    solver,
    *,
    config=None,
    initial_mean=None,
    initial_pi=None,
    callback=None,
):
    """Fit the fixed-budget model to one response using supplied linear operators.

    ``alpha_normalizer`` is the global fitting-genotype trace, not the support
    trace. An unconverged result is diagnostic only; ``refine_joint`` owns the
    retry and complete-pair fallback policy.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("FP64 required")
    cfg = config or VAMPConfig(max_iterations=10000)
    validate_vamp_config(cfg)
    start = time.monotonic()
    p = products.p
    groups = np.asarray(groups)
    support = np.asarray(support)
    theta = np.asarray(theta, float)
    normalizers = np.asarray(normalizers, float)
    response = np.asarray(response, float)
    if (
        p < 1
        or products.n < 1
        or theta.ndim != 1
        or not len(theta)
        or groups.shape != (p,)
        or support.ndim != 1
        or not np.issubdtype(groups.dtype, np.integer)
        or not np.issubdtype(support.dtype, np.integer)
        or normalizers.shape != theta.shape
        or response.shape not in ((products.n,), (products.n, 1))
        or any(
            not np.isfinite(v).all()
            for v in (theta, normalizers, response, [noise, q_sparse, alpha_normalizer])
        )
        or np.any(normalizers < 0)
        or np.any((theta > 0) & (normalizers <= 0))
    ):
        raise ValueError("Invalid joint model dimensions, labels, or finite inputs")
    ng = len(theta)
    s = len(support)
    if (
        groups.shape != (p,)
        or groups.min() < 0
        or groups.max() >= ng
        or len(np.unique(support)) != s
        or np.any(support < 0)
        or np.any(support >= p)
        or np.any(theta < 0)
        or noise <= 0
        or q_sparse < 0
        or alpha_normalizer <= 0
        or (q_sparse > 0 and s == 0)
    ):
        raise ValueError("Invalid model inputs")
    smask = np.zeros(p, bool)
    smask[support] = True
    active = (theta[groups] > 0) | ((q_sparse > 0) & smask)
    if solver is None or not np.array_equal(solver.active, active):
        raise ValueError("Wrong active linear system")
    pa = int(active.sum())
    if not pa:
        raise ValueError("Empty model")
    counts = jnp.asarray(np.r_[p, np.bincount(groups, minlength=ng)], float)
    norms = np.r_[alpha_normalizer, normalizers]
    budgets = np.r_[q_sparse, theta]
    d = jnp.asarray(budgets / np.where(norms > 0, norms, 1.0))
    initial = np.asarray(
        (
            np.r_[s / p, np.repeat(cfg.initial_pi, ng)]
            if initial_pi is None
            else initial_pi
        ),
        float,
    )
    if (
        initial.shape != (ng + 1,)
        or not np.isfinite(initial).all()
        or np.any((initial < 0) | (initial > 1))
        or np.any(initial[budgets > 0] <= 0)
    ):
        raise ValueError("Invalid initial pi")
    prob = jnp.where(d > 0, jnp.maximum(jnp.asarray(initial), cfg.pi_min), 0.0)
    group_ids = jnp.asarray(groups)
    smask = jnp.asarray(smask[:, None])
    active_j = jnp.asarray(active[:, None])
    y = jnp.asarray(np.asarray(response, float).reshape(products.n, 1))
    precision = jnp.array([products.n / (noise + budgets.sum())])
    noise_j = jnp.array([noise])
    natural = isinstance(solver, SupportPrecisionSolver)
    linear_products = (
        products.restricted(active) if hasattr(products, "restricted") else products
    )
    if natural:
        solver.prepare_natural_module(products, y, noise_j)

    def prior_params(prob):
        slab = jnp.where(d > 0, d / jnp.maximum(prob, cfg.pi_min), 0.0)
        return (
            jnp.where(smask, prob[0], 0.0),
            jnp.where(smask, slab[0], 0.0),
            prob[1:][group_ids, None],
            slab[1:][group_ids, None],
        )

    @jax.jit
    def post(r, gam, prob):
        return joint_denoise(r, gam[None, :], *prior_params(prob))

    def moments(posterior):
        a = jnp.concatenate(
            [
                posterior["pip_alpha"].sum(0),
                jnp.bincount(
                    group_ids, weights=posterior["pip_background"][:, 0], length=ng
                ),
            ]
        )
        b = jnp.concatenate(
            [
                posterior["second_alpha"].sum(0),
                jnp.bincount(
                    group_ids, weights=posterior["second_background"][:, 0], length=ng
                ),
            ]
        )
        return a, b

    @jax.jit
    def em_map(r, gam, prob):
        a, b = moments(post(r, gam, prob))
        return constrained_ss_m_step(a, b, counts, d, cfg.pi_min)[0]

    @jax.jit
    def evidence_at(r, gam, prob):
        # Subtract the parameter-independent all-zero channel for precision.
        relative = post(r, gam, prob)["log_evidence"] + 0.5 * (
            jnp.log(2 * jnp.pi / gam)[None, :] + gam[None, :] * r * r
        )
        return jnp.sum(relative) + (p - s) * jnp.log1p(-jnp.where(p > s, prob[0], 0.0))

    update = _make_em_update(em_map, evidence_at, cfg, d > 0)

    r = jnp.zeros((p, 1), jnp.float64)
    initialization = dict(
        method="zero",
        pi=np.asarray(prob).tolist(),
        slab=np.asarray(
            jnp.where(d > 0, d / jnp.maximum(prob, cfg.pi_min), 0.0)
        ).tolist(),
        precision=float(precision[0]),
        alpha_count=p,
        modeled_support_size=s,
        alpha_global_normalizer=float(alpha_normalizer),
        background_counts=np.asarray(counts[1:]).tolist(),
    )
    if initial_mean is not None:
        target = np.asarray(initial_mean, float)
        if target.shape not in ((p,), (p, 1)) or not np.isfinite(target).all():
            raise ValueError("Invalid initial total posterior mean")
        target = target.reshape(p, 1)
        target = np.where(active[:, None], target, 0.0)
        r = invert_mean(jnp.asarray(target), precision[None, :], *prior_params(prob))
        error = float(
            np.linalg.norm(np.asarray(post(r, precision, prob)["mean"]) - target)
            / max(np.linalg.norm(target), 1e-30)
        )
        if not np.isfinite(error) or error > 1e-10:
            raise ArithmeticError(("Initial total posterior mean inversion", error))
        initialization.update(
            method="native_total_posterior_mean", relative_error=error
        )

    @jax.jit
    def rel(a, b):
        return jnp.sqrt(
            jnp.sum((a - b) ** 2)
            / jnp.maximum(jnp.maximum(jnp.sum(a * a), jnp.sum(b * b)), 1e-30)
        )

    @jax.jit
    def nonlinear(r, gam, prob):
        posterior = post(r, gam, prob)
        divergence = gam * jnp.sum(posterior["variance"], axis=0) / pa
        emerror = jnp.max(
            jnp.where(
                d > 0,
                jnp.abs(em_map(r, gam, prob) - prob) / jnp.maximum(prob, cfg.pi_min),
                0.0,
            )
        )
        return posterior["mean"], divergence, emerror

    trace = []
    stable = 0
    converged = False
    for iteration in range(1, cfg.max_iterations + 1):
        old = prob
        if cfg.learn_pi and iteration > cfg.em_warmup:
            prob = update(r, precision, prob)
        mean, div, emerr = nonlinear(r, precision, prob)
        div_host = float(div[0])
        guard = False
        if not np.isfinite(div_host) or div_host <= 0:
            raise ArithmeticError("Invalid joint divergence")
        if natural:
            eta, guarded = solver.stabilize_eta(precision / div, precision)
            guard = bool(np.any(np.asarray(guarded)))
            m2, newprecision, newr = solver.natural_module(
                eta[None, :] * mean - precision[None, :] * r, eta - precision
            )
        else:
            if div_host >= 1:
                raise ArithmeticError("Joint divergence >=1 in sample-space solver")
            gamma2 = precision * (1 - div) / div
            r2 = jnp.where(
                active_j, (mean - div[None, :] * r) / (1 - div)[None, :], 0.0
            )
            solved, fraction = solver.solve_and_trace(
                y - linear_products.forward(r2), gamma2 * noise_j
            )
            correction = jnp.where(active_j, linear_products.transpose(solved), 0.0)
            m2 = r2 + correction
            newprecision = gamma2 * fraction / (1 - fraction)
            newr = r2 + correction / fraction[None, :]
        errors = np.asarray(
            jnp.stack(
                [
                    rel(newr, r),
                    rel(m2, mean),
                    jnp.max(jnp.abs(newprecision - precision) / precision),
                    emerr,
                    jnp.max(
                        jnp.where(
                            d > 0,
                            jnp.abs(prob - old) / jnp.maximum(old, cfg.pi_min),
                            0.0,
                        )
                    ),
                ]
            )
        )
        if not np.isfinite(errors).all():
            raise ArithmeticError("Nonfinite joint diagnostics")
        eligible = iteration >= cfg.min_iterations and (
            not cfg.learn_pi or iteration > cfg.em_warmup
        )
        good = max(errors[:3]) < cfg.tolerance and not guard
        if cfg.learn_pi:
            good = good and max(errors[3:]) < cfg.em_tolerance
        stable = stable + 1 if eligible and good else 0
        converged = stable >= cfg.consecutive_converged
        if not converged:
            nextprecision = (1 - cfg.damping) * precision + cfg.damping * newprecision
            r = (
                (1 - cfg.damping) * precision[None, :] * r
                + cfg.damping * newprecision[None, :] * newr
            ) / nextprecision[None, :]
            precision = nextprecision
        if (
            iteration == 1
            or iteration % cfg.log_every == 0
            or converged
            or iteration == cfg.max_iterations
        ):
            row = dict(
                iteration=iteration,
                seconds=time.monotonic() - start,
                pi=np.asarray(prob).tolist(),
                alpha_slab_variance=float(
                    jnp.where(d[0] > 0, d[0] / jnp.maximum(prob[0], cfg.pi_min), 0.0)
                ),
                message_error=float(errors[0]),
                module_error=float(errors[1]),
                precision_error=float(errors[2]),
                em_error=float(errors[3]),
                pi_change=float(errors[4]),
                divergence=div_host,
                precision_guarded=guard,
                converged=converged,
            )
            trace.append(row)
            if callback:
                callback(row)
        if converged:
            break
    posterior = jax.device_get(post(r, precision, prob))
    slab = np.asarray(jnp.where(d > 0, d / jnp.maximum(prob, cfg.pi_min), 0.0))
    np.testing.assert_allclose(
        np.asarray(prob) * slab * np.asarray(norms), budgets, rtol=1e-12, atol=1e-14
    )
    return dict(
        posterior=posterior,
        status="converged" if converged else "max_iterations",
        pi=np.asarray(prob),
        slab=slab,
        iterations=iteration,
        seconds=time.monotonic() - start,
        trace=trace,
        initialization=initialization,
        linear_solver=solver.diagnostics(),
        alpha_count=p,
        modeled_support_size=s,
        alpha_global_normalizer=float(alpha_normalizer),
        background_counts=np.asarray(counts[1:]),
        variance_budgets=budgets,
        background_variance_budgets=theta,
        prior_expected_variance=np.asarray(prob) * slab * np.asarray(norms),
        precision=np.asarray(precision),
    )
