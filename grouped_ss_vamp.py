"""GPU EM-VAMP for a fixed residual and grouped SS priors.

The default constrains pi[g] * slab_variance[g] = theta[g] / M[g].
An unconstrained mode jointly learns pi and slab variance in all groups.
The Gaussian likelihood uses all groups jointly. No foreground parameters,
noise variance or hyperpriors are estimated here. Genotypes are SNP-major.

References: Rangan et al., arXiv:1610.03082 (VAMP); Fletcher and Schniter,
arXiv:1602.08207 (EM-VAMP); Varadhan and Roland,
doi:10.1111/j.1467-9469.2007.00585.x (safeguarded SQUAREM inner EM).
The optional acceleration applies only to the scalar-channel constrained EM,
with an exact scalar marginal-likelihood acceptance check. That objective is
not the full regression likelihood, and VAMP need not be monotone in it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import time
import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def constrained_ss_m_step(a, b, counts, d, pi_min=1e-8):
    """Maximize fixed-q Q on pi*v=d; arrays broadcast over groups/cases.

    a=sum(PIP), b=sum(E[beta**2]), counts=ALL SNPs in that group. The
    rationalized quadratic root avoids cancellation. Positive d needs a
    strictly positive numerical lower bound. d=0 denotes an inactive group.
    """
    safe_d = jnp.where(d > 0, d, 1.)
    k = b/(2*safe_d)
    c = counts + .5*a + k
    disc = jnp.maximum(c*c - 6*a*k, 0.)
    root = 3*a/jnp.maximum(c+jnp.sqrt(disc), jnp.finfo(c.dtype).tiny)
    root = jnp.where(d > 0, jnp.clip(root, pi_min, 1.), 0.)
    return root, jnp.where(d > 0, safe_d/jnp.maximum(root, pi_min), 0.)


@jax.jit
def unconstrained_ss_m_step(a, b, counts, pi_min=1e-8, variance_min=1e-16):
    """No-hyperprior SS EM: pi=A/P, v=B/A; bounds are numerical only."""
    pi = jnp.clip(a/counts, pi_min, 1.)
    slab = jnp.maximum(b/jnp.maximum(a, jnp.finfo(b.dtype).tiny), variance_min)
    return pi, slab


@jax.jit
def spike_slab_denoise(r, precision, pi, slab_variance):
    """Exact scalar SS posterior; precision is pseudo-observation precision.

    Returns mean, variance, PIP, conditional mean/variance and log evidence
    relative to the all-spike channel. pi=0/1 are handled exactly.
    """
    gain = precision*slab_variance/(1+precision*slab_variance)
    conditional_variance = slab_variance/(1+precision*slab_variance)
    conditional_mean = gain*r
    log_bf = -.5*jnp.log1p(precision*slab_variance) + .5*precision*gain*r*r
    spike = jnp.log1p(-pi)
    slab = jnp.log(pi)+log_bf
    evidence = jnp.logaddexp(spike, slab)
    pip = jnp.exp(slab-evidence)
    mean = pip*conditional_mean
    variance = pip*conditional_variance + pip*(1-pip)*conditional_mean**2
    return mean, variance, pip, conditional_mean, conditional_variance, evidence


@jax.jit
def ss_message_from_mean(mean, precision, pi, slab_variance):
    """Invert the scalar SS mean to initialize a consistent VAMP message.

    At fixed positive prior/precision the mean is odd and strictly monotone.
    Its positive root is bracketed by target/gain and target/(gain*PIP(0)),
    because PIP(r) increases with |r|. Batched GPU bisection in log-r avoids
    loss of accuracy for very wide brackets and needs no genotype products.
    Inactive priors always return zero, regardless of input mean.
    """
    gain = precision*slab_variance/(1+precision*slab_variance)
    active = (pi > 0) & (slab_variance > 0)
    safe_gain = jnp.where(active, gain, 1.)
    log_odds_zero = jnp.log(pi)-jnp.log1p(-pi)-.5*jnp.log1p(precision*slab_variance)
    target = jnp.where(active, jnp.abs(mean), 0.)
    nonzero = active & (target > 0)
    log_pip_zero = -jax.nn.softplus(-log_odds_zero)
    lower = jnp.log(jnp.where(nonzero, target, 1.))-jnp.log(safe_gain)
    upper = lower-jnp.where(active, log_pip_zero, 0.)

    def bisect(_, bracket):
        lo, hi = bracket
        middle = lo+.5*(hi-lo)
        message = jnp.exp(middle)
        pip = jax.nn.sigmoid(log_odds_zero+.5*precision*gain*message*message)
        mapped = pip*gain*message
        below = mapped < target
        return jnp.where(below, middle, lo), jnp.where(below, hi, middle)

    lower, upper = jax.lax.fori_loop(0, 64, bisect, (lower, upper))
    return jnp.where(nonzero, jnp.sign(mean)*jnp.exp(lower+.5*(upper-lower)), 0.)


class GenotypeProducts:
    """Resident genotype storage, chunked products without a full transpose.

    Native inputs use FP32 (~4*n*p bytes); explicit FP64 inputs retain their
    precision. Each product casts only a chunk to the coefficient dtype and
    never constructs a p-by-p LD matrix or a full FP64 copy of FP32 inputs.
    Multiple models sharing the genotype are evaluated in one matrix product.
    """
    def __init__(self, snp_major, chunk_size=4096):
        self.x = jax.device_put(snp_major)
        self.x.block_until_ready()
        if (self.x.ndim != 2 or min(self.x.shape) < 1
                or not (np.issubdtype(self.x.dtype, np.floating)
                        or np.issubdtype(self.x.dtype, np.integer))):
            raise ValueError('Genotypes must be a nonempty real SNP-by-sample matrix')
        if not bool(jnp.all(jnp.isfinite(self.x))):
            raise ValueError('Standardized genotypes must be finite')
        if int(chunk_size) != chunk_size or chunk_size < 1:
            raise ValueError('Genotype chunk size must be a positive integer')
        self.p, self.n = self.x.shape
        self.chunk_size = min(int(chunk_size), self.p)
        size, p, n = self.chunk_size, self.p, self.n
        whole = p//size*size

        @jax.jit
        def forward(x, b):
            def body(i, total):
                block = jax.lax.dynamic_slice(x, (i*size, 0), (size, n)).astype(b.dtype)
                coef = jax.lax.dynamic_slice(b, (i*size, 0), (size, b.shape[1]))
                return total+jnp.matmul(block.T, coef, precision=jax.lax.Precision.HIGHEST)
            total = jax.lax.fori_loop(0, whole//size, body, jnp.zeros((n, b.shape[1]), b.dtype))
            if whole < p:
                total += jnp.matmul(x[whole:].astype(b.dtype).T, b[whole:], precision=jax.lax.Precision.HIGHEST)
            return total

        @jax.jit
        def transpose(x, b):
            def body(i, total):
                block = jax.lax.dynamic_slice(x, (i*size, 0), (size, n)).astype(b.dtype)
                score = jnp.matmul(block, b, precision=jax.lax.Precision.HIGHEST)
                return jax.lax.dynamic_update_slice(total, score, (i*size, 0))
            total = jax.lax.fori_loop(0, whole//size, body, jnp.zeros((p, b.shape[1]), b.dtype))
            if whole < p:
                total = total.at[whole:].set(jnp.matmul(x[whole:].astype(b.dtype), b, precision=jax.lax.Precision.HIGHEST))
            return total
        self._forward = forward
        self._transpose = transpose

    def forward(self, effects):
        return self._forward(self.x, effects)

    def transpose(self, residual):
        return self._transpose(self.x, residual)


class SampleSpectrum:
    """Exact sample-space LMMSE solve and trace, reused over VAMP iterations."""
    def __init__(self, eigenvalues, eigenvectors, active_snp_count):
        values = np.asarray(eigenvalues, float)
        vectors = jnp.asarray(eigenvectors, dtype=jnp.float64)
        if (values.ndim != 1 or len(values) == 0
                or vectors.shape != (len(values), len(values))):
            raise ValueError('Invalid eigensystem dimensions')
        if not np.all(np.isfinite(values)) or not bool(jnp.all(jnp.isfinite(vectors))):
            raise ValueError('Sample eigensystem must be finite')
        if values.min() < -1e-9*max(1., values.max()):
            raise ValueError('Sample Gram must be positive semidefinite')
        self.values = jax.device_put(np.maximum(values, 0))
        self.vectors = jax.device_put(vectors)
        self.p = int(active_snp_count)
        self.n = len(values)
        if self.p <= 0:
            raise ValueError('Need at least one active SNP')

    @classmethod
    def from_genotype(cls, snp_major, active=None):
        x = np.asarray(snp_major, float)
        if active is not None:
            x = x[np.asarray(active, bool)]
        eig, vec = np.linalg.eigh(x.T@x)
        return cls(eig, vec, len(x))

    def solve_and_trace(self, rhs, shift):
        return _spectral_solve(self.values, self.vectors, rhs, shift, self.p)


@jax.jit
def _spectral_solve(eig, vec, rhs, shift, p):
    projected = jnp.matmul(vec.T, rhs, precision=jax.lax.Precision.HIGHEST)
    inverse = 1/(eig[:, None]+shift[None, :])
    solved = jnp.matmul(vec, projected*inverse, precision=jax.lax.Precision.HIGHEST)
    trace_fraction = jnp.sum(eig[:, None]*inverse, axis=0)/p
    return solved, trace_fraction


@dataclass(frozen=True)
class VAMPConfig:
    max_iterations: int = 1000
    min_iterations: int = 30
    damping: float = .2
    em_damping: float = .05
    em_warmup: int = 10
    em_inner_steps: int = 3
    em_acceleration: bool = True
    em_max_step: float = 1e12
    em_acceleration_floor: float = 1e-9
    initial_pi: float = .1
    pi_min: float = 1e-8
    tolerance: float = 1e-5
    em_tolerance: float = 1e-5
    consecutive_converged: int = 5
    log_every: int = 10
    learn_pi: bool = True
    variance_constraint: bool = True
    slab_variance_min: float = 1e-16


@dataclass
class VAMPResult:
    mean: np.ndarray
    variance: np.ndarray
    pip: np.ndarray
    pi: np.ndarray
    slab_variance: np.ndarray
    status: list[str]
    trace: list[dict]
    seconds: float
    iterations: int
    precision: np.ndarray
    initialization: dict | None = None
    final_state: dict | None = None


def fit_grouped_ss_vamp(products: GenotypeProducts, residual, groups, theta,
                        normalizers, noise_variance, spectrum: SampleSpectrum,
                        config: VAMPConfig | None = None,
                        callback: Callable[[dict], None] | None = None,
                        initial_mean=None, initial_state=None) -> VAMPResult:
    """Fit a batch of cases with identical active groups, without test data.

    theta is (G, cases), residual is (n, cases), noise is (cases,). A single
    case may pass one-dimensional inputs. In constrained mode, zero-theta
    SNPs are excluded from Onsager averages and have zero background effects.
    Free mode opens all groups; native theta is used only to initialize v.
    Optional initial_mean (p,cases) is the desired first nonlinear posterior
    mean. It is mapped to a compatible r1 message by scalar SS inversion;
    constrained zero-theta entries are masked to zero even if supplied nonzero.
    initial_state resumes a constrained fit from a previous final_state, on
    exactly the same data and model. Iteration offsets preserve EM warmup.
    Stops on *undamped* message, module agreement, precision and constrained
    EM fixed-point errors, rather than merely small damped iterate changes.
    """
    config = config or VAMPConfig()
    if initial_state is not None and (initial_mean is not None or not config.variance_constraint):
        raise ValueError('Continuation requires a constrained fit without initial_mean')
    if not jax.config.jax_enable_x64:
        raise RuntimeError('Enable jax_enable_x64 before fitting FP64 EM-VAMP')
    if not (0 < config.damping <= 1 and 0 < config.em_damping <= 1):
        raise ValueError('Damping must be in (0,1]')
    if config.max_iterations < 1 or config.em_inner_steps < 1 or config.em_max_step < 1:
        raise ValueError('Invalid iteration count or EM acceleration limit')
    for name in ['max_iterations', 'min_iterations', 'em_warmup', 'em_inner_steps',
                 'consecutive_converged', 'log_every']:
        value = getattr(config, name)
        minimum = 0 if name in ['min_iterations', 'em_warmup'] else 1
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if (not np.isfinite(config.em_max_step) or not np.isfinite(config.em_acceleration_floor)
            or config.em_acceleration_floor < 0 or not np.isfinite(config.tolerance)
            or config.tolerance <= 0 or not np.isfinite(config.em_tolerance)
            or config.em_tolerance <= 0):
        raise ValueError('Invalid numerical tolerances or EM acceleration settings')
    if not np.isfinite(config.slab_variance_min) or config.slab_variance_min <= 0:
        raise ValueError('Slab variance numerical floor must be positive')
    if not (0 < config.pi_min <= config.initial_pi <= 1):
        raise ValueError('Invalid numerical pi bounds or initialization')
    start = time.monotonic()
    groups_host = np.asarray(groups)
    t_host = np.asarray(theta, float)
    if t_host.ndim == 1:
        t_host = t_host[:, None]
    y_host = np.asarray(residual, float)
    if y_host.ndim == 1:
        y_host = y_host[:, None]
    if t_host.ndim != 2 or min(t_host.shape) < 1:
        raise ValueError('theta must be a nonempty group-by-case matrix')
    ng, batch = t_host.shape
    if (not np.issubdtype(groups_host.dtype, np.integer)
            or groups_host.shape != (products.p,)
            or groups_host.min() < 0 or groups_host.max() >= ng):
        raise ValueError('groups must assign every SNP to an integer group in theta')
    counts_host = np.bincount(groups_host, minlength=ng).astype(float)
    norm_host = np.asarray(normalizers, float)
    noise_host = np.atleast_1d(np.asarray(noise_variance, float))
    if (y_host.shape != (products.n, batch) or groups_host.shape != (products.p,)
            or groups_host.min() < 0 or groups_host.max() >= ng
            or norm_host.shape != (ng,) or np.any(norm_host < 0)
            or noise_host.shape != (batch,) or np.any(noise_host <= 0)
            or not np.all(np.isfinite(noise_host)) or not np.all(np.isfinite(y_host))
            or not np.all(np.isfinite(norm_host))
            or np.any(t_host < 0) or not np.all(np.isfinite(t_host))):
        raise ValueError('Inconsistent dimensions or invalid model parameters')
    if np.any(counts_host == 0):
        raise ValueError('Every modeled group must contain at least one SNP')
    if np.any((norm_host == 0) & (np.any(t_host > 0, axis=1) | (not config.variance_constraint))):
        raise ValueError('Active groups require positive GRM normalizers')
    active_groups = t_host > 0 if config.variance_constraint else np.ones_like(t_host, bool)
    if not np.all(active_groups == active_groups[:, :1]):
        raise ValueError('Cases in one batch must have identical active groups')
    active_host = active_groups[groups_host, 0]
    p_active = int(active_host.sum())
    if spectrum.p != p_active or spectrum.n != products.n:
        raise ValueError('Eigensystem must cover exactly the active SNP groups')
    y, group_ids = jax.device_put(y_host), jax.device_put(groups_host)
    active = jax.device_put(active_host[:, None])
    d_host = t_host/np.where(norm_host > 0, norm_host, 1.)[:, None]
    if not config.variance_constraint:
        # Native components initialize the free model but never fix a group.
        # Zero native components receive the positive pooled initial variance
        # so EM can leave the all-zero absorbing state. No test data is used.
        pooled = np.where(t_host.sum(axis=0) > 0, t_host.sum(axis=0), noise_host)/norm_host.sum()
        d_host = np.where(d_host > 0, d_host, pooled[None, :])
    d = jax.device_put(d_host)
    counts = jax.device_put(counts_host[:, None])
    noise = jax.device_put(noise_host)
    pi = jnp.where(d > 0, config.initial_pi, 0.)
    slab = jnp.where(d > 0, d/jnp.maximum(pi, config.pi_min), 0.)
    r1 = jnp.zeros((products.p, batch), jnp.float64)
    # Training-only scale. The next linear module refines this precision.
    initial_total = t_host.sum(axis=0) if config.variance_constraint else np.sum(d_host*norm_host[:, None], axis=0)
    gamma1 = products.n/jnp.maximum(noise+jax.device_put(initial_total), 1e-12)
    initialization = dict(method='zero', initial_pi=np.asarray(pi).T.tolist(),
                          initial_slab_variance=np.asarray(slab).T.tolist(),
                          initial_message_precision=np.asarray(gamma1).tolist())
    if initial_mean is not None:
        target_host = np.asarray(initial_mean, float)
        if target_host.ndim == 1:
            target_host = target_host[:, None]
        if target_host.shape != (products.p, batch) or not np.all(np.isfinite(target_host)):
            raise ValueError('initial_mean must be finite with shape (p,cases)')
        initialization['inactive_initial_entries_discarded'] = np.count_nonzero(
            np.where(active_host[:, None], 0., target_host), axis=0).tolist()
        target_host = np.where(active_host[:, None], target_host, 0.)
        target = jax.device_put(target_host)
        r1 = ss_message_from_mean(target, gamma1[None, :], pi[group_ids], slab[group_ids])
        actual = np.asarray(spike_slab_denoise(r1, gamma1[None, :], pi[group_ids], slab[group_ids])[0])
        relative = np.linalg.norm(actual-target_host, axis=0)/np.maximum(np.linalg.norm(target_host, axis=0), 1e-30)
        if not np.all(np.isfinite(np.asarray(r1))) or np.max(relative) > 1e-10:
            raise ArithmeticError('Initial posterior mean inversion failed')
        initialization.update(method='supplied_posterior_mean',
                              first_mean_relative_l2_error=relative.tolist(),
                              first_mean_max_abs_error=np.max(np.abs(actual-target_host), axis=0).tolist())

    offset = 0
    stable = np.zeros(batch, int)
    done = np.zeros(batch, bool)
    if initial_state is not None:
        saved_pi = np.asarray(initial_state['pi'], float)
        saved_r1 = np.asarray(initial_state['r1'], float)
        saved_precision = np.asarray(initial_state['gamma1'], float)
        stable = np.asarray(initial_state.get('stable', stable))
        done = np.asarray(initial_state.get('done', done), bool)
        offset = initial_state.get('iterations', 0)
        if (saved_pi.shape != (ng, batch) or saved_r1.shape != (products.p, batch)
                or saved_precision.shape != (batch,) or stable.shape != (batch,)
                or done.shape != (batch,)):
            raise ValueError('Inconsistent continuation state shapes')
        if (not np.all(np.isfinite(saved_r1)) or not np.all(np.isfinite(saved_precision))
                or np.any(saved_precision <= 0) or not np.all(np.isfinite(saved_pi))
                or np.any(saved_pi < 0) or np.any(saved_pi > 1)
                or np.any(saved_pi[d_host > 0] < config.pi_min)
                or np.any(saved_pi[d_host == 0] != 0)
                or not np.issubdtype(stable.dtype, np.integer) or np.any(stable < 0)
                or not isinstance(offset, (int, np.integer)) or offset < 0):
            raise ValueError('Invalid continuation messages or probabilities')
        stable, done = stable.copy(), done.copy()
        pi, r1, gamma1 = map(jax.device_put, (saved_pi, saved_r1, saved_precision))
        r1 = jnp.where(active, r1, 0.)
        slab = jnp.where(d > 0, d/jnp.maximum(pi, config.pi_min), 0.)
        initialization = dict(method='continued_EM_VAMP_state',
                              initial_pi=saved_pi.T.tolist(),
                              initial_message_precision=saved_precision.tolist(),
                              iteration_offset=int(offset))

    def reduce_group(values):
        return jnp.stack([jnp.sum(jnp.where((group_ids == g)[:, None], values, 0.), axis=0)
                          for g in range(ng)])

    def posterior(r, gam, prob):
        slab = jnp.where(d > 0, d/jnp.maximum(prob, config.pi_min), 0.)
        return spike_slab_denoise(r, gam[None, :], prob[group_ids], slab[group_ids])

    @jax.jit
    def em_map(r, gam, prob):
        mean, var, pip, mu, s2, evidence = posterior(r, gam, prob)
        a = reduce_group(pip)
        b = reduce_group(pip*(s2+mu*mu))
        new_pi, _ = constrained_ss_m_step(a, b, counts, d, config.pi_min)
        return new_pi, reduce_group(evidence)

    @jax.jit
    def evidence_at(r, gam, prob):
        return reduce_group(posterior(r, gam, prob)[-1])

    @jax.jit
    def learn_prior(r, gam, prob):
        # SQUAREM acceleration of the constrained EM fixed-point mapping.
        # Guard each group independently with the exact scalar evidence.
        def body(_, old):
            one, _ = em_map(r, gam, old)
            if not config.em_acceleration:
                return one
            two, _ = em_map(r, gam, one)
            delta = one-old
            curvature = two-2*one+old
            step = -jnp.clip(jnp.abs(delta)/jnp.maximum(jnp.abs(curvature), 1e-30), 1., config.em_max_step)
            baseline_evidence = evidence_at(r, gam, two)
            def attempt(state):
                trial, current_step, accepted, result = state
                # Keep extrapolation in the local EM basin. A near-zero
                # curvature estimate can otherwise suggest a jump to the
                # all-spike boundary even when the ordinary EM step points
                # in the opposite direction. This is a numerical safeguard,
                # not a restriction on the eventual model pi.
                proposal = jnp.clip(old-2*current_step*delta+current_step*current_step*curvature,
                                    jnp.maximum(config.pi_min, .8*old), jnp.minimum(1., 1.25*old))
                proposal = jnp.where(d > 0, proposal, 0.)
                polished, _ = em_map(r, gam, proposal)
                same_direction = (polished-old)*delta >= 0
                resolved = jnp.abs(delta) > config.em_acceleration_floor*jnp.maximum(old, config.pi_min)
                better = ((evidence_at(r, gam, polished) >= baseline_evidence-1e-10)
                          & same_direction & resolved)
                result = jnp.where(better & ~accepted, polished, result)
                # At step -1 another backtracking trial would be identical.
                # Keep the ordinary-EM fallback for rejected groups and stop
                # retrying them, while other groups can continue backtracking.
                accepted |= better | (jnp.abs(current_step) <= 1.)
                return trial+1, -jnp.maximum(jnp.abs(current_step)*.5, 1.), accepted, result
            # Unaccepted acceleration always falls back to two ordinary EM
            # steps; backtracking uses the current training scalar channel.
            # Already stationary EM maps need no curvature extrapolation.
            resolved = jnp.abs(delta) > config.em_acceleration_floor*jnp.maximum(old, config.pi_min)
            initial = (0, step, ~resolved, two)
            return jax.lax.while_loop(lambda s: (s[0] < 42) & jnp.any(~s[2]), attempt, initial)[3]
        optimum = jax.lax.fori_loop(0, config.em_inner_steps, body, prob)
        return prob+config.em_damping*(optimum-prob)

    @jax.jit
    def nonlinear(r, gam, prob):
        mean, var, pip, mu, s2, evidence = posterior(r, gam, prob)
        divergence = gam*jnp.sum(var, axis=0)/p_active
        new_pi, _ = constrained_ss_m_step(reduce_group(pip), reduce_group(pip*(s2+mu*mu)),
                                         counts, d, config.pi_min)
        # Relative to pi, not the total count: small rare-variant pi changes
        # must not falsely satisfy the EM stopping criterion.
        em_error = jnp.max(jnp.where(d > 0, jnp.abs(new_pi-prob)/jnp.maximum(prob, config.pi_min), 0.), axis=0)
        return mean, var, pip, divergence, em_error

    def free_posterior(r, gam, params):
        slab_variance = jnp.exp(params[1])/jnp.maximum(params[0], config.pi_min)
        return spike_slab_denoise(r, gam[None, :], params[0][group_ids], slab_variance[group_ids])

    @jax.jit
    def free_em_map(r, gam, params):
        mean, var, pip, mu, s2, evidence = free_posterior(r, gam, params)
        new_pi, new_slab = unconstrained_ss_m_step(reduce_group(pip),
            reduce_group(pip*(s2+mu*mu)), counts, config.pi_min, config.slab_variance_min)
        return jnp.stack([new_pi, jnp.log(new_pi*new_slab)]), reduce_group(evidence)

    @jax.jit
    def free_evidence_at(r, gam, params):
        return reduce_group(free_posterior(r, gam, params)[-1])

    @jax.jit
    def learn_free_prior(r, gam, params):
        # Two-parameter SQUAREM in (pi, log(pi*v)). The second coordinate
        # represents total prior variance per marker, which is better
        # identified than slab variance when pi and v trade off. Both remain
        # free; the EM map still sets pi=A/P and v=B/A. Damping is geometric
        # in the variance coordinate and is safeguarded by scalar evidence.
        def body(_, old):
            one, _ = free_em_map(r, gam, old)
            if not config.em_acceleration:
                return one
            two, _ = free_em_map(r, gam, one)
            delta, curvature = one-old, two-2*one+old
            step = -jnp.clip(jnp.sqrt(jnp.sum(delta*delta, axis=0)/
                jnp.maximum(jnp.sum(curvature*curvature, axis=0), 1e-60)), 1., config.em_max_step)
            baseline = free_evidence_at(r, gam, two)
            resolved = jnp.maximum(jnp.abs(delta[0])/jnp.maximum(old[0], config.pi_min),
                                   jnp.abs(delta[1])) > config.em_acceleration_floor
            def attempt(state):
                trial, current_step, accepted, result = state
                proposed = old-2*current_step[None]*delta+current_step[None]**2*curvature
                prob = jnp.clip(proposed[0], jnp.maximum(config.pi_min, .8*old[0]),
                                 jnp.minimum(1., 1.25*old[0]))
                logd = jnp.maximum(jnp.clip(proposed[1], old[1]+jnp.log(.8), old[1]+jnp.log(1.25)),
                                    jnp.log(prob*config.slab_variance_min))
                polished, _ = free_em_map(r, gam, jnp.stack([prob, logd]))
                same_direction = jnp.all((polished-old)*delta >= 0, axis=0)
                better = (free_evidence_at(r, gam, polished) >= baseline-1e-10) & same_direction & resolved
                result = jnp.where((better & ~accepted)[None], polished, result)
                return trial+1, -jnp.maximum(jnp.abs(current_step)*.5, 1.), accepted | better | (jnp.abs(current_step) <= 1.), result
            state = (0, step, ~resolved, two)
            return jax.lax.while_loop(lambda s: (s[0] < 42) & jnp.any(~s[2]), attempt, state)[3]
        optimum = jax.lax.fori_loop(0, config.em_inner_steps, body, params)
        return params+config.em_damping*(optimum-params)

    @jax.jit
    def free_nonlinear(r, gam, prob, slab_variance):
        mean, var, pip, mu, s2, evidence = spike_slab_denoise(
            r, gam[None, :], prob[group_ids], slab_variance[group_ids])
        divergence = gam*jnp.sum(var, axis=0)/p_active
        new_pi, new_slab = unconstrained_ss_m_step(reduce_group(pip), reduce_group(pip*(s2+mu*mu)),
            counts, config.pi_min, config.slab_variance_min)
        pi_error = jnp.max(jnp.abs(new_pi-prob)/jnp.maximum(prob, config.pi_min), axis=0)
        slab_error = jnp.max(jnp.abs(new_slab-slab_variance)/jnp.maximum(slab_variance, config.slab_variance_min), axis=0)
        return mean, var, pip, divergence, jnp.maximum(pi_error, slab_error)

    @jax.jit
    def relative_norm(a, b):
        scale = jnp.maximum(jnp.sum(a*a, axis=0), jnp.sum(b*b, axis=0))
        return jnp.sqrt(jnp.sum((a-b)**2, axis=0)/jnp.maximum(scale, 1e-30))

    trace = []
    status = ['max_iterations']*batch
    for iteration in range(offset+1, offset+config.max_iterations+1):
        old_pi, old_slab = pi, slab
        if config.learn_pi and iteration > config.em_warmup:
            if config.variance_constraint:
                learned_pi = learn_prior(r1, gamma1, pi)
                pi = jnp.where(jax.device_put(done)[None, :], old_pi, learned_pi)
                slab = jnp.where(d > 0, d/jnp.maximum(pi, config.pi_min), 0.)
            else:
                learned = learn_free_prior(r1, gamma1, jnp.stack([pi, jnp.log(pi*slab)]))
                frozen_case = jax.device_put(done)[None, :]
                pi = jnp.where(frozen_case, old_pi, learned[0])
                learned_slab = jnp.maximum(jnp.exp(learned[1])/jnp.maximum(learned[0], config.pi_min), config.slab_variance_min)
                slab = jnp.where(frozen_case, old_slab, learned_slab)
        if config.variance_constraint:
            m1, var1, pip, alpha1, em_error = nonlinear(r1, gamma1, pi)
        else:
            m1, var1, pip, alpha1, em_error = free_nonlinear(r1, gamma1, pi, slab)
        alpha_host = np.asarray(alpha1)
        if np.any(~np.isfinite(alpha_host)) or np.any(alpha_host <= 0) or np.any(alpha_host >= 1):
            raise ArithmeticError(f'Invalid nonlinear Onsager divergence at {iteration}: {alpha_host}')
        gamma2 = gamma1*(1-alpha1)/alpha1
        r2 = jnp.where(active, (m1-alpha1[None, :]*r1)/(1-alpha1)[None, :], 0.)
        rhs = y-products.forward(r2)
        solved, trace_fraction = spectrum.solve_and_trace(rhs, gamma2*noise)
        correction = jnp.where(active, products.transpose(solved), 0.)
        m2 = r2+correction
        # Avoid subtracting nearly equal quantities when p >> n.
        gamma_proposed = gamma2*trace_fraction/(1-trace_fraction)
        r_proposed = r2+correction/trace_fraction[None, :]
        message_error = relative_norm(r_proposed, r1)
        module_error = relative_norm(m2, m1)
        precision_error = jnp.abs(gamma_proposed-gamma1)/gamma1
        pi_change = jnp.max(jnp.where(d > 0, jnp.abs(pi-old_pi)/jnp.maximum(old_pi, config.pi_min), 0.), axis=0)
        slab_change = jnp.max(jnp.abs(slab-old_slab)/jnp.maximum(old_slab, config.slab_variance_min), axis=0)
        errors = np.asarray(jnp.stack([message_error, module_error, precision_error, em_error, pi_change, slab_change]))
        if not np.all(np.isfinite(errors)):
            raise ArithmeticError(f'Nonfinite fixed-point diagnostics at iteration {iteration}')
        eligible = (iteration >= config.min_iterations and
                    (not config.learn_pi or iteration > config.em_warmup))
        good = np.max(errors[:3], axis=0) < config.tolerance
        if config.learn_pi:
            good &= (errors[3] < config.em_tolerance) & (errors[4] < config.em_tolerance)
            if not config.variance_constraint:
                good &= errors[5] < config.em_tolerance
        stable = np.where(good & eligible, stable+1, 0)
        done |= stable >= config.consecutive_converged
        # Freeze finished cases in a batch to retain their actual fixed point.
        damp = jnp.where(jax.device_put(done), 0., config.damping)
        gamma_next = (1-damp)*gamma1+damp*gamma_proposed
        r1 = ((1-damp)[None, :]*gamma1[None, :]*r1
              +damp[None, :]*gamma_proposed[None, :]*r_proposed)/gamma_next[None, :]
        gamma1 = gamma_next
        if config.variance_constraint:
            slab = jnp.where(d > 0, d/jnp.maximum(pi, config.pi_min), 0.)
            constraint = np.asarray(jnp.where(d > 0, jnp.abs(pi*slab-d)/jnp.maximum(d, 1e-300), 0.))
            if constraint.max() > 1e-12:
                raise ArithmeticError('Prior variance equality violated')
        if iteration == 1 or iteration % config.log_every == 0 or done.all():
            row = dict(iteration=iteration, seconds=time.monotonic()-start, pi=np.asarray(pi).T.tolist(),
                       gamma1=np.asarray(gamma1).tolist(), alpha1=alpha_host.tolist(),
                       message_error=errors[0].tolist(), module_error=errors[1].tolist(),
                       precision_error=errors[2].tolist(), em_error=errors[3].tolist(),
                       pi_change=errors[4].tolist(), converged=done.tolist(),
                       slab_variance=np.asarray(slab).T.tolist(), slab_change=errors[5].tolist(),
                       prior_variance_components=np.asarray(pi*slab*norm_host[:, None]).T.tolist(),
                       prior_constraint_max_relative_error=float(constraint.max()) if config.variance_constraint else None)
            trace.append(row)
            if callback:
                callback(row)
        if done.all():
            break
    if config.variance_constraint:
        mean, variance, pip, _, _ = nonlinear(r1, gamma1, pi)
    else:
        mean, variance, pip, _, _ = free_nonlinear(r1, gamma1, pi, slab)
    status = ['converged' if flag else 'max_iterations' for flag in done]
    return VAMPResult(np.asarray(mean), np.asarray(variance), np.asarray(pip), np.asarray(pi),
                      np.asarray(slab),
                      status, trace, time.monotonic()-start, iteration, np.asarray(gamma1), initialization,
                      dict(r1=np.asarray(r1), gamma1=np.asarray(gamma1), pi=np.asarray(pi),
                           stable=stable, done=done, iterations=iteration))
