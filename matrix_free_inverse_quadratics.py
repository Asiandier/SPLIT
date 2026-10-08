"""Reproducible inverse quadratics with independent probes and no GRM.

For V = X diag(c) X' + noise I, estimate h_j = x_j' V^-1 x_j.
A fixed reference reduces variance without changing this expectation. It must
be independent of these probes. Exact audit solves do not replace estimates.
The returned, unrefined prefix can be extended without recomputing its probes.
"""
from __future__ import annotations

import time
import numpy as np
from .quadratic_control_variates import quadratic_control_sum


def estimate_inverse_quadratics(x, backend, variance, noise, norms, groups, reference, *,
                                design_identity, draws, block_draws=512, batch_draws=64,
                                audit_per_group=32, seed=2026100602, previous=None,
                                median_error=.01, q95_error=.03, progress=None):
    """Return statistics and a numerical audit, retaining failed estimates.

    x has SNPs on rows. Reuse is allowed only for identical model, design,
    reference and probe definitions, before any exact-marker refinement.
    Audit failure is explicit; a caller may extend the draw budget or fail.
    """
    began = time.monotonic()
    variance = np.asarray(variance, float); norms = np.asarray(norms, float)
    groups = np.asarray(groups); reference = np.asarray(reference, float)
    if (len(x.shape) != 2 or any(a.shape != (len(x),) for a in [variance,norms,groups,reference])
            or not np.issubdtype(groups.dtype, np.integer) or np.any(groups < 0)
            or not all(np.isfinite(a).all() for a in [variance,norms,reference])
            or np.any(variance < 0) or np.any(norms < 0) or np.any(reference < 0)
            or not np.isfinite(noise) or noise <= 0 or not isinstance(design_identity, str) or not design_identity
            or not all(isinstance(a, (int,np.integer)) and a > 0 for a in [draws,block_draws,batch_draws,audit_per_group])
            or block_draws % batch_draws or draws % block_draws or (draws//block_draws) % 2
            or draws < 2*block_draws or not isinstance(seed, (int,np.integer)) or seed < 0
            or not np.isfinite(median_error) or not np.isfinite(q95_error) or min(median_error,q95_error) <= 0):
        raise ValueError('Aligned model arrays, positive noise and balanced probe blocks are required')
    use = norms > 0
    if not np.any(use) or np.any(reference[~use] != 0):
        raise ValueError('A nonempty design and zero reference for zero-norm SNPs are required')
    rng = np.random.default_rng(seed+3)
    audit = np.sort(np.concatenate([rng.choice(np.flatnonzero(use & (groups==g)),
        min(audit_per_group, np.count_nonzero(use & (groups==g))), replace=False)
        for g in np.unique(groups[use])]))
    identity = dict(prior_variance=variance, norm_squared=norms, groups=groups, control_reference=reference,
        audit_indices=audit, residual_variance=np.array(noise), design_identity=np.array(design_identity),
        seed=np.array(seed), block_draws=np.array(block_draws), batch_draws=np.array(batch_draws))
    blocks = []; reused = 0; maximum_residual = 0.; solve_count = 0

    def solve(rhs):
        nonlocal maximum_residual, solve_count
        answer, record = backend.solve_precise(rhs, variance, noise)
        error = record.get('maximum_relative_residual', np.inf)
        if not record.get('original_system_verified', False) or not np.isfinite(error) or error > 1e-10:
            raise ArithmeticError('Inverse-quadratic covariance solve failed its original-system residual gate')
        maximum_residual = max(maximum_residual, float(error)); solve_count += 1
        return answer

    if previous is not None:
        for key, value in identity.items():
            if key not in previous or not np.array_equal(previous[key], value):
                raise ValueError('Inverse-quadratic prefix inputs changed: '+key)
        old = np.asarray(previous['h_blocks'], float)
        if (old.ndim != 2 or old.shape[1] != len(x) or len(old) % 2
                or len(old)*block_draws > draws or not np.isfinite(old).all()
                or len(previous.get('exact_indices', []))):
            raise ValueError('Only an unrefined, finite, balanced prefix can be reused')
        exact = np.asarray(previous['exact_h'], float)
        if exact.shape != audit.shape or not np.isfinite(exact).all() or np.any(exact <= 0):
            raise ValueError('The prefix must include its aligned positive exact audit')
        blocks = list(old.copy()); reused = len(blocks)*block_draws
    else:
        rhs = np.asarray(x[audit], float).T
        answer = solve(rhs)
        exact = np.sum(rhs*answer, axis=0)
        if not np.isfinite(exact).all() or np.any(exact <= 0):
            raise ArithmeticError('Exact inverse-quadratic audit is nonpositive')
    for block in range(len(blocks), draws//block_draws):
        total = np.zeros(len(x))
        for offset in range(0, block_draws, batch_draws):
            rng = np.random.default_rng(np.random.SeedSequence([seed,x.shape[1],block,offset]))
            probes = rng.standard_normal((x.shape[1],batch_draws), dtype=np.float32).astype(float)
            answer = solve(probes)
            projected = np.asarray(backend.scores(probes), float)
            solved = np.asarray(backend.scores(answer), float)
            total[use] += quadratic_control_sum(projected[use], solved[use], reference[use], norms[use])
            if progress:
                progress(dict(draws=block*block_draws+offset+batch_draws, total=draws))
        blocks.append(total/block_draws)
    blocks = np.asarray(blocks)
    h = blocks.mean(axis=0); error = abs(h[audit]/exact-1)
    accepted = bool(np.all(h[use] > 0) and np.all(variance*h < 1)
        and np.median(error) <= median_error and np.quantile(error,.95) <= q95_error)
    data = dict(identity, h=h, h_blocks=blocks, exact_h=exact, exact_indices=np.array([],int))
    record = dict(numeric_accepted=accepted, draws=draws, reused_draws=reused,
        newly_computed_draws=draws-reused, median_relative_error=float(np.median(error)),
        q95_relative_error=float(np.quantile(error,.95)), maximum_solve_relative_residual=maximum_residual,
        new_solve_calls=solve_count, seconds=time.monotonic()-began, explicit_grm=False,
        audit_estimates_replaced=False, reference_must_be_independent_of_probes=True)
    return data, record
