"""Fixed-budget variance weights from a BLUP point estimate.

The exponent controls concentration, not a posterior probability. Applying
this rule repeatedly and normalizing group traces is an empirical prediction
procedure, not guaranteed EM/ARD ascent. No inverse quadratics are required.
"""
from __future__ import annotations

import numpy as np
def allocate_group_trace(score, variance, norm_squared, groups):
    """d_j = c_j a_j B_g / sum_{k in g}(c_k a_k ||x_k||^2).

    B_g = sum_{k in g}(c_k ||x_k||^2). This reduces to variances
    proportional to a_j when c is constant within a group, as in COHERIT-LD4.
    An uninformative all-zero score group retains its original c; inactive
    c=0 coordinates stay exactly zero. Norms must use the same fitted design.
    """
    score, variance, norm_squared = (np.asarray(a, float) for a in (score, variance, norm_squared))
    groups = np.asarray(groups)
    if (score.ndim != 1 or not len(score)
            or any(a.shape != score.shape for a in (variance, norm_squared, groups))
            or not np.issubdtype(groups.dtype, np.integer) or np.any(groups < 0)
            or any(not np.isfinite(a).all() or np.any(a < 0) for a in (score, variance, norm_squared))):
        raise ValueError('Aligned nonnegative scores, variances, norms and integer groups required')
    out = np.zeros_like(variance)
    records = []
    for g in np.unique(groups):
        ids = np.flatnonzero(groups == g)
        c, s, n = variance[ids], score[ids], norm_squared[ids]
        budget = float(c@n)
        maximum = float(np.max(s[c > 0])) if np.any(c > 0) else 0.
        denominator = float((c*(s/maximum))@n) if maximum > 0 else 0.
        fallback = denominator == 0
        out[ids] = c if fallback else c*(s/maximum)*(budget/denominator)
        assigned = float(out[ids]@n)
        relative = abs(assigned-budget)/budget if budget > 0 else abs(assigned)
        if not np.isfinite(out[ids]).all() or relative > 5e-13:
            raise ArithmeticError('Group covariance trace not preserved')
        records.append(dict(group=int(g),original_trace=budget,assigned_trace=assigned,
                            relative_trace_error=relative,zero_score_fallback=bool(fallback)))
    return out, records



def effect_power_variance(effect, variance, norm_squared, groups, power=2.):
    """Allocate c_j*|effect_j|**power within original group trace budgets.

    Scale within each group before exponentiation, so the result is invariant
    to each group's effect units and does not overflow on finite large pilots.
    An all-zero pilot group retains c. Zero original variance stays zero.
    """
    effect, variance, norms = (np.asarray(a, float) for a in
                               (effect, variance, norm_squared))
    groups = np.asarray(groups)
    if (effect.ndim != 1 or not len(effect)
            or any(a.shape != effect.shape for a in (variance, norms, groups))
            or any(not np.isfinite(a).all() for a in (effect, variance, norms))
            or np.any(variance < 0) or np.any(norms < 0)
            or not np.issubdtype(groups.dtype, np.integer) or np.any(groups < 0)
            or isinstance(power, (bool, np.bool_)) or not np.isscalar(power)
            or not np.isfinite(power) or power <= 0
            or np.any(effect[variance == 0] != 0)):
        raise ValueError('Finite aligned BLUP inputs and a positive exponent required')
    score = np.zeros_like(effect)
    for group in np.unique(groups):
        use = (groups == group) & (variance > 0) & (norms > 0)
        if not np.any(use):
            continue
        magnitude = np.abs(effect[use])
        maximum = float(magnitude.max())
        if maximum > 0:
            score[use] = (magnitude / maximum)**power
    d, trace = allocate_group_trace(score, variance, norms, groups)
    active = variance > 0
    ratio = d[active]/variance[active]
    return d, dict(power=float(power), groups=trace,
                   variance_ratio_quantiles=np.quantile(ratio, [0., .5, .9, .99, 1.]).tolist()
                   if len(ratio) else [0.]*5,
                   zero_variance_snps=int(np.count_nonzero(d == 0)),
                   uses_inverse_quadratics=False, posterior_probability=False)
