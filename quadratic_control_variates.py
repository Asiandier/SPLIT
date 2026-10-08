"""Control variates for matrix-free projected inverse quadratics.

For a fixed reference h0_j, n_j=||x_j||^2 and independent e~N(0,I),
 h0_j + (x_j'e) * (x_j'V^-1 e - h0_j/n_j * x_j'e)
has expectation x_j'V^-1 x_j. A pilot reference must be independent of the
new probes (or deterministic). This alters numerical precision, not the model.
The same identity holds for any zero-mean probes with identity covariance.
"""
import numpy as np


def quadratic_control_sum(projected_probes, projected_solutions, reference, norms):
    """Sum unbiased controlled quadratic estimates over columns of probes.

    All inputs are for a bounded marker block. No covariance matrix is formed.
    Negative noisy estimates are deliberately retained for external checks.
    """
    probes = np.asarray(projected_probes, dtype=float)
    solved = np.asarray(projected_solutions, dtype=float)
    reference = np.asarray(reference, dtype=float)
    norms = np.asarray(norms, dtype=float)
    if probes.ndim != 2 or solved.shape != probes.shape:
        raise ValueError('Projected probes and solutions must be matching matrices')
    if reference.shape != (len(probes),) or norms.shape != reference.shape:
        raise ValueError('Reference and norms must have one entry per marker')
    if np.any(norms <= 0) or not all(np.all(np.isfinite(a)) for a in (probes, solved, reference, norms)):
        raise ValueError('Inputs must be finite and genotype norms positive')
    center = reference / norms
    return (reference * probes.shape[1]
            + np.sum(probes * (solved - center[:, None] * probes), axis=1))
