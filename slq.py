"""Consistent matrix-free SLQ values and parameter derivatives.

Only the small recurrence is differentiated. Streamed matrix products stay
outside JAX transformations; their pullback reuses streamed component products
for both vector and parameter derivatives. No genotype tape is retained.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def default_workspace_bytes(gpu_budget_bytes=None):
    """Shared SLQ budget policy for fitting and memory planning."""
    return (
        min(2 * 1024**3, max(1, int(gpu_budget_bytes / 8)))
        if gpu_budget_bytes is not None else 256 * 1024**2
    )


def workspace_layout(n, nsamples, m, workspace_bytes, *, itemsize=4):
    """Estimate the exact batching policy and retained/reverse workspaces.

    One probe is the minimum batch, even when its workspace exceeds the
    requested budget. The planner must account for that minimum explicitly.
    """
    if n < 1 or nsamples < 1 or m < 1 or workspace_bytes < 1:
        raise ValueError("SLQ dimensions and workspace budget must be positive.")
    depth = min(int(m), int(n))
    per_probe = itemsize*n*(8*(depth+1)+32)
    capacity = max(1, int(workspace_bytes)//per_probe)
    width = min(int(nsamples), 1 << (capacity.bit_length()-1))
    retained = itemsize*n*(nsamples*(2*depth+1) + width*(6*(depth+1)+32))
    eager = per_probe*width
    peak = max(eager, retained) if retained <= workspace_bytes else eager
    # Small dense quadrature matrices are live alongside the Lanczos state.
    peak += 8*itemsize*width*depth*depth
    return dict(batch_width=width, retained_bytes=retained, peak_bytes=peak)


@partial(jax.jit, donate_argnums=(0,))
def _set_row(array, index, value):
    return array.at[index].set(value)


@jax.jit
def _step(basis, product, previous_beta, index):
    """Two-pass orthogonalised Lanczos, with a finite breakdown derivative."""
    q = basis[index]
    previous = jnp.where(index > 0, basis[jnp.maximum(index-1, 0)], 0)
    active = jnp.sum(q*q, axis=0) > 0
    w = product - previous * previous_beta
    alpha = jnp.sum(q*w, axis=0)
    w = w - q*alpha
    # Mask future rows: backward evaluation uses the final basis, not an
    # O(depth^2 * n * probes) collection of basis snapshots.
    prefix = jnp.where((jnp.arange(basis.shape[0]) <= index)[:, None, None], basis, 0)
    for _ in range(2):
        coefficients = jnp.einsum('ins,ns->is', prefix, w, precision=jax.lax.Precision.HIGHEST)
        w = w - jnp.einsum('ins,is->ns', prefix, coefficients,
                           precision=jax.lax.Precision.HIGHEST)
    square = jnp.sum(w*w, axis=0)
    norm = jnp.sqrt(jnp.where(square > 0, square, 1))
    scale = jnp.sqrt(jnp.sum(product*product, axis=0))
    threshold = 16*jnp.finfo(product.dtype).eps*jnp.maximum(scale, jnp.abs(alpha))
    keep = active & (square > threshold*threshold)
    beta = jnp.where(keep, norm, 0)
    next_q = jnp.where(keep[None, :], w/jnp.where(keep, norm, 1), 0)
    # Decoupled padding has log(1)=0; a repeated eigenvalue there must not
    # create a 0/0 in the small-matrix spectral derivative.
    return next_q, jnp.where(active, alpha, 1), beta


@jax.jit
def _step_pullback(basis, product, previous_beta, index, q_bar, alpha_bar, beta_bar):
    _, pullback = jax.vjp(lambda q, w, b: _step(q, w, b, index),
                         basis, product, previous_beta)
    return pullback((q_bar, alpha_bar, beta_bar))


@jax.jit
def _quadrature(alphas, betas, weight):
    """Value and Frechet derivative, including coincident eigenvalues.

    Differentiating eigenvectors individually has artificial singularities at
    repeated eigenvalues. The divided-difference matrix of log does not.
    """
    depth, samples = alphas.shape
    matrix = jnp.zeros((samples, depth, depth), dtype=alphas.dtype)
    idx = jnp.arange(depth)
    matrix = matrix.at[:, idx, idx].set(alphas.T)
    off = jnp.arange(depth-1)
    matrix = matrix.at[:, off+1, off].set(betas.T)
    matrix = matrix.at[:, off, off+1].set(betas.T)
    values, vectors = jnp.linalg.eigh(matrix)
    floor = jnp.asarray(max(float(jnp.finfo(alphas.dtype).eps)*depth, 1e-6), alphas.dtype)
    clipped = jnp.maximum(values, floor)
    log_values = jnp.log(clipped)
    leading = vectors[:, 0, :]
    value = weight*jnp.sum(leading*leading*log_values)
    difference = values[:, :, None]-values[:, None, :]
    scale = jnp.maximum(jnp.abs(values[:, :, None]), jnp.abs(values[:, None, :]))
    close = jnp.abs(difference) <= 8*jnp.finfo(alphas.dtype).eps*jnp.maximum(scale, floor)
    denominator = jnp.where(close, 1, difference)
    # log1p is accurate for clustered eigenvalues, without subtracting logs.
    relative = (clipped[:, :, None]-clipped[:, None, :])/clipped[:, None, :]
    nearby = jnp.abs(relative) < .5
    logarithm_difference = jnp.where(
        nearby, jnp.log1p(jnp.where(nearby, relative, 0)),
        log_values[:, :, None]-log_values[:, None, :],
    )
    midpoint = .5*(values[:, :, None]+values[:, None, :])
    derivative = jnp.where(midpoint > floor, 1/jnp.maximum(midpoint, floor), 0)
    divided = jnp.where(close, derivative, logarithm_difference/denominator)
    spectral = divided * leading[:, :, None] * leading[:, None, :]
    gradient = weight*jnp.matmul(jnp.matmul(vectors, spectral, precision=jax.lax.Precision.HIGHEST),
                                 jnp.swapaxes(vectors, 1, 2), precision=jax.lax.Precision.HIGHEST)
    return value, jnp.diagonal(gradient, axis1=1, axis2=2).T, 2*gradient[:, off, off+1].T


def _batch_pullback(matvec_pullback, basis, products, beta, a_bar, b_bar):
    """Reverse an already computed Lanczos batch without repeating matvecs."""
    q_bar = jnp.zeros_like(basis)
    zero_beta = jnp.zeros(basis.shape[2], dtype=basis.dtype)
    gradient = None
    for index in range(products.shape[0]-1, -1, -1):
        dq, dw, db = _step_pullback(
            basis, products[index], beta[index-1] if index else zero_beta,
            jnp.asarray(index), q_bar[index+1], a_bar[index], b_bar[index],
        )
        q_bar = q_bar + dq
        input_bar, part = matvec_pullback(dw, basis[index])
        # A user callback may promote its output (for example an FP64
        # kernel with an FP32 Krylov state). Keep adjoints on the same
        # declared SLQ precision as the corresponding forward recurrence.
        input_bar = jnp.asarray(input_bar, dtype=q_bar.dtype)
        q_bar = _set_row(q_bar, index, q_bar[index]+input_bar)
        gradient = part if gradient is None else gradient+part
        if index:
            b_bar = b_bar.at[index-1].add(db)
    return gradient


def _log_quadrature_gap(alpha, beta, lower_bound):
    """Gauss / left-Radau log bounds per unit-norm probe, in FP64.

    For log, Gauss is an upper bound and Radau at a positive spectral lower
    bound is a lower bound. This controls quadrature error separately from
    Monte Carlo error; probes and their number are never changed.
    """
    from scipy.linalg import eigh_tridiagonal
    alphas, betas = np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
    largest_gap = 0.
    for a, b in zip(alphas.T, betas.T):
        zeros = np.flatnonzero(b == 0)
        depth = int(zeros[0]+1) if zeros.size else len(a)
        values, vectors = eigh_tridiagonal(a[:depth], b[:depth-1])
        # If the operator violates the supplied bound (including roundoff),
        # fall back to the requested depth rather than accepting a false bound.
        if values[0] <= lower_bound:
            return float('inf')
        if zeros.size:
            continue
        gauss = np.sum(vectors[0]**2*np.log(values))
        last_inverse = np.sum(vectors[-1]**2/(values-lower_bound))
        endpoint = lower_bound+b[-1]**2*last_inverse
        nodes, vec = eigh_tridiagonal(np.r_[a, endpoint], b)
        if nodes[0] <= 0:
            return float('inf')
        radau = np.sum(vec[0]**2*np.log(nodes))
        gap = gauss-radau
        if not np.isfinite(gap) or gap < -1e-10*max(1., abs(gauss)):
            return float('inf')
        largest_gap = max(largest_gap, gap)
    return largest_gap


def logdet_value_and_grad(matvec, matvec_pullback, n, key, *, nsamples, m,
                         workspace_bytes=256*1024**2, dtype=jnp.float32,
                         accept_value=None, lower_bound=None,
                         quadrature_tolerance=1e-6, diagnostics=None):
    """Return a fixed-probe SLQ value and its own parameter gradient.

    ``matvec_pullback(left, right)`` returns ``(matvec(left), gradient)``,
    with gradient[k] = sum(right * D_k(left)). Symmetry allows the caller to
    reuse each D_k(left) in both outputs. Callbacks receive concrete arrays,
    never tracers, and must implement fixed symmetric linear operators. For
    preconditioned SLQ the reference must be frozen; contractions use its
    inverse square root on both sides. Zero variance components are allowed.
    Pass ``None`` as the pullback for a value-only evaluation of the same target.

    An optional ``accept_value(value)`` gates the gradient for line search.
    If all forward states plus reverse workspace fit the budget estimate,
    defer reverse passes until acceptance. Otherwise retain the original
    streamed eager derivative, without extra caching or repeated forward matvecs.

    If a positive spectral ``lower_bound`` is supplied, stop a batch when its
    Gauss/Radau log gap is at most ``quadrature_tolerance`` per sample. ``m``
    remains the maximum depth; tolerance zero requests that fixed depth.
    Reverse exactly the recurrence actually used,
    so the line-search objective and its local derivative remain consistent.
    """
    if lower_bound is not None and (not np.isfinite(lower_bound) or lower_bound <= 0):
        raise ValueError('SLQ spectral lower bound must be finite and positive')
    if not np.isfinite(quadrature_tolerance) or quadrature_tolerance < 0:
        raise ValueError('SLQ quadrature tolerance must be finite and nonnegative')
    depth = min(int(m), int(n))
    # The bounds apply to log(x), not to the clipped spectral extension used
    # by _quadrature near singularity. Preserve fixed-depth behavior unless
    # the whole spectrum is certified above the original clipping floor.
    spectral_floor = max(float(np.finfo(dtype).eps)*depth, 1e-6)
    if lower_bound is not None and lower_bound <= spectral_floor:
        lower_bound = None
    # Q, products, their adjoints and recurrence/VJP workspaces. Batching
    # bounds retained state independently of the total number of probes.
    layout = workspace_layout(n, nsamples, m, workspace_bytes, itemsize=np.dtype(dtype).itemsize)
    width = layout["batch_width"]
    defer_gradient = (
        accept_value is not None and matvec_pullback is not None
        and layout["retained_bytes"] <= workspace_bytes
    )
    pending = []
    keys = jax.random.split(key, nsamples)
    total_value, total_gradient = jnp.asarray(0., dtype=dtype), None
    for start in range(0, nsamples, width):
        count = min(width, nsamples-start)
        signs = jax.vmap(lambda k:jax.random.rademacher(k, (n,), dtype=jnp.int32))(
            keys[start:start+count]).astype(dtype).T
        basis = jnp.zeros((depth+1, n, count), dtype=dtype)
        basis = _set_row(basis, 0, signs/jnp.sqrt(jnp.asarray(n, dtype=dtype)))
        products = None if matvec_pullback is None else jnp.zeros((depth, n, count), dtype=dtype)
        alpha, beta = [], []
        previous_beta = jnp.zeros(count, dtype=dtype)
        gap = None
        for index in range(depth):
            product = jnp.asarray(matvec(basis[index]), dtype=dtype)
            next_q, a, b = _step(basis, product, previous_beta, jnp.asarray(index))
            if products is not None:
                products = _set_row(products, index, product)
            basis = _set_row(basis, index+1, next_q)
            alpha.append(a); beta.append(b)
            previous_beta = b
            if (lower_bound is not None and quadrature_tolerance > 0
                    and 8 <= index+1 < depth and (index+1) % 4 == 0):
                gap = _log_quadrature_gap(jnp.stack(alpha), jnp.stack(beta), lower_bound)
                if gap <= quadrature_tolerance:
                    break
        actual_depth = len(alpha)
        if diagnostics is not None:
            diagnostics.setdefault('batch_depths', []).append(actual_depth)
            diagnostics.setdefault('log_quadrature_gaps', []).append(gap)
        if products is not None:
            products = products[:actual_depth]
        alphas = jnp.stack(alpha)
        betas = jnp.stack(beta[:-1]) if actual_depth > 1 else jnp.empty((0, count), dtype=dtype)
        value, a_bar, b_small = _quadrature(alphas, betas, jnp.asarray(n/nsamples, dtype=dtype))
        total_value = total_value+value
        if matvec_pullback is None:
            del basis
            continue
        b_bar = jnp.concatenate([b_small, jnp.zeros((1, count), dtype=dtype)])
        if defer_gradient:
            pending.append((basis, products, beta, a_bar, b_bar))
        else:
            gradient = _batch_pullback(matvec_pullback, basis, products, beta, a_bar, b_bar)
            total_gradient = gradient if total_gradient is None else total_gradient+gradient
        del basis, products
    if not bool(jnp.isfinite(total_value)):
        raise FloatingPointError("Nonfinite SLQ value.")
    if accept_value is not None and not accept_value(total_value):
        return total_value, None
    for batch in pending:
        gradient = _batch_pullback(matvec_pullback, *batch)
        total_gradient = gradient if total_gradient is None else total_gradient+gradient
    if total_gradient is not None and not bool(jnp.all(jnp.isfinite(total_gradient))):
        raise FloatingPointError("Nonfinite SLQ derivative.")
    return total_value, total_gradient
