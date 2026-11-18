"""Sparse-only, FP64 selected-span factorization with bounded GPU scratch.

QR supplies the orthonormal span. A verified inverse of its triangular factor
can certify full column rank without computing singular vectors. Difficult
or deficient ranks retain the original QR/SVD test and its ambiguity guard.
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

logger = logging.getLogger(__name__)


def gpu_workspace_bytes(device, budget_bytes):
    """Respect the fit budget and currently available allocator/physical RAM."""
    if device is None or device.platform != 'gpu' or budget_bytes is None:
        return 0
    stats = device.memory_stats() or {}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip().split(',')
    local = int(getattr(device, 'local_hardware_id', device.id))
    selector = visible[local].strip() if visible != [''] and local < len(visible) else str(local)
    try:
        # Unlike the ordinary planner's advisory query, do not substitute an
        # estimated free-memory fraction when the physical query fails.
        values = subprocess.check_output([
            'nvidia-smi', '--id='+selector, '--query-gpu=memory.total,memory.free',
            '--format=csv,noheader,nounits',
        ], text=True, timeout=5).strip().split(',')
        total, free = [float(v)*1024**2 for v in values]
    except (OSError, subprocess.SubprocessError, ValueError):
        logger.info('Selected GPU basis unavailable: physical free memory could not be verified')
        return 0
    if stats.get('bytes_limit') is not None and stats.get('bytes_in_use') is not None:
        live = stats['bytes_in_use']
        reusable = max(0, stats.get('pool_bytes', live)-live)
        return max(0, int(min(min(budget_bytes, stats['bytes_limit'])-live, free+reusable)))
    return max(0, int(free-max(0, total-budget_bytes)))


def _product(a, b, device, available):
    """Return a host product, tiling rows instead of duplicating a tall array."""
    if device is None or available < b.nbytes + 16*(a.shape[1]+b.shape[1]):
        return a @ b
    from . import sparse_fp64
    accelerated = sparse_fp64.initialize()
    scratch = available//4 if accelerated else 0
    rows = max(1, int((available-b.nbytes-scratch)//(16*(a.shape[1]+b.shape[1]))))
    result = np.empty((a.shape[0], b.shape[1]), dtype=np.float64)
    with jax.default_device(device), jax.experimental.enable_x64():
        right = jax.device_put(np.asarray(b, order='C'), device)
        for start in range(0, a.shape[0], rows):
            left = jax.device_put(np.asarray(a[start:start+rows], order='C'), device)
            value = (sparse_fp64.matmul(left, right, workspace_limit=scratch)
                     if accelerated else left @ right)
            result[start:start+rows] = np.asarray(value)
            del left, value
        del right
    return result


def _qr(x, device, available):
    """Householder QR; a TSQR tree bounds tall-factor workspace by row blocks."""
    n, k = x.shape
    # Input, Householder/orthogonal output and layout/solver copy, plus R and
    # its solver work. Never construct a sample-by-sample square matrix.
    reserve = 24*k*k
    row_limit = int(max(0, available-reserve)//max(1, 24*k))
    if device is None or row_limit < min(n, 2*k):
        return (*sla.qr(x, mode='economic', check_finite=False), 'cpu_householder')
    if n <= row_limit:
        with jax.default_device(device), jax.experimental.enable_x64():
            source = jax.device_put(np.asarray(x, order='C'), device)
            q, r = jnp.linalg.qr(source, mode='reduced')
            q, r = np.asarray(q), np.asarray(r)
        return q, r, 'gpu_householder'
    blocks = math.ceil(n/row_limit)
    edges = np.linspace(0, n, blocks+1, dtype=int)
    q = np.empty_like(x)
    factors = []
    for start, stop in zip(edges[:-1], edges[1:]):
        block_q, block_r, _ = _qr(x[start:stop], device, available)
        q[start:stop] = block_q
        factors.append(block_r)
    reduced_q, r, _ = _qr(np.concatenate(factors), device, available)
    for block, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        q[start:stop] = _product(q[start:stop], reduced_q[block*k:(block+1)*k], device, available)
    return q, r, 'gpu_tsqr'


def _full_rank_inverse(r, n, device, available):
    """Certify sigma_min > 10*n*eps*sigma_max using a residual bound.

If ||I-RW||_2 <= e < 1, sigma_min(R) >= (1-e)/||W||_F.
Frobenius norms bound spectral norms; gamma_k bounds GEMM rounding. A
certificate can fail conservatively, in which case the original SVD decides.
"""
    k = r.shape[0]
    if r.shape != (k, k):
        return None
    try:
        inverse = sla.solve_triangular(r, np.eye(k), check_finite=False)
    except (sla.LinAlgError, ValueError):
        return None
    norm_r, norm_w = np.linalg.norm(r), np.linalg.norm(inverse)
    if not np.isfinite(norm_r*norm_w):
        return None
    residual = _product(r, inverse, device, available)
    residual.flat[::k+1] -= 1.
    eps = np.finfo(np.float64).eps
    gamma = k*eps/(1-k*eps)
    # The extra norm term covers subtraction of the identity and norm rounding.
    error = np.linalg.norm(residual) + gamma*norm_r*norm_w + gamma*math.sqrt(k)
    if error >= 1 or (1-error)/norm_w <= 10*n*eps*norm_r:
        return None
    return inverse


def selected_basis(x, *, hinv=None, device=None, gpu_budget_bytes=None):
    """Build exactly the selected span, retaining the existing rank threshold."""
    available = gpu_workspace_bytes(device, gpu_budget_bytes)
    if not available:
        device = None
    started = time.perf_counter()
    q, r, backend = _qr(x, device, available)
    qr_seconds = time.perf_counter()-started
    started = time.perf_counter()
    inverse = _full_rank_inverse(r, max(x.shape), device, available)
    if inverse is not None:
        # Positive diagonal fixes the otherwise arbitrary QR signs across CPU,
        # GPU and TSQR backends; the projected likelihood is basis invariant.
        signs = np.where(np.diag(r) < 0, -1., 1.)
        # NumPy views copied from a JAX device buffer can be read-only.
        if not q.flags.writeable:
            q = q.copy()
        q *= signs
        inverse *= signs
        basis, rank, method = q, q.shape[1], 'verified_triangular_inverse'
        warm = None if hinv is None else _product(hinv, inverse, device, available).astype(np.float32)
    else:
        u, singular, vt = sla.svd(r, full_matrices=False, check_finite=False)
        cutoff = max(x.shape)*np.finfo(np.float64).eps*singular[0]
        ranks = [int(np.sum(singular > a*cutoff)) for a in (.1, 1., 10.)]
        if len(set(ranks)) != 1:
            raise ValueError('Numerically ambiguous sparse mean-space rank.')
        rank, method = ranks[1], 'svd_fallback'
        basis = _product(q, u[:, :rank], device, available)
        transform = vt[:rank].T/singular[:rank]
        warm = None if hinv is None else _product(hinv, transform, device, available).astype(np.float32)
    logger.info('[selected basis] backend=%s rank_method=%s qr=%.2fs rank=%.2fs',
                backend, method, qr_seconds, time.perf_counter()-started)
    return basis, rank, warm
