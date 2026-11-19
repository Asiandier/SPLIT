"""Budget sparse compute and packed prefetch separately from ordinary REML."""
from __future__ import annotations

from .pipeline_common import run_planner
from .slq import workspace_layout
from .sparse_packed_stream import PREFETCH_WINDOW
from .suggest_params_v3 import _DEFAULT_GPU_FREE, _GPU_HEADROOM

_GIB = 2**30
_WIDTH_ALIGNMENT = 256
# Bound the latency of fetching one block independently of total panel size.
# Smaller budgets shrink this tile further using the live-workspace plan.
_PACKED_TRANSFER_TARGET = 256 * 2**20


def sparse_slq_workspace_bytes(gpu_budget_bytes) -> int:
    """Reserve one quarter for SLQ; the planner budgets compute separately.

    The packed source has its own half-budget allowance. This workspace is
    specific to sparse fitting; ordinary REML retains its existing policy.
    """
    return max(1, int(gpu_budget_bytes)//4)


def projected_slq_workspace(n, nsamples, depth, *, workspace_bytes,
                            available_bytes, basis_bytes):
    """Fit the extra projection state by releasing the SLQ tape per batch.

    available_bytes is the headroom above the ordinary SLQ peak. The
    non-SLQ part of that estimate stays reserved even at smaller batch widths.
    """
    original = workspace_layout(n, nsamples, depth, workspace_bytes)
    if basis_bytes <= available_bytes:
        return workspace_bytes
    per_probe = original["bytes_per_probe"]
    # The smallest budget that preserves the original power-of-two capacity
    # drops deferred gradients whenever the full tape does not fit.
    capacity = 1 << (original["batch_width"] - 1).bit_length()
    while capacity:
        candidate = min(workspace_bytes, capacity * per_probe)
        peak = workspace_layout(n, nsamples, depth, candidate)["peak_bytes"]
        if basis_bytes <= available_bytes + original["peak_bytes"] - peak:
            return candidate
        capacity //= 2
    return None


def sparse_fp64_workspace_bytes(gpu_budget_bytes, *, plan=None, n_samples=None,
                               rhs_batch_size=1024) -> int:
    """Use remaining solve memory, up to the nonconcurrent SLQ allowance.

    A fixed quota can crowd out genotype blocks at large n. The native FP64
    operator already needs space for two decoded blocks; emulation reuses
    that phase's remaining memory for its scratch buffer and tiles as needed.
    """
    cap = max(0, int(gpu_budget_bytes)//4)
    if plan is None:
        return cap
    if n_samples is None:
        raise ValueError("Sample count is required for planned FP64 workspace.")
    n, w, r = int(n_samples), int(plan.call_width), int(rhs_batch_size)
    occupied = plan.gpu_fixed_gib * _GIB + 80*n*r + _prefetch_bytes(n, w) + 8*n*w
    return min(cap, max(0, int(gpu_budget_bytes-occupied)))


def _prefetch_bytes(n, width):
    return PREFETCH_WINDOW * n * ((width+3)//4)


def sparse_solve_peak_bytes(plan, n_samples, rhs_batch_size, workspace_bytes):
    """Include selected-mean solves absent from the ordinary REML planner.

    Account for live FP32 Krylov vectors plus FP64 residual/correction vectors
    (80 bytes per sample/RHS). Native FP64 can retain two decoded blocks;
    emulated FP64 needs one decoded block and its explicit workspace. Packed
    prefetch has a bounded device window. These are alternative live phases.
    """
    n, w, r = int(n_samples), int(plan.call_width), int(rhs_batch_size)
    fixed = plan.gpu_fixed_gib * _GIB
    vectors = 80*n*r
    prefetch = _prefetch_bytes(n, w)
    return fixed+vectors+prefetch+max(16*n*w, 8*n*w+int(workspace_bytes))


def run_sparse_planner(*, gpu_free=None, gpu_budget=None, rhs_batch_size=1024,
                       auto_rhs_batch=False, device="gpu", **kwargs):
    """Size mandatory work first and assign spare memory to packed blocks."""
    available = float(gpu_free if gpu_free is not None else _DEFAULT_GPU_FREE) * _GPU_HEADROOM
    resolved_budget = available
    if gpu_budget is not None:
        resolved_budget = float(gpu_budget)
        if gpu_free is not None and str(device).lower() != "cpu":
            resolved_budget = min(resolved_budget, available)
    kwargs.setdefault('slq_workspace_bytes', sparse_slq_workspace_bytes(resolved_budget))
    if str(device).lower() == "cpu":
        plan = run_planner(gpu_free=gpu_free, gpu_budget=gpu_budget, **kwargs)
        plan.sparse_rhs_batch_size = rhs_batch_size
        return plan

    n = int(kwargs['n_samples'])
    if n < 1:
        raise ValueError("Sparse GPU planner requires at least one sample.")
    cap = kwargs.pop('requested_call_width', None)
    if cap is None:
        cap = max(_WIDTH_ALIGNMENT,
                  4*_PACKED_TRANSFER_TARGET//n//_WIDTH_ALIGNMENT*_WIDTH_ALIGNMENT)

    def make_plan(width):
        return run_planner(gpu_free=gpu_free, gpu_budget=resolved_budget,
                           packed_device_cache_bytes=0, requested_call_width=width, **kwargs)

    def extra_prefetch(plan):
        # Ordinary REML already includes two device input blocks.
        return (min(PREFETCH_WINDOW, plan.n_calls)-min(2, plan.n_calls))*plan.block_bytes

    def peak(plan):
        return max(plan.gpu_peak_gib*_GIB + extra_prefetch(plan),
                   sparse_solve_peak_bytes(plan, n, rhs_batch_size, 0))

    minimum = make_plan(min(cap, _WIDTH_ALIGNMENT))
    # When possible keep a useful transfer tile. Sacrificing nearly all
    # genotype columns for a single wide RHS batch repeats Python/DMA setup
    # in every covariance scan. On tighter budgets the second loop still
    # finds the smallest feasible layout.
    target = make_plan(min(cap, max(_WIDTH_ALIGNMENT, cap//4//_WIDTH_ALIGNMENT*_WIDTH_ALIGNMENT)))
    trial = rhs_batch_size
    while auto_rhs_batch and rhs_batch_size > 1 and target.feasible and peak(target) > resolved_budget:
        rhs_batch_size = max(1, rhs_batch_size//2)
    if peak(target) > resolved_budget:
        rhs_batch_size = trial
    while auto_rhs_batch and rhs_batch_size > 1 and peak(minimum) > resolved_budget:
        rhs_batch_size = max(1, rhs_batch_size//2)
    if not minimum.feasible or peak(minimum) > resolved_budget:
        raise SystemExit("Sparse selected-mean solve exceeds the GPU budget; "
                         "reduce --candidate-pcg-rhs-batch-size or increase the budget. "
                         + minimum.note)

    plan = make_plan(cap)
    if not plan.feasible or peak(plan) > resolved_budget:
        lo, hi = 1, max(1, plan.call_width//_WIDTH_ALIGNMENT)
        plan = minimum
        while lo <= hi:
            mid = (lo+hi)//2
            trial = make_plan(mid*_WIDTH_ALIGNMENT)
            if trial.feasible and peak(trial) <= resolved_budget:
                plan = trial
                lo = mid+1
            else:
                hi = mid-1

    # Emulated FP64 can use the second decode tile as scratch. A larger
    # workspace is optional; prioritize persistent compressed blocks over it.
    mandatory = peak(plan)
    total_markers = sum(kwargs['p_list'])
    cache = min(max(0, int(resolved_budget-mandatory)), n*(total_markers+3*plan.n_calls)//4)
    extra = extra_prefetch(plan) + cache
    plan.packed_device_cache_bytes = cache
    plan.gpu_fixed_gib += cache/_GIB
    plan.gpu_streamer_state_gib += cache/_GIB
    for field in ('gpu_steady_peak_gib', 'gpu_precond_build_peak_gib',
                  'gpu_precompute_peak_gib', 'gpu_projection_peak_gib', 'gpu_slq_peak_gib'):
        setattr(plan, field, getattr(plan, field) + extra/_GIB)
    plan.host_ring_gib = min(PREFETCH_WINDOW//2, plan.n_calls)*plan.block_bytes/_GIB
    plan.host_anon_est_gib = 3+plan.host_ring_gib
    plan.sparse_rhs_batch_size = rhs_batch_size
    workspace = sparse_fp64_workspace_bytes(
        resolved_budget, plan=plan, n_samples=n, rhs_batch_size=rhs_batch_size,
    )
    estimate = sparse_solve_peak_bytes(plan, n, rhs_batch_size, workspace)
    plan.gpu_peak_gib = max((mandatory + cache)/_GIB, estimate/_GIB)
    plan.note += (f" Sparse cache={cache/_GIB:.2f}GiB; "
                  f"device prefetch={PREFETCH_WINDOW} blocks; "
                  f"selected-mean peak={estimate/_GIB:.2f}GiB; "
                  f"RHS batch={rhs_batch_size}; FP64 workspace<={workspace/_GIB:.2f}GiB.")
    return plan
