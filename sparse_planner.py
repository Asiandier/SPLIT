"""Sparse-workflow cache policy, separate from ordinary REML defaults."""
from __future__ import annotations

from .pipeline_common import run_planner
from .slq import workspace_layout
from .suggest_params_v3 import _DEFAULT_GPU_FREE, _GPU_HEADROOM


def sparse_packed_device_cache_budget(gpu_budget_bytes=None) -> int:
    """Allow a full packed source to use up to half the sparse GPU budget.

    The shared planner reserves this allowance before sizing compute blocks.
    Native REML fitting and subsequent background refinement reuse the same
    allowance; they do not retain two copies of the packed source.
    """
    return (max(0, int(gpu_budget_bytes)//2)
            if gpu_budget_bytes is not None else 256 * 1024**2)


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
    occupied = plan.gpu_fixed_gib * 1024**3 + 80*n*r + n*w/2 + 8*n*w
    return min(cap, max(0, int(gpu_budget_bytes-occupied)))


def sparse_solve_peak_bytes(plan, n_samples, rhs_batch_size, workspace_bytes):
    """Include selected-mean solves absent from the ordinary REML planner.

    Account for live FP32 Krylov vectors plus FP64 residual/correction vectors
    (80 bytes per sample/RHS). Native FP64 can retain two decoded blocks;
    emulated FP64 needs one decoded block and its explicit workspace. Packed
    prefetch adds two quarter-byte blocks. These are alternative live phases.
    """
    n, w, r = int(n_samples), int(plan.call_width), int(rhs_batch_size)
    fixed = plan.gpu_fixed_gib * 1024**3
    vectors = 80*n*r
    prefetch = n*w/2
    return fixed+vectors+prefetch+max(16*n*w, 8*n*w+int(workspace_bytes))


def run_sparse_planner(*, gpu_free=None, gpu_budget=None, rhs_batch_size=1024,
                       device="gpu", **kwargs):
    """Opt the sparse workflow into its cache policy without changing REML."""
    resolved_budget = (
        float(gpu_budget)
        if gpu_budget is not None
        else float(gpu_free if gpu_free is not None else _DEFAULT_GPU_FREE) * _GPU_HEADROOM
    )
    kwargs.setdefault('slq_workspace_bytes', sparse_slq_workspace_bytes(resolved_budget))
    plan = run_planner(
        gpu_free=gpu_free,
        gpu_budget=gpu_budget,
        packed_device_cache_bytes=sparse_packed_device_cache_budget(resolved_budget),
        **kwargs,
    )
    # CPU fitting has no live GPU solve or FFI workspace. Retain its previous
    # plan rather than rejecting it against an unrelated device-memory limit.
    if str(device).lower() == "cpu":
        return plan
    def peak(p):
        # Size blocks for the mandatory native FP64 path first. Optional
        # emulation scratch must fit around these arrays, not displace them.
        return sparse_solve_peak_bytes(p, kwargs['n_samples'], rhs_batch_size, 0)
    # Keep ordinary-REML planning intact. Re-plan only the sparse block width
    # when its selected-mean solve has the larger live-memory requirement.
    if peak(plan) > resolved_budget and plan.call_width > 256:
        lo, hi = 1, plan.call_width//256
        best = None
        while lo <= hi:
            mid = (lo+hi)//2
            candidate = run_planner(
                gpu_free=gpu_free, gpu_budget=gpu_budget,
                packed_device_cache_bytes=sparse_packed_device_cache_budget(resolved_budget),
                **dict(kwargs, requested_call_width=mid*256),
            )
            if peak(candidate) <= resolved_budget:
                best = candidate
                lo = mid+1
            else:
                hi = mid-1
        if best is not None:
            plan = best
    workspace = sparse_fp64_workspace_bytes(
        resolved_budget, plan=plan, n_samples=kwargs['n_samples'],
        rhs_batch_size=rhs_batch_size,
    )
    estimate = sparse_solve_peak_bytes(plan, kwargs['n_samples'], rhs_batch_size, workspace)
    plan.feasible = bool(plan.feasible and estimate <= resolved_budget)
    plan.gpu_peak_gib = max(plan.gpu_peak_gib, estimate/1024**3)
    plan.note += (f" Sparse selected-mean solve peak={estimate/1024**3:.1f}GiB; "
                  f"RHS batch={rhs_batch_size}; FP64 workspace<={workspace/1024**3:.1f}GiB.")
    if not plan.feasible:
        raise SystemExit("Sparse selected-mean solve exceeds the GPU budget; "
                         "reduce --candidate-pcg-rhs-batch-size. " + plan.note)
    return plan
