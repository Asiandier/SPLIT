"""Sparse-workflow cache policy, separate from ordinary REML defaults."""
from __future__ import annotations

from .pipeline_common import run_planner
from .suggest_params_v3 import _DEFAULT_GPU_FREE, _GPU_HEADROOM


def sparse_packed_device_cache_budget(gpu_budget_bytes=None) -> int:
    """Allow a full packed source to use up to half the sparse GPU budget.

    The shared planner reserves this allowance before sizing compute blocks.
    Native REML fitting and subsequent background refinement reuse the same
    allowance; they do not retain two copies of the packed source.
    """
    return (max(0, int(gpu_budget_bytes)//2)
            if gpu_budget_bytes is not None else 256 * 1024**2)


def run_sparse_planner(*, gpu_free=None, gpu_budget=None, **kwargs):
    """Opt the sparse workflow into its cache policy without changing REML."""
    resolved_budget = (
        float(gpu_budget)
        if gpu_budget is not None
        else float(gpu_free if gpu_free is not None else _DEFAULT_GPU_FREE) * _GPU_HEADROOM
    )
    return run_planner(
        gpu_free=gpu_free,
        gpu_budget=gpu_budget,
        packed_device_cache_bytes=sparse_packed_device_cache_budget(resolved_budget),
        **kwargs,
    )
