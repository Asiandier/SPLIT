"""Live device headroom for optional sparse caches and workspaces."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache


@lru_cache(maxsize=1)
def _cuda_driver():
    try:
        return ct.CDLL("libcuda.so.1")
    except OSError:
        return None


def _cuda_memory(device):
    """Report physical headroom for a CUDA ordinal in the current process."""
    driver = _cuda_driver()
    if driver is None:
        return None
    ordinal = int(getattr(device, "local_hardware_id", device.id))
    selected, current = ct.c_int(), ct.c_int()
    free, total = ct.c_size_t(), ct.c_size_t()
    context = ct.c_void_p()
    retained = pushed = False
    try:
        if driver.cuDeviceGet(ct.byref(selected), ordinal):
            return None
        if driver.cuCtxGetDevice(ct.byref(current)) or current.value != selected.value:
            if driver.cuDevicePrimaryCtxRetain(ct.byref(context), selected):
                return None
            retained = True
            if driver.cuCtxPushCurrent_v2(context):
                return None
            pushed = True
        if driver.cuMemGetInfo_v2(ct.byref(free), ct.byref(total)):
            return None
        return free.value, total.value
    except AttributeError:
        return None
    finally:
        if pushed:
            driver.cuCtxPopCurrent_v2(ct.byref(context))
        if retained:
            driver.cuDevicePrimaryCtxRelease_v2(selected)


def available_device_bytes(device, budget_bytes=None):
    """Return allocator and physical headroom, or None if neither is known.

    The platform allocator has no JAX memory statistics. Under a pooling
    allocator, already reserved free pool storage is reusable.
    """
    stats = device.memory_stats() or {}
    live = stats.get("bytes_in_use")
    limit = stats.get("bytes_limit")
    budget = None if budget_bytes is None or budget_bytes <= 0 else int(budget_bytes)
    memory = _cuda_memory(device) if getattr(device, "platform", None) == "gpu" else None
    if live is not None and limit is not None:
        limit = min(limit, budget) if budget is not None else limit
        headroom = limit-live
        if memory is not None:
            free, _ = memory
            reusable = max(0, stats.get("pool_bytes", live)-live)
            headroom = min(headroom, free+reusable)
        return max(0, int(headroom))
    if memory is not None:
        free, total = memory
        return free if budget is None else max(0, free-max(0, total-budget))
    return None


def device_allocation_failed(error):
    return isinstance(error, MemoryError) or any(
        text in str(error) for text in ("RESOURCE_EXHAUSTED", "OUT_OF_MEMORY"))
