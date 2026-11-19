"""Bounded GPU caching and asynchronous prefetch for sparse packed products."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging

import jax
import numpy as np

from .geno_stream import _pin_buffer, _unpin_buffer
from .sparse_memory import available_device_bytes, device_allocation_failed

logger = logging.getLogger(__name__)

# Two transfers can be staged while the consumer queues computations. Drain
# after one window so device allocations cannot accumulate behind dispatch.
PREFETCH_TRANSFERS = 2
PREFETCH_WINDOW = 2 * PREFETCH_TRANSFERS


class SparsePackedStream:
    """Keep a stable prefix of immutable packed blocks and stream the rest.

    Sequential scans reuse the same prefix. A recency cache would evict that
    prefix on every full pass when the data exceed the budget. Transferred
    blocks and host staging are independent of the persistent cache limit.
    """

    def __init__(self, device, read_block, block_sizes, *, budget_bytes):
        self.device = device
        self.read_block = read_block
        self.block_sizes = tuple(int(size) for size in block_sizes)
        self.budget_bytes = budget_bytes
        self.prefetch_bytes = min(PREFETCH_WINDOW, len(self.block_sizes))*max(self.block_sizes, default=0)
        self._resident = []
        self.resident_bytes = 0
        self._pool = None
        self._staging = []

    @property
    def resident_calls(self):
        return len(self._resident)

    def _open(self):
        if self._pool is not None:
            return
        size = max(self.block_sizes)
        transfers = min(PREFETCH_TRANSFERS, len(self.block_sizes))
        try:
            for _ in range(transfers):
                host = np.empty(size, dtype=np.uint8)
                self._staging.append((host, _pin_buffer(host, self.device)))
            self._pool = ThreadPoolExecutor(max_workers=transfers, thread_name_prefix="split-packed-h2d")
        except BaseException:
            self.close()
            raise

    def _stage(self, call, slot, after=None):
        if after is not None:
            after.block_until_ready()
        source = self.read_block(call)
        host = self._staging[slot][0][:source.size].reshape(source.shape)
        np.copyto(host, source)
        return host

    def set_cache_limit(self, limit_bytes, *, reserve_bytes=0):
        """Trim or grow the prefix using the remaining live device budget."""
        available = available_device_bytes(self.device, self.budget_bytes)
        if available is None:
            limit = 0
        else:
            limit = min(int(limit_bytes), max(0, available + self.resident_bytes
                        - int(reserve_bytes) - self.prefetch_bytes))
        self.trim(limit)
        while self.resident_calls < len(self.block_sizes):
            call = self.resident_calls
            size = self.block_sizes[call]
            if self.resident_bytes + size > limit:
                break
            self._open()
            try:
                block = jax.device_put(self._stage(call, 0), self.device).block_until_ready()
            except (RuntimeError, MemoryError) as exc:
                if not device_allocation_failed(exc):
                    raise
                break
            self._resident.append(block)
            self.resident_bytes += size
        logger.debug("Sparse packed cache: %d/%d blocks, %.3f GiB resident",
                     self.resident_calls, len(self.block_sizes), self.resident_bytes / 2**30)
        return self.resident_bytes

    def trim(self, limit_bytes):
        while self.resident_bytes > limit_bytes and self._resident:
            block = self._resident.pop()
            self.resident_bytes -= block.nbytes
            # Releasing our reference leaves any previously dispatched JAX
            # consumer's reference valid until that consumer completes.
            del block

    @contextmanager
    def borrow_cache(self, reserve_bytes):
        """Use phase-specific slack and return it before the next wide solve."""
        previous = self.resident_bytes
        try:
            if self.budget_bytes is not None:
                self.set_cache_limit(self.budget_bytes, reserve_bytes=reserve_bytes)
            yield
        finally:
            self.trim(previous)

    def block(self, call):
        """Access for one-off products that do not use the pipelined scan."""
        return self._resident[call] if call < self.resident_calls else self.read_block(call)

    def reduce(self, calls, state, step):
        """Apply step in logical order with a bounded device input window."""
        calls = tuple(calls)
        resident = self.resident_calls
        misses = iter(call for call in calls if call >= resident)
        next_call = next(misses, None)
        if next_call is None:
            for call in calls:
                state = step(state, call, self._resident[call])
            return state

        self._open()
        pending = deque()
        consumed = []
        transfers = [None]*len(self._staging)

        def stage_next(slot):
            nonlocal next_call
            if next_call is not None:
                pending.append((slot, self._pool.submit(
                    self._stage, next_call, slot, transfers[slot])))
                next_call = next(misses, None)

        def upload_next():
            slot, work = pending[0]
            block = jax.device_put(work.result(), self.device)
            pending.popleft()
            transfers[slot] = block
            stage_next(slot)
            return block

        upcoming = None
        try:
            for slot in range(len(self._staging)):
                stage_next(slot)
            upcoming = upload_next()
            for call in calls:
                if call < resident:
                    state = step(state, call, self._resident[call])
                    continue
                consumed.append(upcoming)
                # Submit DMA before launching arithmetic on the current block
                # so JAX can overlap the copy stream with the compute stream.
                # Only the host reader waits for DMA before reusing its slot.
                upcoming = upload_next() if pending else None
                state = step(state, call, consumed[-1])
                if len(consumed) == PREFETCH_WINDOW-1:
                    jax.block_until_ready(state)
                    consumed.clear()
                    # The lookahead copy survives in upcoming and overlaps
                    # the last kernel of the window. Retire consumed inputs.
                    transfers[:] = [None]*len(transfers)
            # Keep the window bounded between scans as well as within one.
            if consumed:
                jax.block_until_ready(state)
        except BaseException:
            # Workers must leave both host slots safe for a later retry or
            # close, even if the arithmetic or first transfer failed.
            for _, work in pending:
                if not work.cancel():
                    try:
                        work.result()
                    except Exception:
                        logger.debug("Discarding failed genotype prefetch", exc_info=True)
            for block in transfers:
                if block is not None:
                    try:
                        block.block_until_ready()
                    except Exception:
                        logger.debug("Discarding failed device transfer", exc_info=True)
            if upcoming is not None:
                try:
                    upcoming.block_until_ready()
                except Exception:
                    logger.debug("Discarding failed device lookahead", exc_info=True)
            raise
        return state

    def close(self):
        self.trim(0)
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        for host, pinned in self._staging:
            if pinned:
                _unpin_buffer(host, self.device)
        self._staging.clear()
