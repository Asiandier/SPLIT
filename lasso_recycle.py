"""Bounded-memory subspace reuse for one fixed SPD Lasso covariance."""
import numpy as np
import jax
import jax.numpy as jnp

from .sparse_memory import available_device_bytes, device_allocation_failed


class RecycledInverse:
    """Keep U and AU with U' AU approximately identity, within one operator.

    The coarse correction is U U' r. The balanced inverse preconditioner is
    U U' + (I-U(AU)') M (I-(AU)U'). It is SPD for every SPD M; final acceptance
    always uses the original operator's independently computed FP64 residual.
    """
    def __init__(self, n, workspace_bytes=0, memory_limit=None):
        self.n = int(n)
        self.workspace_bytes = int(workspace_bytes)
        self.memory_limit = memory_limit
        self.rank = 0
        self.capacity = 0
        self.u = self.w = self.u32 = self.w32 = None
        self.dependent = self.memory_skips = self.allocation_failures = 0

    def clear(self):
        had_basis = bool(self.rank)
        self.u = self.w = self.u32 = self.w32 = None
        self.rank = self.capacity = 0
        return had_basis

    def coarse(self, residual):
        if not self.rank:
            return jnp.zeros_like(residual), residual
        weight = self.u.T @ residual
        return self.u @ weight, residual-self.w @ weight

    def preconditioner(self, base):
        if not self.rank:
            return base
        u64, w64, u32, w32 = self.u, self.w, self.u32, self.w32
        def apply(value):
            u, w = (u32, w32) if value.dtype == jnp.float32 else (u64, w64)
            weight = u.T @ value
            fine = value-w @ weight
            fine = base(fine) if base is not None else fine
            return fine+u @ (weight-w.T @ fine)
        return apply

    def observe(self, value, product):
        if value.shape != (self.n, 1) or product.shape != value.shape:
            raise ValueError('Recycling requires one matching solution/operator column')
        original_norm = float(jnp.linalg.norm(value))
        if not original_norm:
            return
        v, av = value, product
        if self.rank:
            # Two-pass modified Gram-Schmidt in the covariance inner product.
            for _ in range(2):
                weight = self.u.T @ av
                v, av = v-self.u @ weight, av-self.w @ weight
        energy = float(jnp.sum(v*av))
        norm = float(jnp.linalg.norm(v))
        if (not np.isfinite(energy) or energy <= 0.
                or norm <= np.sqrt(np.finfo(float).eps)*original_norm):
            self.dependent += 1
            return
        if self.rank == self.n:
            return
        capacity = self.capacity if self.rank < self.capacity else min(self.n, max(1, 2*self.capacity))
        device = next(iter(value.devices()))
        room = available_device_bytes(device, self.memory_limit)
        if room is not None:
            # Old bases are still live. Budget the replacement FP64/FP32
            # copies and the update's temporaries around that live state.
            affordable = max(0, room-self.workspace_bytes)//(48*self.n)
            if affordable < max(self.capacity, self.rank+1):
                self.memory_skips += 1
                return
            capacity = min(capacity, affordable)
        elif device.platform == 'gpu':
            self.memory_skips += 1
            return

        u, w = self.u, self.w
        factor = np.sqrt(energy)
        try:
            if capacity != self.capacity:
                if self.rank:
                    padding = ((0, 0), (0, capacity-self.capacity))
                    u = jnp.pad(u, padding)
                    w = jnp.pad(w, padding)
                else:
                    u = jnp.zeros((self.n, capacity), dtype=jnp.float64)
                    w = jnp.zeros_like(u)
            u = u.at[:, self.rank].set(v[:, 0]/factor)
            w = w.at[:, self.rank].set(av[:, 0]/factor)
            u32, w32 = u.astype(jnp.float32), w.astype(jnp.float32)
            jax.block_until_ready((u, w, u32, w32))
        except (RuntimeError, MemoryError) as error:
            if not device_allocation_failed(error):
                raise
            self.allocation_failures += 1
            return
        self.u, self.w, self.u32, self.w32 = u, w, u32, w32
        self.capacity = capacity
        self.rank += 1
