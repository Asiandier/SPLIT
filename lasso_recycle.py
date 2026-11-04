"""Bounded-memory subspace reuse for one fixed SPD Lasso covariance."""
import numpy as np
import jax.numpy as jnp


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
        self.dependent = self.memory_skips = 0

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
        if self.rank == self.capacity:
            capacity = min(self.n, max(1, 2*self.capacity))
            device = next(iter(value.devices()))
            stats = device.memory_stats()
            if stats is not None:
                limit = stats['bytes_limit']
                if self.memory_limit is not None and self.memory_limit > 0:
                    limit = min(limit, self.memory_limit)
                # New FP64/FP32 bases plus allocation/copy intermediates.
                need = int(stats['bytes_in_use'])+48*self.n*capacity+self.workspace_bytes
                if need > limit:
                    self.memory_skips += 1
                    return
            u = jnp.zeros((self.n, capacity), dtype=jnp.float64)
            w = jnp.zeros_like(u)
            if self.rank:
                u = u.at[:, :self.capacity].set(self.u)
                w = w.at[:, :self.capacity].set(self.w)
            self.u, self.w, self.capacity = u, w, capacity
        factor = np.sqrt(energy)
        self.u = self.u.at[:, self.rank].set(v[:, 0]/factor)
        self.w = self.w.at[:, self.rank].set(av[:, 0]/factor)
        self.u32, self.w32 = self.u.astype(jnp.float32), self.w.astype(jnp.float32)
        self.rank += 1
