"""Resident BED decoding and batched FP64 prediction with training scaling."""
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp


def packed_bed(prefix, p, n):
    path = Path(str(prefix)+'.bed')
    with path.open('rb') as handle:
        if handle.read(3) != b'\x6c\x1b\x01':
            raise ValueError('Expected a SNP-major PLINK BED file')
    if path.stat().st_size != 3+p*((n+3)//4):
        raise ValueError('BED dimensions do not match BIM/FAM')
    return np.memmap(path, mode='r', offset=3, dtype=np.uint8, shape=(p, (n+3)//4))


def _inputs(packed, rows, means, inv):
    rows = np.asarray(rows)
    means, inv = np.asarray(means, np.float32), np.asarray(inv, np.float32)
    p = len(packed)
    if (packed.ndim != 2 or packed.dtype != np.uint8 or p == 0
            or rows.ndim != 1 or len(rows) == 0
            or not np.issubdtype(rows.dtype, np.integer) or rows.min() < 0
            or rows.max() >= packed.shape[1]*4 or len(set(rows.tolist())) != len(rows)
            or means.shape != (p,) or inv.shape != (p,)
            or not np.all(np.isfinite(means)) or not np.all(np.isfinite(inv))):
        raise ValueError('Invalid BED rows or normalization')
    return jax.device_put(rows.astype(np.int32)), means, inv


def decode_numpy(packed, rows, means, inv):
    """Independent host reference; missing genotypes become centered zero."""
    rows = np.asarray(rows, int)
    codes = (packed[:, rows//4] >> (2*(rows % 4))[None, :]) & 3
    x = (np.asarray([2., 0., 1., 0.], np.float32)[codes]
         - np.asarray(means, np.float32)[:, None])*np.asarray(inv, np.float32)[:, None]
    x[codes == 1] = 0.
    return x


def decode_resident(packed, rows, means, inv, chunk_size=1024):
    row, means, inv = _inputs(packed, rows, means, inv)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    p, n, size = len(packed), len(rows), min(chunk_size, len(packed))
    whole = p//size*size

    @jax.jit
    def decode(bed, mu, scaling):
        def block(b, m, iv):
            codes = (b[:, row//4] >> (2*(row % 4))[None, :]) & 3
            centered = jax.lax.optimization_barrier(
                jnp.asarray([2., 0., 1., 0.], jnp.float32)[codes]-m[:, None])
            return jnp.where(codes == 1, 0., centered*iv[:, None])
        def body(i, total):
            b = jax.lax.dynamic_slice(bed, (i*size, 0), (size, bed.shape[1]))
            m = jax.lax.dynamic_slice(mu, (i*size,), (size,))
            iv = jax.lax.dynamic_slice(scaling, (i*size,), (size,))
            return jax.lax.dynamic_update_slice(total, block(b, m, iv), (i*size, 0))
        result = jax.lax.fori_loop(0, whole//size, body, jnp.zeros((p, n), jnp.float32))
        if whole < p:
            result = result.at[whole:].set(block(bed[whole:], mu[whole:], scaling[whole:]))
        return result
    result = decode(jax.device_put(packed), jax.device_put(means), jax.device_put(inv))
    result.block_until_ready()
    return result


class PackedBedPredictor:
    """Reuse resident packed genotypes and one compiled kernel across batches.

    Only a SNP chunk is decoded at a time, and its normalization remains FP32
    to match the training design. Products and accumulators use FP64. Different
    coefficient widths share the resident arrays; JAX caches each width once.
    """
    def __init__(self, packed, rows, means, inv, chunk_size=1024):
        if not jax.config.jax_enable_x64:
            raise RuntimeError('FP64 BED products require jax_enable_x64')
        row, means, inv = _inputs(packed, rows, means, inv)
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('Invalid chunk size')
        self.p = len(packed)
        self.bed = jax.device_put(packed)
        self.means, self.inv = jax.device_put(means), jax.device_put(inv)
        p, n, size = self.p, len(rows), min(chunk_size, self.p)
        whole = p//size*size

        @jax.jit
        def compute(bed, mu, scaling, coef):
            def product(b, m, iv, effects):
                codes = (b[:, row//4] >> (2*(row % 4))[None, :]) & 3
                centered = jax.lax.optimization_barrier(
                    jnp.asarray([2., 0., 1., 0.], jnp.float32)[codes]-m[:, None])
                x = jnp.where(codes == 1, 0., centered*iv[:, None])
                return jnp.matmul(x.astype(jnp.float64).T, effects,
                                  precision=jax.lax.Precision.HIGHEST)
            def body(i, total):
                b = jax.lax.dynamic_slice(bed, (i*size, 0), (size, bed.shape[1]))
                m = jax.lax.dynamic_slice(mu, (i*size,), (size,))
                iv = jax.lax.dynamic_slice(scaling, (i*size,), (size,))
                effects = jax.lax.dynamic_slice(coef, (i*size, 0), (size, coef.shape[1]))
                return total+product(b, m, iv, effects)
            result = jax.lax.fori_loop(0, whole//size, body,
                                      jnp.zeros((n, coef.shape[1]), jnp.float64))
            if whole < p:
                result += product(bed[whole:], mu[whole:], scaling[whole:], coef[whole:])
            return result
        self._compute = compute

    def __call__(self, coefficients):
        if not jax.config.jax_enable_x64:
            raise RuntimeError('FP64 BED products require jax_enable_x64')
        coefficients = np.asarray(coefficients, float)
        if (coefficients.ndim != 2 or coefficients.shape[0] != self.p
                or coefficients.shape[1] == 0 or not np.all(np.isfinite(coefficients))):
            raise ValueError('Invalid coefficients')
        return np.asarray(self._compute(self.bed, self.means, self.inv,
                                        jax.device_put(coefficients)))


def packed_predict(packed, rows, means, inv, coefficients, chunk_size=1024):
    """One-shot FP64 prediction; use PackedBedPredictor for repeated batches."""
    return PackedBedPredictor(packed, rows, means, inv, chunk_size)(coefficients)
