"""Optional packed genotype products for narrow sparse-workflow RHS blocks.

Deterministic staged reductions avoid materializing a full floating-point
genotype block. Unsupported CUDA build environments retain native JAX.
"""
from __future__ import annotations

import ctypes
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_checked = False
_library = None


def initialize() -> bool:
    global _checked, _library
    with _lock:
        if _checked:
            return _library is not None
        _checked = True
        try:
            from jax import ffi
            nvcc = shutil.which("nvcc")
            if nvcc is None:
                candidates = sorted(Path("/usr/local").glob("cuda*/bin/nvcc"))
                nvcc = str(candidates[-1]) if candidates else None
            if nvcc is None:
                raise RuntimeError("CUDA compiler is not available")
            source = Path(__file__).with_suffix(".cu")
            digest = hashlib.sha256(source.read_bytes() + jax.__version__.encode()
                                    + str(nvcc).encode()).hexdigest()[:20]
            cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"gpu_reml/ffi"
            cache.mkdir(parents=True, exist_ok=True)
            target = cache/f"sparse_packed_{digest}.so"
            if not target.exists():
                temporary = target.with_suffix(f".{os.getpid()}.so")
                command = [nvcc, "-std=c++17", "-O3", "--shared", "-Xcompiler", "-fPIC",
                           "-arch=compute_70", "-code=compute_70", "-I"+ffi.include_dir(),
                           str(source), "-o", str(temporary)]
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            bridge = ctypes.CDLL(str(target))
            for name, symbol in [("split_sparse_packed_f32", "PackedFloat"),
                                 ("split_sparse_packed_f64", "PackedDouble")]:
                ffi.register_ffi_target(name, ffi.pycapsule(getattr(bridge, symbol)), platform="CUDA")
            # Exercise the actual device/driver before a long-running fit.
            with jax.experimental.enable_x64():
                for dtype in (jnp.float32, jnp.float64):
                    value = gram_product(jnp.asarray([[0x24], [0xff]], dtype=jnp.uint8),
                        jnp.ones(4, dtype=jnp.float32), jnp.ones(4, dtype=jnp.float32),
                        jnp.asarray([[1.], [2.]], dtype=dtype), 4, 3)
                    if not np.array_equal(np.asarray(value), np.asarray([[3.], [0.]])):
                        raise RuntimeError("Packed genotype backend setup check failed")
            _library = bridge
            logger.info("Sparse narrow genotype products: deterministic packed CUDA")
        except (ImportError, OSError, RuntimeError, AttributeError, ValueError,
                TypeError, subprocess.CalledProcessError) as exc:
            _library = None
            reason = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                reason += ": " + (exc.stderr or "")[-1000:]
            logger.info("Sparse narrow genotype products: native JAX (%s)", reason)
        return _library is not None


def supports(value) -> bool:
    # Measured crossover for staged reductions versus matrix-matrix products;
    # these bounds choose an implementation, not a statistical approximation.
    return (_library is not None
            and ((value.dtype == jnp.float32 and value.shape[1] <= 5)
                 or (value.dtype == jnp.float64 and value.shape[1] <= 12)))


def _centered_product(packed, means, value, true_width, missing, *, transpose):
    from jax import ffi
    n, width = packed.shape[0], packed.shape[1]*4
    shape = (((n+511)//512, width, value.shape[1]) if transpose
             else ((packed.shape[1]+511)//512, n, value.shape[1]))
    target = "split_sparse_packed_f64" if value.dtype == jnp.float64 else "split_sparse_packed_f32"
    partial = ffi.ffi_call(target, jax.ShapeDtypeStruct(shape, value.dtype),
        input_layouts=[(0, 1), (0,), (), (), (0, 1)], output_layouts=(0, 1, 2),
        vmap_method="sequential")(
            packed, means, jnp.asarray(true_width, dtype=jnp.int32),
            jnp.asarray(missing, dtype=jnp.uint8), value, transpose=bool(transpose),
        )
    return jnp.sum(partial, axis=0)


def transpose_product(packed, means, inverse, value, true_width, missing):
    inner = _centered_product(packed, means, value, true_width, missing, transpose=True)
    width = packed.shape[1]*4
    scale = jnp.where(jnp.arange(width) < true_width, inverse[:width], 0).astype(value.dtype)
    return scale[:, None]*inner


def gram_product(packed, means, inverse, value, true_width, missing):
    inner = _centered_product(packed, means, value, true_width, missing, transpose=True)
    width = packed.shape[1]*4
    inverse = jnp.where(jnp.arange(width) < true_width, inverse[:width], 0).astype(value.dtype)
    return _centered_product(packed, means, (inverse*inverse)[:, None]*inner,
                             true_width, missing, transpose=False)
