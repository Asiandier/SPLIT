"""Optional high-throughput sparse GEMMs, with bounded workspace.

cuBLAS >=13.1 is loaded privately, alongside the existing JAX CUDA library.
The small FFI bridge is built once in the user's cache. Native JAX remains
available on CPUs, unsupported installations, and the ordinary REML path.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import shlex
import subprocess
import threading

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_checked = False
_library = None
_reason = "not requested"


def configure(streamers, workspace_bytes):
    """Enable only for explicitly opted-in sparse GPU streamers."""
    gpu_streamers = [s for s in streamers if getattr(s.dev, "platform", None) == "gpu"]
    if not gpu_streamers or int(workspace_bytes) <= 0:
        return False
    with jax.default_device(gpu_streamers[0].dev):
        if not initialize():
            return False
        from . import sparse_packed
        sparse_packed.initialize()
    for streamer in gpu_streamers:
        streamer._sparse_fp64_workspace_bytes = int(workspace_bytes)
    return True


def initialize() -> bool:
    """Register the CUDA FFI lazily; never alter global JAX precision/math."""
    global _checked, _library, _reason
    with _lock:
        if _checked:
            return _library is not None
        _checked = True
        try:
            if os.environ.get("SPLIT_SPARSE_FP64", "auto") == "native":
                raise RuntimeError("native backend explicitly requested")
            if os.environ.get("CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT"):
                raise RuntimeError("fixed-mantissa override is incompatible with accurate sparse solves")
            override = os.environ.get("SPLIT_CUBLAS_LIBRARY")
            if override:
                vendor = Path(override).resolve(strict=True)
            else:
                spec = importlib.util.find_spec("nvidia.cu13")
                if spec is None or not spec.submodule_search_locations:
                    raise RuntimeError("optional nvidia-cublas>=13.1 package is not installed")
                vendor = Path(next(iter(spec.submodule_search_locations)))/"lib/libcublas.so.13"
            from jax import ffi
            source = Path(__file__).with_suffix(".cc")
            digest = hashlib.sha256(source.read_bytes()+jax.__version__.encode()
                                    +str(vendor).encode()).hexdigest()[:20]
            cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"gpu_reml/ffi"
            cache.mkdir(parents=True, exist_ok=True)
            target = cache/f"sparse_fp64_{digest}.so"
            if not target.exists():
                temporary = target.with_suffix(f".{os.getpid()}.so")
                command = shlex.split(os.environ.get("CXX", "c++")) + [
                    "-std=c++17", "-O3", "-shared", "-fPIC",
                    "-I"+ffi.include_dir(), str(source), "-ldl", "-o", str(temporary),
                ]
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            bridge = ctypes.CDLL(str(target))
            init = bridge.split_fp64_init
            init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
            init.restype = ctypes.c_int
            error = ctypes.create_string_buffer(4096)
            if init(os.fsencode(vendor), error, len(error)):
                raise RuntimeError(error.value.decode())
            ffi.register_ffi_target("split_sparse_fp64", ffi.pycapsule(bridge.SplitFp64Matmul), platform="CUDA")
            ffi.register_ffi_target("split_sparse_fp32", ffi.pycapsule(bridge.SplitFp32Matmul), platform="CUDA")
            # Detect driver/library/FFI incompatibility at setup, before a long
            # fit reaches its first high-precision information solve.
            with jax.experimental.enable_x64():
                for dtype in (jnp.float64, jnp.float32):
                    probe = jnp.ones((1, 1), dtype=dtype)
                    value = matmul(probe, probe, workspace_limit=256*1024**2)
                    if float(value[0, 0]) != 1.:
                        raise RuntimeError("Sparse GEMM backend setup check failed")
            _library = bridge
            _reason = str(vendor)
            logger.info("Sparse GEMM backend: native FP32 and dynamic-precision FP64 cuBLAS (%s)", vendor)
        except (ImportError, OSError, RuntimeError, AttributeError, ValueError,
                TypeError, subprocess.CalledProcessError) as exc:
            _reason = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                _reason += ": " + (exc.stderr or "")[-2000:]
            logger.info("Sparse GEMM backend: native JAX (%s)", _reason)
        return _library is not None


def matmul(a, b, *, transpose_a=False, transpose_b=False, workspace_limit=0, split_k=False):
    """Matrix product with explicit transposes and row-major FFI operands."""
    if (a.ndim != 2 or b.ndim != 2 or a.dtype != b.dtype
            or a.dtype not in (jnp.float32, jnp.float64)):
        raise ValueError("Sparse GEMM requires matching float32 or float64 matrices")
    m, k = a.shape[::-1] if transpose_a else a.shape
    kb, n = b.shape[::-1] if transpose_b else b.shape
    if k != kb:
        raise ValueError("Sparse GEMM dimension mismatch")
    if min(m, n, k) == 0:
        # Empty covariate/support blocks are valid in sparse fitting. cuBLAS
        # cannot dereference empty buffers; preserve NumPy/JAX dot semantics.
        return jnp.zeros((m, n), dtype=a.dtype)
    from jax import ffi
    return ffi.ffi_call(
        "split_sparse_fp64" if a.dtype == jnp.float64 else "split_sparse_fp32",
        jax.ShapeDtypeStruct((m,n), a.dtype),
        input_layouts=[(0,1),(0,1)], output_layouts=(0,1), vmap_method="sequential",
    )(a, b, transpose_a=bool(transpose_a), transpose_b=bool(transpose_b), split_k=bool(split_k),
      workspace_limit=np.int64(workspace_limit))
