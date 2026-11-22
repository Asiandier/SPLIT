// Optional sparse-only native FP32 and dynamic-precision FP64 GEMMs.
// No CUDA runtime or cuBLAS is linked globally. XLA owns the workspace and
// supplies the execution stream. All buffers use explicit row-major layouts.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using Matrix = ffi::Buffer<ffi::F64,2>;
using Matrix32 = ffi::Buffer<ffi::F32,2>;
namespace {
using Handle = void*;
using Stream = void*;
struct Api {
  void* library = nullptr;
  int (*create)(Handle*) = nullptr;
  int (*destroy)(Handle) = nullptr;
  int (*version)(Handle, int*) = nullptr;
  int (*stream)(Handle, Stream) = nullptr;
  int (*math)(Handle, int) = nullptr;
  int (*workspace)(Handle, void*, size_t) = nullptr;
  int (*mantissa)(Handle, int*) = nullptr;
  int (*control)(Handle, int*) = nullptr;
  int (*offset)(Handle, int*) = nullptr;
  int (*gemm)(Handle, int, int, int, int, int, const double*, const double*,
              int, const double*, int, const double*, double*, int) = nullptr;
  int (*gemm_ex)(Handle, int, int, int, int, int, const void*, const void*,
                 int, int, const void*, int, int, const void*, void*,
                 int, int, int, int) = nullptr;
} api;

void Check(int status, const char* operation) {
  if (status) throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
}
template <typename T> void Load(T& fn, const char* name) {
  fn = reinterpret_cast<T>(dlsym(api.library, name));
  if (!fn) throw std::runtime_error(std::string("Missing cuBLAS API: ") + name);
}
struct Context {
  Handle handle = nullptr;
  int bits = 0;
  Context() {
    Check(api.create(&handle), "create");
    try {
    int version = 0, control = -1, offset = -1;
    Check(api.version(handle, &version), "version");
    if (version < 130100) throw std::runtime_error("FP64 emulation needs cuBLAS >=13.1");
    Check(api.control(handle, &control), "mantissa control");
    Check(api.offset(handle, &offset), "mantissa offset");
    if (control != 0 || offset != 0)
      throw std::runtime_error("FP64 backend requires default dynamic precision without a bit offset");
    Check(api.mantissa(handle, &bits), "mantissa budget");
    // cuBLAS 13.1 returns 0 for its unresolved library default, not zero
    // mantissa bits. Its documented dynamic default is 79 bits (10 slices):
    // docs.nvidia.com/cuda/archive/13.0.3/cublas/index.html
    // section "Default Library Configurations". Resolve only this known
    // release; a future unresolved default must fall back to native JAX.
    // This sizes scratch only. Do not set mantissa control, count or offset.
    if (bits == 0 && version >= 130100 && version < 130200) bits = 79;
    if (bits <= 0) throw std::runtime_error("Unknown default FP64 workspace precision");
    // CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH; the library selects native FP64
    // when emulation is not profitable or dynamic precision needs more bits.
    Check(api.math(handle, 8), "FP64 math mode");
    } catch (...) {
      api.destroy(handle);
      handle = nullptr;
      throw;
    }
  }
  ~Context() { if (handle) api.destroy(handle); }
  Context(const Context&) = delete;
};

size_t Round(size_t value, size_t step) { return (value + step - 1) / step * step; }
size_t Workspace(size_t m, size_t n, size_t k, int bits) {
  // NVIDIA's documented bound for real, dynamic-precision FP64 emulation.
  size_t pm = Round(m,1024), pn = Round(n,1024), pk = Round(k,128);
  size_t slices = (static_cast<size_t>(bits) + 8) / 8;
  size_t gemm = (pm+pn)*pk*slices + 4*(pm+pn);
  size_t adp = 4*((m+n)*((k+63)/64) + m*n);
  return (std::max(gemm,adp)*5 + 3)/4 + 128*1024*1024;
}

template <ffi::DataType D>
ffi::Error Matmul(Stream stream, int32_t device, ffi::ScratchAllocator scratch,
                  bool ta, bool tb, bool split_k, int64_t limit,
                  ffi::Buffer<D,2> a, ffi::Buffer<D,2> b,
                  ffi::ResultBuffer<D,2> out) {
  constexpr bool fp64 = D == ffi::F64;
  using Scalar = std::conditional_t<fp64, double, float>;
  try {
    if (!api.library) return ffi::Error::Internal("Sparse GEMM backend not initialized");
    // Handles are never concurrently mutated or shared between devices/streams.
    thread_local std::map<std::pair<int32_t,uintptr_t>, Context> contexts;
    auto& ctx = contexts.try_emplace({device,reinterpret_cast<uintptr_t>(stream)}).first->second;
    Check(api.stream(ctx.handle,stream), "set stream");
    auto ad = a.dimensions(), bd = b.dimensions(), cd = out->dimensions();
    int64_t m=ta?ad[1]:ad[0], k=ta?ad[0]:ad[1], n=tb?bd[0]:bd[1];
    if (k != (tb?bd[1]:bd[0]) || cd[0]!=m || cd[1]!=n || m<1 || n<1 || k<1
        || std::max({m,n,k,ad[1],bd[1]})>INT32_MAX)
      return ffi::Error::InvalidArgument("Invalid sparse GEMM dimensions");
    int64_t tm=m, tn=n, tk=k;
    const auto fits = [&](int64_t rows, int64_t cols, int64_t inner) {
      return Workspace(rows,cols,inner,ctx.bits)+255<=static_cast<size_t>(limit);
    };
    if (fp64 && split_k && limit>0 && !fits(m,n,k) && fits(m,n,std::min<int64_t>(k,128))) {
      // A tall Gram product may exceed the cap even for one output tile.
      // Accumulate FP64 products of consecutive row tiles.
      int64_t low=std::min<int64_t>(k,128), high=k;
      while (low<high) {
        int64_t middle=low+(high-low+1)/2;
        if (fits(m,n,middle)) low=middle;
        else high=middle-1;
      }
      tk=low;
    }
    while (fp64 && limit>0 && !fits(tm,tn,tk) && (tm>1 || tn>1)) {
      if (tm>=tn && tm>1) tm=(tm+1)/2; else tn=(tn+1)/2;
    }
    // Native FP32 needs no precision-emulation scratch. Bound its GEMM
    // workspace independently; it is much smaller than the FP64 allowance.
    size_t bytes=fp64 ? Workspace(tm,tn,tk,ctx.bits)
        : static_cast<size_t>(std::min<int64_t>(64*1024*1024, std::max<int64_t>(0,limit-255)));
    void* work=nullptr;
    if (limit>0 && bytes+255<=static_cast<size_t>(limit)) {
      // XLA's public allocator accepts alignments up to 16 bytes; cuBLAS
      // requires 256. Own the padded allocation, pass its aligned interior.
      auto allocated=scratch.Allocate(bytes+255,16);
      // A planner bound does not guarantee a contiguous allocator block.
      // Reduce output tiles, retaining full precision, if fragmentation or
      // concurrent live buffers prevent the preferred scratch allocation.
      while (!allocated && fp64 && (tm>1 || tn>1)) {
        const size_t previous=bytes;
        do {
          if (tm>=tn && tm>1) tm=(tm+1)/2; else tn=(tn+1)/2;
          bytes=Workspace(tm,tn,tk,ctx.bits);
        } while (bytes>=previous && (tm>1 || tn>1));
        if (bytes>=previous) break;
        allocated=scratch.Allocate(bytes+255,16);
      }
      if (allocated) {
        work=reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(*allocated)+255)&~uintptr_t(255));
        Check(api.math(ctx.handle,fp64?8:0), "GEMM math mode");
      } else {
        tm=m;tn=n;tk=k;bytes=0;
        Check(api.math(ctx.handle,0), "native math without scratch");
      }
    } else {
      tm=m;tn=n;tk=k;bytes=0;
      Check(api.math(ctx.handle,0), "native math");
    }
    // Explicit workspace prevents cuBLAS from allocating outside XLA's budget.
    Check(api.workspace(ctx.handle,work,bytes), "set workspace");
    const Scalar alpha=1.;
    for (int64_t i=0;i<m;i+=tm) for (int64_t j=0;j<n;j+=tn) {
      int rows=static_cast<int>(std::min(tm,m-i));
      int cols=static_cast<int>(std::min(tn,n-j));
      // C_row^T = op(B_row)^T op(A_row)^T, using column-major cuBLAS.
      if constexpr (fp64) {
        for (int64_t offset=0;offset<k;offset+=tk) {
          int inner=static_cast<int>(std::min(tk,k-offset));
          const Scalar beta=offset==0?0.:1.;
          const Scalar* ap=a.typed_data()+(ta?offset*ad[1]+i:i*ad[1]+offset);
          const Scalar* bp=b.typed_data()+(tb?j*bd[1]+offset:offset*bd[1]+j);
          Check(api.gemm(ctx.handle,tb?1:0,ta?1:0,cols,rows,inner,
              &alpha,bp,static_cast<int>(bd[1]),ap,static_cast<int>(ad[1]),&beta,
              out->typed_data()+i*n+j,static_cast<int>(n)), "Dgemm");
        }
      } else {
        // CUDA_R_32F=0, CUBLAS_COMPUTE_32F=68, CUBLAS_GEMM_DEFAULT=-1.
        // Use native FP32, never FAST_TF32/FP16/BF16 or reduced precision.
        const Scalar beta=0.;
        const Scalar* ap=a.typed_data()+(ta?i:i*ad[1]);
        const Scalar* bp=b.typed_data()+(tb?j*bd[1]:j);
        Check(api.gemm_ex(ctx.handle,tb?1:0,ta?1:0,cols,rows,static_cast<int>(k),
            &alpha,bp,0,static_cast<int>(bd[1]),ap,0,static_cast<int>(ad[1]),&beta,
            out->typed_data()+i*n+j,0,static_cast<int>(n),68,-1), "FP32 GemmEx");
      }
    }
    return ffi::Error::Success();
  } catch (const std::exception& e) { return ffi::Error::Internal(e.what()); }
}
}  // namespace

extern "C" int split_fp64_init(const char* library, char* error, size_t capacity) {
  try {
    if (api.library) return 0;
    api.library=dlopen(library,RTLD_NOW|RTLD_LOCAL);
    if (!api.library) throw std::runtime_error(dlerror());
    Load(api.create,"cublasCreate_v2"); Load(api.destroy,"cublasDestroy_v2");
    Load(api.version,"cublasGetVersion_v2"); Load(api.stream,"cublasSetStream_v2");
    Load(api.math,"cublasSetMathMode"); Load(api.workspace,"cublasSetWorkspace_v2");
    Load(api.mantissa,"cublasGetFixedPointEmulationMaxMantissaBitCount");
    Load(api.control,"cublasGetFixedPointEmulationMantissaControl");
    Load(api.offset,"cublasGetFixedPointEmulationMantissaBitOffset");
    Load(api.gemm,"cublasDgemm_v2");
    Load(api.gemm_ex,"cublasGemmEx");
    return 0;
  } catch (const std::exception& e) {
    if (capacity) { std::strncpy(error,e.what(),capacity-1); error[capacity-1]='\0'; }
    if (api.library) dlclose(api.library);
    api = Api{};
    return 1;
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SplitFp64Matmul, Matmul<ffi::F64>,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<Stream>>().Ctx<ffi::DeviceOrdinal>()
      .Ctx<ffi::ScratchAllocator>().Attr<bool>("transpose_a").Attr<bool>("transpose_b")
      .Attr<bool>("split_k").Attr<int64_t>("workspace_limit").Arg<Matrix>()
      .Arg<Matrix>().Ret<Matrix>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(SplitFp32Matmul, Matmul<ffi::F32>,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<Stream>>().Ctx<ffi::DeviceOrdinal>()
      .Ctx<ffi::ScratchAllocator>().Attr<bool>("transpose_a").Attr<bool>("transpose_b")
      .Attr<bool>("split_k").Attr<int64_t>("workspace_limit").Arg<Matrix32>()
      .Arg<Matrix32>().Ret<Matrix32>());
