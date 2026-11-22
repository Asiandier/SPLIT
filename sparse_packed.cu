// Deterministic packed genotype products for the sparse workflow.
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include "xla/ffi/api/ffi.h"
namespace ffi=xla::ffi;

template<class T> __global__ void Transpose(const uint8_t* g,const float* means,
    const int32_t* valid,const uint8_t* missing,const T* v,T* out,
    int64_t n,int64_t stride,int64_t rhs) {
  __shared__ T partial[8][32][4];
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  int64_t pack=blockIdx.x*32+lane, first=blockIdx.y*512;
  T sum[4]={0,0,0,0}, mean[4]={0,0,0,0};
  int width=*valid, miss=*missing;
  if(pack<stride) for(int j=0;j<4;++j) mean[j]=static_cast<T>(means[pack*4+j]);
  if(pack<stride) for(int64_t row=first+warp;row<n && row<first+512;row+=8) {
    uint8_t raw=g[row*stride+pack]; T value=v[row*rhs+blockIdx.z];
    for(int j=0;j<4;++j) {
      int genotype=(raw>>(2*j))&3;
      if(pack*4+j<width && genotype!=miss) sum[j]+=(static_cast<T>(genotype)-mean[j])*value;
    }
  }
  for(int j=0;j<4;++j)partial[warp][lane][j]=sum[j];
  __syncthreads();
  if(warp==0 && pack<stride)for(int j=0;j<4;++j) {
    T value=0;for(int k=0;k<8;++k)value+=partial[k][lane][j];
    out[(static_cast<int64_t>(blockIdx.y)*stride*4+pack*4+j)*rhs+blockIdx.z]=value;
  }
}

template<class T,bool Lookup> __global__ void Forward(const uint8_t* g,const float* means,
    const int32_t* valid,const uint8_t* missing,const T* v,T* out,
    int64_t n,int64_t stride,int64_t rhs) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int64_t row=static_cast<int64_t>(blockIdx.x)*8+warp, first=blockIdx.y*512;
  if(row>=n)return;
  T sum=0;
  int width=0,miss=0;
  if constexpr(!Lookup) {width=*valid;miss=*missing;}
  for(int64_t pack=first+lane;pack<stride && pack<first+512;pack+=32) {
    uint8_t raw=g[row*stride+pack];
    for(int j=0;j<4;++j) {
      int genotype=(raw>>(2*j))&3;
      if constexpr(Lookup)
        sum+=v[(pack*4+j)*4+genotype];
      else if(pack*4+j<width && genotype!=miss)
        sum+=(static_cast<T>(genotype)-static_cast<T>(means[pack*4+j]))*v[(pack*4+j)*rhs+blockIdx.z];
    }
  }
  for(int offset=16;offset>0;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);
  if(lane==0)out[(static_cast<int64_t>(blockIdx.y)*n+row)*rhs+blockIdx.z]=sum;
}

template<ffi::DataType D> ffi::Error Call(cudaStream_t stream,bool transpose,bool lookup,
    ffi::Buffer<ffi::U8,2> packed,ffi::Buffer<ffi::F32,1> means,
    ffi::Buffer<ffi::S32,0> valid,ffi::Buffer<ffi::U8,0> missing,
    ffi::Buffer<D,2> value,ffi::ResultBuffer<D,3> out) {
  using T=std::conditional_t<D==ffi::F64,double,float>;
  int64_t n=packed.dimensions()[0],stride=packed.dimensions()[1],rhs=value.dimensions()[1];
  auto od=out->dimensions();
  if(n<1 || stride<1 || rhs<1 || means.dimensions()[0]<stride*4
      || (lookup && (transpose || rhs!=1 || D!=ffi::F64))
      || value.dimensions()[0]!=(transpose?n:stride*(lookup?16:4))
      || od[0]!=(transpose?(n+511)/512:(stride+511)/512)
      || od[1]!=(transpose?stride*4:n) || od[2]!=rhs)
    return ffi::Error::InvalidArgument("Invalid packed GEMV dimensions");
  if(transpose)Transpose<T><<<dim3((stride+31)/32,(n+511)/512,rhs),256,0,stream>>>(
      packed.typed_data(),means.typed_data(),valid.typed_data(),missing.typed_data(),value.typed_data(),out->typed_data(),n,stride,rhs);
  else if(lookup)Forward<T,true><<<dim3((n+7)/8,(stride+511)/512,1),256,0,stream>>>(
      packed.typed_data(),means.typed_data(),valid.typed_data(),missing.typed_data(),value.typed_data(),out->typed_data(),n,stride,rhs);
  else Forward<T,false><<<dim3((n+7)/8,(stride+511)/512,rhs),256,0,stream>>>(
      packed.typed_data(),means.typed_data(),valid.typed_data(),missing.typed_data(),value.typed_data(),out->typed_data(),n,stride,rhs);
  auto error=cudaGetLastError();
  if(error!=cudaSuccess)return ffi::Error::Internal(cudaGetErrorString(error));
  return ffi::Error::Success();
}
using Packed=ffi::Buffer<ffi::U8,2>;
using Means=ffi::Buffer<ffi::F32,1>;
using Valid=ffi::Buffer<ffi::S32,0>;
using Missing=ffi::Buffer<ffi::U8,0>;
template<ffi::DataType D>using Value=ffi::Buffer<D,2>;
template<ffi::DataType D>using Output=ffi::Buffer<D,3>;
#define BIND(D) ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Attr<bool>("transpose")\
  .Attr<bool>("lookup")\
  .Arg<Packed>().Arg<Means>().Arg<Valid>().Arg<Missing>().Arg<Value<D>>().Ret<Output<D>>()
XLA_FFI_DEFINE_HANDLER_SYMBOL(PackedFloat,Call<ffi::F32>,BIND(ffi::F32));
XLA_FFI_DEFINE_HANDLER_SYMBOL(PackedDouble,Call<ffi::F64>,BIND(ffi::F64));
