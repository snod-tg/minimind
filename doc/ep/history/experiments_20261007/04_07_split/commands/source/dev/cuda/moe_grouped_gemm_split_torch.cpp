// Tensor checks and dispatcher registration; no Python headers or counts synchronization.
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <climits>

size_t split_metadata_bytes();
int split_parts(int,int,int,int);
void split_launch(void*,const void*,const void*,const int64_t*,void*,float*,int,int,int,int,int,int,cudaStream_t);
namespace {
void aligned(const at::Tensor& t,size_t bytes,const char* name) {
    TORCH_CHECK(t.numel()==0 || reinterpret_cast<uintptr_t>(t.const_data_ptr())%bytes==0,
                name," storage offset must preserve ",bytes,"-byte alignment");
}
void inputs(int family,const at::Tensor& x,const at::Tensor& b,const at::Tensor& counts) {
    TORCH_CHECK(x.is_cuda() && b.is_cuda() && counts.is_cuda(),"inputs must be CUDA tensors");
    TORCH_CHECK(x.device()==b.device() && x.device()==counts.device(),"inputs must share a CUDA device");
    TORCH_CHECK(x.dim()==2 && b.dim()==3 && counts.dim()==1,"expected x[M,K], packed_b[E,K,N], counts[E]");
    TORCH_CHECK(x.is_contiguous() && b.is_contiguous() && counts.is_contiguous(),"inputs must be contiguous");
    TORCH_CHECK((x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16) && b.scalar_type()==x.scalar_type(),
                "x and packed_b must share FP16/BF16 dtype");
    TORCH_CHECK(counts.scalar_type()==at::kLong,"counts must be int64");
    TORCH_CHECK(b.size(0)>=1 && b.size(0)<=16 && counts.size(0)==b.size(0),"expected 1<=E<=16 and counts[E]");
    int64_t m=x.size(0),k=x.size(1),n=b.size(2);
    TORCH_CHECK(family==0 || (family==1 && m<=256),"decode supports M<=256");
    TORCH_CHECK(m<=INT_MAX && b.size(1)==k,"M exceeds int32 or K dimensions differ");
    TORCH_CHECK((k==768 && n==2432)||(k==2432 && n==768)||(k==1024 && n==1024),"unsupported K/N");
    TORCH_CHECK(!x.requires_grad() && !b.requires_grad(),
                "grouped_gemm is forward-only; inputs must not require gradients");
    aligned(x,16,"x");aligned(b,16,"packed_b");aligned(counts,8,"counts");
}
void buffer(const at::Tensor& t,const at::Tensor& x,at::ScalarType dtype,const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device()==x.device(),name," must share x's CUDA device");
    TORCH_CHECK(t.scalar_type()==dtype && t.is_contiguous(),name," has wrong dtype or strides");
    TORCH_CHECK(!t.requires_grad(),name," must not require gradients");
}
void grouped_out(int family,const at::Tensor& x,const at::Tensor& b,const at::Tensor& counts,
    at::Tensor out,at::Tensor metadata,at::Tensor partial) {
    inputs(family,x,b,counts);
    buffer(out,x,x.scalar_type(),"out");buffer(metadata,x,at::kByte,"metadata");buffer(partial,x,at::kFloat,"partial");
    int m=x.size(0),k=x.size(1),n=b.size(2),e=b.size(0);
    TORCH_CHECK(out.dim()==2 && out.size(0)==m && out.size(1)==n,"out must have shape [M,N]");
    TORCH_CHECK(metadata.dim()==1 && metadata.numel()>=int64_t(split_metadata_bytes()),"metadata too small");
    int parts=m?split_parts(family,m,n,k):0;
    TORCH_CHECK(partial.dim()==1 && partial.numel()>=int64_t(parts)*m*n,"partial workspace too small");
    aligned(out,16,"out");aligned(metadata,8,"metadata");aligned(partial,8,"partial");
    // Any write buffer must be disjoint from inputs and other write buffers.
    for(const auto& t:{out,metadata,partial})for(const auto& read:{x,b,counts})at::assert_no_overlap(t,read);
    at::assert_no_overlap(out,metadata);at::assert_no_overlap(out,partial);at::assert_no_overlap(metadata,partial);
    c10::cuda::CUDAGuard guard(x.device());
    auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
    split_launch(out.mutable_data_ptr(),x.const_data_ptr(),b.const_data_ptr(),counts.const_data_ptr<int64_t>(),
        metadata.mutable_data_ptr(),partial.mutable_data_ptr<float>(),family,x.scalar_type()==at::kHalf?0:1,e,m,n,k,stream.stream());
}
at::Tensor grouped(int family,const at::Tensor& x,const at::Tensor& b,const at::Tensor& counts) {
    inputs(family,x,b,counts);
    c10::cuda::CUDAGuard guard(x.device());
    int m=x.size(0),k=x.size(1),n=b.size(2);
    auto out=at::empty({m,n},x.options());
    auto metadata=at::empty({int64_t(split_metadata_bytes())},x.options().dtype(at::kByte));
    auto partial=at::empty({int64_t(m?split_parts(family,m,n,k):0)*m*n},x.options().dtype(at::kFloat));
    grouped_out(family,x,b,counts,out,metadata,partial);
    return out;
}
}
TORCH_LIBRARY(my_cuda_kernels_split,m) {
    m.def("prefill(Tensor x, Tensor packed_b, Tensor counts) -> Tensor");
    m.def("decode(Tensor x, Tensor packed_b, Tensor counts) -> Tensor");
    m.def("prefill_out(Tensor x, Tensor packed_b, Tensor counts, Tensor(a!) out, Tensor(b!) metadata, Tensor(c!) partial) -> ()");
    m.def("decode_out(Tensor x, Tensor packed_b, Tensor counts, Tensor(a!) out, Tensor(b!) metadata, Tensor(c!) partial) -> ()");
    m.def("metadata_bytes() -> int");
    m.def("parts(int family, int m, int n, int k) -> int");
    m.impl("metadata_bytes",[](){return int64_t(split_metadata_bytes());});
    m.impl("parts",[](int64_t family,int64_t m,int64_t n,int64_t k){
        TORCH_CHECK((family==0 || family==1) && m>=0 && m<=INT_MAX && (family!=1 || m<=256) &&
                    ((k==768 && n==2432)||(k==2432 && n==768)||(k==1024 && n==1024)),"unsupported workspace shape");
        return int64_t(split_parts(family,m,n,k));
    });
}
TORCH_LIBRARY_IMPL(my_cuda_kernels_split,CUDA,m) {
    m.impl("prefill",[](const at::Tensor& x,const at::Tensor& b,const at::Tensor& counts){return grouped(0,x,b,counts);});
    m.impl("decode",[](const at::Tensor& x,const at::Tensor& b,const at::Tensor& counts){return grouped(1,x,b,counts);});
    m.impl("prefill_out",[](const at::Tensor& x,const at::Tensor& b,const at::Tensor& counts,at::Tensor out,at::Tensor metadata,at::Tensor partial){grouped_out(0,x,b,counts,out,metadata,partial);});
    m.impl("decode_out",[](const at::Tensor& x,const at::Tensor& b,const at::Tensor& counts,at::Tensor out,at::Tensor metadata,at::Tensor partial){grouped_out(1,x,b,counts,out,metadata,partial);});
}
