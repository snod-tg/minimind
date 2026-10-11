#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
extern "C" void ep_sort(const int64_t*,int,int,int,int64_t*,int64_t*,int64_t*,int*,int*,int*,cudaStream_t);
extern "C" void ep_pack(const void*,const int64_t*,const int64_t*,void*,int64_t*,int,int,int,int,int,cudaStream_t);
extern "C" void ep_gather(const void*,const int64_t*,void*,int,int,int,cudaStream_t);
extern "C" void ep_combine(const void*,const void*,const int64_t*,void*,int,int,int,int,int,int,cudaStream_t);
extern "C" void ep_masked_sort(const int64_t*,int,int,int64_t*,int64_t*,int64_t*,cudaStream_t);
extern "C" void ep_pack_fixed(const void*,const int64_t*,const int64_t*,const int64_t*,void*,int64_t*,int64_t*,int,int,int,int,int,int,cudaStream_t);
extern "C" void ep_scatter_fixed(const void*,const int64_t*,void*,int,int,int,cudaStream_t);
extern "C" void ep_sort_masked(const int64_t*,int,int,int64_t*,int64_t*,int64_t*,int*,int*,int*,cudaStream_t);
using at::Tensor;
static void check(const Tensor& t,const Tensor& ref) {
    TORCH_CHECK(t.is_cuda() && t.device()==ref.device() && t.is_contiguous(),"expected contiguous same-device CUDA tensor");
    TORCH_CHECK(!t.requires_grad(),"raw out API cannot accept autograd tensors; use functional EP training ops");
}
static void ints(const Tensor& t,const Tensor& ref) {check(t,ref);TORCH_CHECK(t.scalar_type()==at::kLong,"mapping/counts must be int64");}
static int type(const Tensor& t) {
    if(t.scalar_type()==at::kFloat)return 0;
    if(t.scalar_type()==at::kHalf)return 1;
    TORCH_CHECK(t.scalar_type()==at::kBFloat16,"supported types: FP32/FP16/BF16");return 2;
}
static void sort_out(const Tensor& ids,int64_t divisor,const Tensor& perm,const Tensor& inv,
                     const Tensor& counts,const Tensor& pos,const Tensor& blocks,const Tensor& offsets) {
    ints(ids,ids);ints(perm,ids);ints(inv,ids);ints(counts,ids);
    int64_t n=ids.numel(),b=(n+255)/256;
    TORCH_CHECK(ids.dim()==1 && divisor>=1 && n<=2147483647,"invalid partition input");
    TORCH_CHECK(perm.numel()==n && inv.numel()==n && (counts.numel()>=1 && counts.numel()<=1024),"invalid map sizes");
    for(auto t:{pos,blocks,offsets}){check(t,ids);TORCH_CHECK(t.scalar_type()==at::kInt,"scratch must be int32");}
    TORCH_CHECK(pos.numel()>=n && blocks.numel()>=b*counts.numel() && offsets.numel()>=b*counts.numel(),"insufficient scratch capacity");
    c10::cuda::CUDAGuard guard(ids.device());
    ep_sort(ids.data_ptr<int64_t>(),n,divisor,counts.numel(),perm.data_ptr<int64_t>(),inv.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),pos.data_ptr<int>(),blocks.data_ptr<int>(),offsets.data_ptr<int>(),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
static void pack_out(const Tensor& x,const Tensor& ids,const Tensor& perm,int64_t topk,int64_t experts,const Tensor& out,const Tensor& local) {
    check(x,x);check(out,x);ints(ids,x);ints(perm,x);ints(local,x);
    TORCH_CHECK(experts>=1 && experts<=1024,"invalid local expert count");
    TORCH_CHECK(x.dim()==2 && topk>=1 && out.dim()==2 && out.scalar_type()==x.scalar_type(),"invalid pack dimensions/types");
    TORCH_CHECK(ids.numel()==x.size(0)*topk && perm.numel()==ids.numel() && local.numel()==ids.numel() && out.size(0)==ids.numel() && out.size(1)==x.size(1),"invalid pack sizes");
    c10::cuda::CUDAGuard guard(x.device());ep_pack(x.data_ptr(),ids.data_ptr<int64_t>(),perm.data_ptr<int64_t>(),out.data_ptr(),local.data_ptr<int64_t>(),out.size(0),x.size(1),topk,experts,type(x),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
static void gather_out(const Tensor& x,const Tensor& map,const Tensor& out) {
    check(x,x);check(out,x);ints(map,x);
    TORCH_CHECK(x.dim()==2 && out.dim()==2 && out.size(1)==x.size(1) && out.scalar_type()==x.scalar_type() && map.numel()==out.size(0),"invalid gather sizes/types");
    c10::cuda::CUDAGuard guard(x.device());ep_gather(x.data_ptr(),map.data_ptr<int64_t>(),out.data_ptr(),out.size(0),x.size(1),type(x),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
static void combine_out(const Tensor& x,const Tensor& weights,const Tensor& inv,int64_t topk,const Tensor& out) {
    check(x,x);check(weights,x);check(out,x);ints(inv,x);
    TORCH_CHECK(x.dim()==2 && out.dim()==2 && topk>=1  ,"invalid combine dimensions/types");
    TORCH_CHECK(x.size(0)>=out.size(0)*topk && x.size(1)==out.size(1) && inv.numel()==out.size(0)*topk && weights.numel()==inv.numel(),"invalid combine sizes");
    c10::cuda::CUDAGuard guard(x.device());ep_combine(x.data_ptr(),weights.data_ptr(),inv.data_ptr<int64_t>(),out.data_ptr(),out.size(0),out.size(1),topk,type(x),type(weights),type(out),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
static void sort_masked_out(const Tensor& ids,const Tensor& perm,const Tensor& inv,const Tensor& counts,const Tensor& pos,const Tensor& blocks,const Tensor& offsets){
    ints(ids,ids);ints(perm,ids);ints(inv,ids);ints(counts,ids);
    int64_t n=ids.numel(),b=(n+255)/256,bins=counts.numel();
    TORCH_CHECK(ids.dim()==1 && n<=2147483647 && bins>=1 && bins<=1024 && perm.numel()==n && inv.numel()==n,"invalid masked partition input");
    for(auto t:{pos,blocks,offsets}){check(t,ids);TORCH_CHECK(t.scalar_type()==at::kInt,"scratch must be int32");}
    TORCH_CHECK(pos.numel()>=n && blocks.numel()>=b*bins && offsets.numel()>=b*bins,"insufficient masked scratch capacity");
    c10::cuda::CUDAGuard guard(ids.device());ep_sort_masked(ids.data_ptr<int64_t>(),n,bins,perm.data_ptr<int64_t>(),inv.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),pos.data_ptr<int>(),blocks.data_ptr<int>(),offsets.data_ptr<int>(),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
static void masked_sort_out(const Tensor& ids,const Tensor& perm,const Tensor& inv,const Tensor& counts) {
    ints(ids,ids);ints(perm,ids);ints(inv,ids);ints(counts,ids);
    TORCH_CHECK(ids.dim()==1 && ids.numel()<=256 && perm.numel()==ids.numel() && inv.numel()==ids.numel() && (counts.numel()>=1 && counts.numel()<=1024),"masked Decode partition supports at most 256 rows");
    c10::cuda::CUDAGuard guard(ids.device());ep_masked_sort(ids.data_ptr<int64_t>(),ids.numel(),counts.numel(),perm.data_ptr<int64_t>(),inv.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
static void pack_fixed_out(const Tensor& x,const Tensor& ids,const Tensor& perm,const Tensor& counts,int64_t topk,int64_t experts,const Tensor& out,const Tensor& local,const Tensor& inverse) {
    check(x,x);check(out,x);ints(ids,x);ints(perm,x);ints(counts,x);ints(local,x);ints(inverse,x);
    TORCH_CHECK(x.dim()==2 && out.dim()==2 && out.scalar_type()==x.scalar_type() && topk>=1,"invalid fixed pack dimensions/types");
    TORCH_CHECK(counts.numel()>=1 && counts.numel()<=1024 && experts>=1 && experts<=1024,"invalid fixed rank/local-expert count");
    TORCH_CHECK(out.size(0)%counts.numel()==0 && out.size(0)/counts.numel()>=ids.numel() && out.size(1)==x.size(1) && local.numel()==out.size(0),"invalid fixed capacity");
    TORCH_CHECK(ids.numel()==x.size(0)*topk && perm.numel()==ids.numel() && inverse.numel()==ids.numel() && (counts.numel()>=1 && counts.numel()<=1024),"invalid fixed mapping sizes");
    c10::cuda::CUDAGuard guard(x.device());ep_pack_fixed(x.data_ptr(),ids.data_ptr<int64_t>(),perm.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),out.data_ptr(),local.data_ptr<int64_t>(),inverse.data_ptr<int64_t>(),x.size(1),topk,out.size(0)/counts.numel(),counts.numel(),experts,type(x),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
static void scatter_fixed_out(const Tensor& x,const Tensor& inv,const Tensor& out) {
    check(x,x);check(out,x);ints(inv,x);
    TORCH_CHECK(x.dim()==2 && out.dim()==2 && x.scalar_type()==out.scalar_type() && x.size(1)==out.size(1) && inv.numel()==out.size(0),"invalid fixed scatter sizes/types");
    c10::cuda::CUDAGuard guard(x.device());ep_scatter_fixed(x.data_ptr(),inv.data_ptr<int64_t>(),out.data_ptr(),out.size(0),out.size(1),type(x),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY(my_cuda_ep,m) {
 m.def("sort_masked_out(Tensor ids, Tensor(a!) perm, Tensor(b!) inverse, Tensor(c!) counts, Tensor(d!) positions, Tensor(e!) blocks, Tensor(f!) offsets) -> ()");
 m.def("masked_sort_out(Tensor ids, Tensor(a!) perm, Tensor(b!) inverse, Tensor(c!) counts) -> ()");
 m.def("pack_fixed_out(Tensor x, Tensor ids, Tensor perm, Tensor counts, int topk, int experts, Tensor(a!) out, Tensor(b!) local, Tensor(c!) inverse) -> ()");
 m.def("scatter_fixed_out(Tensor x, Tensor inverse, Tensor(a!) out) -> ()");
 m.def("sort_out(Tensor ids, int divisor, Tensor(a!) perm, Tensor(b!) inverse, Tensor(c!) counts, Tensor(d!) positions, Tensor(e!) blocks, Tensor(f!) offsets) -> ()");
 m.def("pack_out(Tensor x, Tensor ids, Tensor perm, int topk, int experts, Tensor(a!) out, Tensor(b!) local_ids) -> ()");
 m.def("gather_out(Tensor x, Tensor map, Tensor(a!) out) -> ()");
 m.def("combine_out(Tensor x, Tensor weights, Tensor inverse, int topk, Tensor(a!) out) -> ()");
}
TORCH_LIBRARY_IMPL(my_cuda_ep,CUDA,m) {m.impl("sort_masked_out",sort_masked_out);m.impl("masked_sort_out",masked_sort_out);m.impl("pack_fixed_out",pack_fixed_out);m.impl("scatter_fixed_out",scatter_fixed_out);m.impl("sort_out",sort_out);m.impl("pack_out",pack_out);m.impl("gather_out",gather_out);m.impl("combine_out",combine_out);}
