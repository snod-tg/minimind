// Latest prefill/decode kernels; no duplicate GEMM implementation in the binding.
#include "moe_grouped_gemm_forward_prefill.cu"
#include "moe_grouped_gemm_forward_decode.cu"

size_t split_metadata_bytes() { return sizeof(grouped9::Metadata); }
int split_parts(int family,int m,int n,int k) {
    if(family==0 || !m)return 0;
    int parts=grouped9::decode_config(grouped9::decode_default(m,n,k)).parts;
    return parts>1?parts:0;
}
void split_launch(void* out,const void* x,const void* b,const int64_t* counts,
    void* metadata,float* partial,int family,int dtype,int e,int m,int n,int k,cudaStream_t stream) {
    auto run=[&](auto* c,const auto* a,const auto* weights) {
        if(family==0)grouped9::launch_prefill(c,a,weights,counts,
            static_cast<grouped9::Metadata*>(metadata),e,m,n,k,stream);
        else grouped9::launch_decode(c,a,weights,counts,partial,e,m,n,k,stream);
    };
    if(dtype==0)run(static_cast<half*>(out),static_cast<const half*>(x),static_cast<const half*>(b));
    else run(static_cast<__nv_bfloat16*>(out),static_cast<const __nv_bfloat16*>(x),static_cast<const __nv_bfloat16*>(b));
}
