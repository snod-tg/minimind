// kernel9_prefill: compact expert-major A[M,K], B[E,K,N], C[M,N].
// GPU counts may change in a graph; host knows only total M/shape.
#include "common.h"
namespace grouped9 {
__global__ void prepare_prefill(Metadata* meta,const int64_t* counts,int e,int m,int bm,int bn,int n) {
    if(threadIdx.x)return;meta->offsets[0]=meta->tiles[0]=0;meta->valid=1;
    for(int i=0;i<e;++i) {
        int64_t r=counts[i];if(r<0 || r>m){meta->valid=0;r=0;}
        meta->offsets[i+1]=meta->offsets[i]+r;
        meta->tiles[i+1]=meta->tiles[i]+((r+bm-1)/bm)*((n+bn-1)/bn);
    }
    if(meta->offsets[e]!=m)meta->valid=0;
}
template<class T,int N,int K,int BM,int BN,int BK,int WM,int WN,int ST,bool DB,bool CG,bool EPI,bool UNROLL,bool CROSS>
__global__ __launch_bounds__((BM/WM)*(BN/WN)*32) void moe_grouped_gemm_forward_kernel9_prefill(
    T* c,const T* a,const T* b,const Metadata* meta,int experts,int order) {
    extern __shared__ __align__(16) unsigned char storage[];
    if(!meta->valid)return;
    int64_t total=meta->tiles[experts];
    for(int64_t task=blockIdx.x;task<total;task+=gridDim.x) {
        int e=0;while(task>=meta->tiles[e+1])++e;
        int64_t off=meta->offsets[e],rows=meta->offsets[e+1]-off;
        int mt=int((rows+BM-1)/BM),nt=(N+BN-1)/BN,local=int(task-meta->tiles[e]),r,col;
        if(order==0){r=local%mt;col=local/mt;}
        else if(order==1){r=local/nt;col=local%nt;}
        else { // DeepGEMM-like grouped M swizzle: nearby tasks reuse both A and B.
            int group=local/(8*nt),base=group*8,height=min(8,mt-base),within=local-group*8*nt;
            r=base+within%height;col=within/height;
        }
        gg9_tile<T,BM,BN,BK,WM,WN,ST,N,K,T,DB,CG,EPI,UNROLL,CROSS,true>(c+size_t(off)*N,a+size_t(off)*K,b+size_t(e)*K*N,
            int(rows),r*BM,col*BN,reinterpret_cast<T*>(storage));
        if(gridDim.x<total)__syncthreads();
    }
}
struct PrefillConfig {int bm,bn,bk,wm,wn,st;bool db,cg,epi,unroll,cross;};
inline PrefillConfig prefill_config(int id) {
    static constexpr PrefillConfig cfg[]={
        {32,128,32,32,32,3,0,0,1,0,0},
        {64,128,32,32,64,2,0,0,1,0,0},
        {128,128,32,64,64,2,0,1,1,0,0},
        {64,64,32,32,32,3,0,1,0,0,0},
        {64,64,32,32,32,3,0,1,1,0,1},
        {128,128,32,64,64,2,0,1,1,0,1},
        {64,128,32,32,64,4,1,1,1,0,1},
        {64,64,32,32,32,4,1,1,1,0,1}
    };
    if(id<0 || id>=sizeof(cfg)/sizeof(cfg[0]))throw std::runtime_error("prefill config out of range");return cfg[id];
}
inline int prefill_default(int m,int n,int k) {return n==1024?(m<=2048?0:1):n==768 && m<=4096?4:n==2432 && m>4096?5:2;}
inline int prefill_default_order(int m,int n,int k) {return n==1024?0:n==2432 && m>4096?2:1;}
inline int prefill_default_blocks(int m,int n,int k) {
    int multiplier=n==768 && m>4096?4:n==1024 && m>2048?3:0;
    if(!multiplier)return 0;int device,sms;
    CUDA_CHECK(cudaGetDevice(&device));CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));return multiplier*sms;
}
template<class T,int N,int K> void prefill_shape(T* c,const T* a,const T* b,const int64_t* counts,
    Metadata* meta,float* partial,int experts,int m,int capacity,int config,int blocks,int order,bool masked,cudaStream_t stream,int mode) {
    auto cf=prefill_config(config);if(!m)return;
    if(mode!=1)prepare_prefill<<<1,32,0,stream>>>(meta,counts,experts,m,cf.bm,cf.bn,N);
    if(mode==2)return;
    int64_t cap=((int64_t(m)+cf.bm-1)/cf.bm+std::min(m,experts)-1)*((N+cf.bn-1)/cf.bn);
    unsigned grid=unsigned(std::min<int64_t>(blocks?blocks:INT32_MAX,cap));
    #define LAUNCH(BM,BN,BK,WM,WN,ST,DB,CG,EPI,UNROLL,CROSS) \
      if constexpr(ST*(BM+BN)*BK*sizeof(T)>48*1024) cudaFuncSetAttribute(moe_grouped_gemm_forward_kernel9_prefill<T,N,K,BM,BN,BK,WM,WN,ST,DB,CG,EPI,UNROLL,CROSS>,cudaFuncAttributeMaxDynamicSharedMemorySize,ST*(BM+BN)*BK*sizeof(T)); \
      moe_grouped_gemm_forward_kernel9_prefill<T,N,K,BM,BN,BK,WM,WN,ST,DB,CG,EPI,UNROLL,CROSS><<<grid,(BM/WM)*(BN/WN)*32,ST*(BM+BN)*BK*sizeof(T),stream>>>(c,a,b,meta,experts,order)
    switch(config){
        case 0:LAUNCH(32,128,32,32,32,3,0,0,1,0,0);break;
        case 1:LAUNCH(64,128,32,32,64,2,0,0,1,0,0);break;
        case 2:LAUNCH(128,128,32,64,64,2,0,1,1,0,0);break;
        case 3:LAUNCH(64,64,32,32,32,3,0,1,0,0,0);break;
        case 4:LAUNCH(64,64,32,32,32,3,0,1,1,0,1);break;
        case 5:LAUNCH(128,128,32,64,64,2,0,1,1,0,1);break;
        case 6:LAUNCH(64,128,32,32,64,4,1,1,1,0,1);break;
        case 7:LAUNCH(64,64,32,32,32,4,1,1,1,0,1);break;
    }
    #undef LAUNCH
}
template<class T> void launch_prefill(T* c,const T* a,const T* b,const int64_t* counts,Metadata* meta,
    int experts,int m,int n,int k,cudaStream_t stream,int config=-1,int blocks=-1,int order=-1,int mode=0) {
    if(config<0)config=prefill_default(m,n,k);
    if(order<0)order=prefill_default_order(m,n,k);
    if(blocks<0)blocks=prefill_default_blocks(m,n,k);
    if(!target(experts,n,k)||m<0||blocks<0||order<0||order>2||mode<0||mode>2)throw std::runtime_error("unsupported prefill arguments");
    int capacity=m;float* partial=nullptr;bool masked=false;
    if(n==2432) prefill_shape<T,2432,768>(c,a,b,counts,meta,partial,experts,m,capacity,config,blocks,order,masked,stream,mode);
    else if(n==768) prefill_shape<T,768,2432>(c,a,b,counts,meta,partial,experts,m,capacity,config,blocks,order,masked,stream,mode);
    else prefill_shape<T,1024,1024>(c,a,b,counts,meta,partial,experts,m,capacity,config,blocks,order,masked,stream,mode);
    auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace grouped9

#ifdef GROUPED_PREFILL_MAIN
int main(int argc,char** argv){try {
    auto o=grouped9::cli_options(argc,argv,false);gemm_device_info();
    auto run=[&](auto* c,const auto* a,const auto* b,const int64_t* counts,grouped9::Metadata* meta,float*,int e,int m,int n,int k){grouped9::launch_prefill(c,a,b,counts,meta,e,m,n,k,nullptr);};
    return o.dtype=="bf16"?grouped9::cli_run<__nv_bfloat16>(o,false,run):grouped9::cli_run<half>(o,false,run);
}catch(const std::exception& ex){fprintf(stderr,"%s\n",ex.what());return 1;}}
#endif
