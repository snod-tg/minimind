// kernel9_decode: compact inputs, or masked A[E,capacity,K]/C[E,capacity,N].
// counts on GPU; shape/capacity fixed for Graph reuse; no separate metadata launch.
#include "common.h"
namespace grouped9 {
struct DecodeConfig {
    int bm,bn,bk,wm,wn,st;bool db,cg;int parts;
    int schedule; // 0=expert capacity grid, 1=compact persistent (2 CTA/SM)
};
inline DecodeConfig decode_config(int id) {
    static constexpr DecodeConfig dispatch_configs[]={
        {16,64,64,16,16,3,0,1,1,0},
        {16,64,64,16,16,3,0,1,4,0},
        {16,64,64,16,16,3,0,1,8,0},
        {16,128,64,16,32,2,1,1,4,0},
        {32,64,64,16,32,3,1,1,1,0},
        {16,128,32,16,32,3,0,0,8,0},
        {32,128,32,16,64,3,0,1,1,0},
        {32,64,64,16,32,3,1,1,1,1},
    };
    if(id<0 || id>=sizeof(dispatch_configs)/sizeof(dispatch_configs[0]))throw std::runtime_error("decode config out of range");return dispatch_configs[id];
}
inline int decode_default(int m,int n,int k) {return n==1024?5:n==2432?(m<=16?1:m>64?7:0):(m<=64?2:3);}
// Compact persistent scheduling skips empty experts without a metadata kernel.
// Each CTA advances by gridDim.x; the upper bound covers arbitrary counts summing to M.
// Each warp derives the same expert/row from GPU counts; M allocation stays dynamic.
template<int BM,int NT>
__device__ __forceinline__ bool decode_task(const int64_t* counts,int experts,int m,
    int task,int& expert,int& off,int& rows,int& row,int& col) {
    int lane=threadIdx.x&31;
    int64_t count=lane<experts?counts[lane]:0;
    if(__ballot_sync(0xffffffff,count<0 || count>m))return false;
    if(experts==2) {
        int a=int(__shfl_sync(0xffffffff,count,0)),b=int(__shfl_sync(0xffffffff,count,1));
        if(int64_t(a)+b!=m)return false;
        int tiles_a=((a+BM-1)/BM)*NT,tiles_b=((b+BM-1)/BM)*NT;
        if(task>=tiles_a+tiles_b)return false;
        expert=task<tiles_a?0:1;off=expert?a:0;rows=expert?b:a;
        task-=expert?tiles_a:0;
    } else {
        int64_t prefix=count,tiles=((count+BM-1)/BM)*NT;
        #pragma unroll
        for(int delta=1;delta<32;delta*=2) {
            int64_t prev=__shfl_up_sync(0xffffffff,prefix,delta),pt=__shfl_up_sync(0xffffffff,tiles,delta);
            if(lane>=delta){prefix+=prev;tiles+=pt;}
        }
        if(__shfl_sync(0xffffffff,prefix,experts-1)!=m)return false;
        unsigned mask=__ballot_sync(0xffffffff,lane<experts && task<tiles);
        if(!mask)return false;expert=__ffs(mask)-1;
        rows=int(__shfl_sync(0xffffffff,count,expert));
        off=int(__shfl_sync(0xffffffff,prefix,expert))-rows;
        int end=int(__shfl_sync(0xffffffff,tiles,expert));
        task-=end-((rows+BM-1)/BM)*NT;
    }
    row=task/NT*BM;col=task%NT;return true;
}
template<class T,int N,int K,int BM,int BN,int BK,int WM,int WN,int ST,bool DB,bool CG,int PARTS,int SCHEDULE>
__global__ __launch_bounds__((BM/WM)*(BN/WN)*32) void moe_grouped_gemm_forward_kernel9_decode(
    T* c,float* partial,const T* a,const T* b,const int64_t* counts,int experts,int m,int capacity,bool masked) {
    extern __shared__ __align__(16) unsigned char storage[];
    for(int task=blockIdx.x;;task+=gridDim.x) {
        int e=blockIdx.y,off,rows,row,col;
        if constexpr(SCHEDULE!=0) {
            if(!masked) {
                if(!decode_task<BM,N/BN>(counts,experts,m,task,e,off,rows,row,col))return;
                col*=BN;
            } else {
                if(!expert_bounds(counts,experts,m,e,off,rows,true,capacity))return;
                row=(blockIdx.x/(N/BN))*BM;col=(blockIdx.x%(N/BN))*BN;
            }
        } else {
            if(!expert_bounds(counts,experts,m,e,off,rows,masked,capacity))return;
            row=(blockIdx.x/(N/BN))*BM;col=(blockIdx.x%(N/BN))*BN;
        }
        if(masked && rows>capacity)return;
        if(row>=rows)return;int stride=masked?experts*capacity:m,physical=masked?e*capacity:off;
        int chunk=(K/BK+PARTS-1)/PARTS,first=blockIdx.z*chunk,last=min(K/BK,first+chunk);
        if constexpr(PARTS==1 && SCHEDULE!=0) gg9_tile<T,BM,BN,BK,WM,WN,ST,N,K,T,DB,CG,true,false,true,true>(c+size_t(physical)*N,
            a+size_t(physical)*K,b+size_t(e)*K*N,rows,row,col,reinterpret_cast<T*>(storage));
        else if constexpr(PARTS==1) gg9_tile<T,BM,BN,BK,WM,WN,ST,N,K,T,DB,CG>(c+size_t(physical)*N,
            a+size_t(physical)*K,b+size_t(e)*K*N,rows,row,col,reinterpret_cast<T*>(storage),first,last);
        else gg9_tile<T,BM,BN,BK,WM,WN,ST,N,K,float,DB,CG>(partial+(size_t(blockIdx.z)*stride+physical)*N,
            a+size_t(physical)*K,b+size_t(e)*K*N,rows,row,col,reinterpret_cast<T*>(storage),first,last);
        if constexpr(SCHEDULE!=0) {
            if(masked)return;
            // Finish all shared epilogue reads before reusing storage for the next tile.
            __syncthreads();
        } else return;
    }
}

template<class T,int N,int PARTS> __global__ void reduce_decode(T* c,const float* partial,
    const int64_t* counts,int experts,int m,int capacity,bool masked) {
    int e=blockIdx.y,off,rows;if(!expert_bounds(counts,experts,m,e,off,rows,masked,capacity))return;
    if(masked && rows>capacity)return;int stride=masked?experts*capacity:m,physical=masked?e*capacity:off;
    int index=(blockIdx.x*128+threadIdx.x)*2;
    if(index>=int64_t(rows)*N)return;float x=0,y=0;
    #pragma unroll
    for(int p=0;p<PARTS;++p){float2 v=*reinterpret_cast<const float2*>(partial+(size_t(p)*stride+physical)*N+index);x+=v.x;y+=v.y;}
    gg9_store2(c+size_t(physical)*N+index,x,y);
}
template<class T,int N,int K> void decode_shape(T* c,const T* a,const T* b,const int64_t* counts,
    Metadata* meta,float* partial,int experts,int m,int capacity,int config,int blocks,int order,bool masked,cudaStream_t stream,int mode) {
    auto cf=decode_config(config);if(!m||mode==2)return;
    int rows=masked?capacity:m;
    dim3 grid(unsigned(((int64_t(rows)+cf.bm-1)/cf.bm)*(N/cf.bn)),experts,cf.parts);
    if(cf.schedule && !masked) {
        grid.x=unsigned(((int64_t(m)+cf.bm-1)/cf.bm+std::min(m,experts)-1)*(N/cf.bn));grid.y=1;
        int device,sms;CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
        grid.x=std::min(grid.x,unsigned(2*sms));
    }
    #define LAUNCH(BM,BN,BK,WM,WN,ST,DB,CG,P,SCHEDULE) \
      if constexpr(ST*(BM+BN)*BK*sizeof(T)>48*1024) cudaFuncSetAttribute(moe_grouped_gemm_forward_kernel9_decode<T,N,K,BM,BN,BK,WM,WN,ST,DB,CG,P,SCHEDULE>,cudaFuncAttributeMaxDynamicSharedMemorySize,ST*(BM+BN)*BK*sizeof(T)); \
      moe_grouped_gemm_forward_kernel9_decode<T,N,K,BM,BN,BK,WM,WN,ST,DB,CG,P,SCHEDULE><<<grid,(BM/WM)*(BN/WN)*32,ST*(BM+BN)*BK*sizeof(T),stream>>>(c,partial,a,b,counts,experts,m,capacity,masked); \
      if constexpr(P>1)reduce_decode<T,N,P><<<dim3((int64_t(rows)*N+255)/256,experts),128,0,stream>>>(c,partial,counts,experts,m,capacity,masked)
    switch(config){
        case 0:LAUNCH(16,64,64,16,16,3,0,1,1,0);break;
        case 1:LAUNCH(16,64,64,16,16,3,0,1,4,0);break;
        case 2:LAUNCH(16,64,64,16,16,3,0,1,8,0);break;
        case 3:LAUNCH(16,128,64,16,32,2,1,1,4,0);break;
        case 4:LAUNCH(32,64,64,16,32,3,1,1,1,0);break;
        case 5:LAUNCH(16,128,32,16,32,3,0,0,8,0);break;
        case 6:LAUNCH(32,128,32,16,64,3,0,1,1,0);break;
        case 7:LAUNCH(32,64,64,16,32,3,1,1,1,1);break;
    }
    #undef LAUNCH
}
// Optional GPU-only compact <-> masked conversion. Its cost belongs in timing.
template<class T,int WIDTH,bool PACK> __global__ void decode_relayout(T* dst,const T* src,
    const int64_t* counts,int experts,int m,int capacity) {
    int e=blockIdx.y,off,rows;if(!expert_bounds(counts,experts,m,e,off,rows,false,capacity))return;
    unsigned invalid=__ballot_sync(0xffffffff,(threadIdx.x&31)<experts && counts[threadIdx.x&31]>capacity);
    if(invalid)return;
    int64_t index=int64_t(blockIdx.x)*128+threadIdx.x;if(index>=int64_t(rows)*WIDTH)return;
    int64_t compact=int64_t(off)*WIDTH+index,masked=int64_t(e)*capacity*WIDTH+index;
    if constexpr(PACK)dst[masked]=src[compact];else dst[compact]=src[masked];
}
template<class T,int WIDTH,bool PACK> void relayout(T* dst,const T* src,const int64_t* counts,
    int experts,int m,int capacity,cudaStream_t stream) {
    decode_relayout<T,WIDTH,PACK><<<dim3((int64_t(capacity)*WIDTH+127)/128,experts),128,0,stream>>>(dst,src,counts,experts,m,capacity);
}
template<class T> void launch_decode(T* c,const T* a,const T* b,const int64_t* counts,float* partial,
    int experts,int m,int n,int k,cudaStream_t stream,int config=-1,int capacity=0,bool masked=false,int mode=0) {
    if(config<0)config=decode_default(m,n,k);auto cf=decode_config(config);
    if(!target(experts,n,k)||m<0||m>256||mode<0||mode>2 || (masked && capacity<=0) || (m && cf.parts>1 && !partial))
        throw std::runtime_error("decode supports total M=0..256; invalid shape/workspace");
    Metadata* meta=nullptr;int blocks=0,order=0;
    if(n==2432) decode_shape<T,2432,768>(c,a,b,counts,meta,partial,experts,m,capacity,config,blocks,order,masked,stream,mode);
    else if(n==768) decode_shape<T,768,2432>(c,a,b,counts,meta,partial,experts,m,capacity,config,blocks,order,masked,stream,mode);
    else decode_shape<T,1024,1024>(c,a,b,counts,meta,partial,experts,m,capacity,config,blocks,order,masked,stream,mode);
    auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace grouped9

#ifdef GROUPED_DECODE_MAIN
int main(int argc,char** argv){try {
    auto o=grouped9::cli_options(argc,argv,true);gemm_device_info();
    auto run=[&](auto* c,const auto* a,const auto* b,const int64_t* counts,grouped9::Metadata*,float* partial,int e,int m,int n,int k){grouped9::launch_decode(c,a,b,counts,partial,e,m,n,k,nullptr);};
    return o.dtype=="bf16"?grouped9::cli_run<__nv_bfloat16>(o,true,run):grouped9::cli_run<half>(o,true,run);
}catch(const std::exception& ex){fprintf(stderr,"%s\n",ex.what());return 1;}}
#endif
