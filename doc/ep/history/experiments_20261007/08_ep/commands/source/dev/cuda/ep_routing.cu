#include "common.h"
#include <cassert>

// Stable small-bin partition. Every CTA counts 256 assignments; ballot gives the
// position inside its bin without atomics. Stability preserves expert GEMM rows.
template<int Bins> __global__ void count_positions(const int64_t* ids,int n,int divisor,int* positions,int* block_counts) {
    __shared__ int warp_counts[Bins][8];
    int i=blockIdx.x*256+threadIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int bin=i<n?ids[i]/divisor:-1;
    unsigned masks[Bins];
    for(int e=0;e<Bins;++e){masks[e]=__ballot_sync(0xffffffff,bin==e);if(lane==0)warp_counts[e][warp]=__popc(masks[e]);}
    __syncthreads();
    if(i<n){int before=0;for(int w=0;w<warp;++w)before+=warp_counts[bin][w];positions[i]=before+__popc(masks[bin]&(lane==0?0:(0xffffffffu>>(32-lane))));}
    if(threadIdx.x==0)for(int e=0;e<Bins;++e){int total=0;for(int w=0;w<8;++w)total+=warp_counts[e][w];block_counts[e*gridDim.x+blockIdx.x]=total;}
}
template<int Bins> __global__ void prefix_counts(const int* blocks,int* offsets,int64_t* counts,int num_blocks) {
    __shared__ int scan[256];
    for(int e=0;e<Bins;++e){
        int carry=0;
        for(int base=0;base<num_blocks;base+=256){
            int b=base+threadIdx.x,own=b<num_blocks?blocks[e*num_blocks+b]:0;
            scan[threadIdx.x]=own;__syncthreads();
            for(int d=1;d<256;d*=2){int add=threadIdx.x>=d?scan[threadIdx.x-d]:0;__syncthreads();scan[threadIdx.x]+=add;__syncthreads();}
            if(b<num_blocks)offsets[e*num_blocks+b]=carry+scan[threadIdx.x]-own;
            carry+=scan[255];__syncthreads();
        }
        if(threadIdx.x==0)counts[e]=carry;
        __syncthreads();
    }
}
template<int Bins> __global__ void scatter_maps(const int64_t* ids,int n,int divisor,const int* positions,const int* offsets,
                             const int64_t* counts,int64_t* perm,int64_t* inverse) {
    int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;
    int bin=ids[i]/divisor,base=0;for(int e=0;e<bin;++e)base+=counts[e];
    int dst=base+offsets[bin*gridDim.x+blockIdx.x]+positions[i];perm[dst]=i;inverse[i]=dst;
}
// One CTA completes count -> prefix -> scatter for small Decode batches.
template<int Bins,bool Masked=false> __global__ void sort_small(const int64_t* ids,int n,int divisor,int64_t* perm,int64_t* inverse,int64_t* counts) {
    __shared__ int warp_counts[Bins][8];
    int i=threadIdx.x,lane=i&31,warp=i>>5;
    int bin=i<n && (!Masked || ids[i]>=0)?ids[i]/divisor:-1;
    unsigned masks[Bins];
    for(int e=0;e<Bins;++e){masks[e]=__ballot_sync(0xffffffff,bin==e);if(lane==0)warp_counts[e][warp]=__popc(masks[e]);}
    __syncthreads();
    int base=0,before=0;
    for(int e=0;e<Bins;++e){
        int total=0;for(int w=0;w<8;++w){total+=warp_counts[e][w];if(e==bin && w<warp)before+=warp_counts[e][w];}
        if(e<bin)base+=total;
        if(i==0)counts[e]=total;
    }
    if(i<n){if(bin>=0){int dst=base+before+__popc(masks[bin]&(lane==0?0:(0xffffffffu>>(32-lane))));perm[dst]=i;inverse[i]=dst;}else inverse[i]=-1;}
}

template<class T> __global__ void pack_rows(const T* x, const int64_t* ids,
                          const int64_t* perm, T* out, int64_t* local_ids,
                          int rows, int hidden, int topk, int experts) {
    int row=blockIdx.x;
    if(row>=rows) return;
    int src=perm[row];
    // Full 16-byte vectors when rows are aligned, scalar tail otherwise.
    if(hidden*sizeof(T)%16==0) {
        auto* a=reinterpret_cast<const uint4*>(x+(src/topk)*hidden);
        auto* b=reinterpret_cast<uint4*>(out+row*hidden);
        for(int j=threadIdx.x;j<hidden*sizeof(T)/16;j+=blockDim.x) b[j]=a[j];
    } else {
        for(int j=threadIdx.x;j<hidden;j+=blockDim.x) out[row*hidden+j]=x[(src/topk)*hidden+j];
    }
    if(threadIdx.x==0) local_ids[row]=ids[src]%experts;
}

template<class T> __global__ void gather_rows(const T* x,const int64_t* map,T* out,int rows,int hidden) {
    int row=blockIdx.x;
    if(hidden*sizeof(T)%16==0) {
        auto* a=reinterpret_cast<const uint4*>(x+map[row]*hidden);
        auto* b=reinterpret_cast<uint4*>(out+row*hidden);
        for(int j=threadIdx.x;j<hidden*sizeof(T)/16;j+=blockDim.x) b[j]=a[j];
    } else {
        for(int j=threadIdx.x;j<hidden;j+=blockDim.x) out[row*hidden+j]=x[map[row]*hidden+j];
    }
}

template<class T,class W,class O> __global__ void combine_rows(const T* returned,const W* weight,
                       const int64_t* inverse,O* out,int tokens,int hidden,int topk) {
    int token=blockIdx.x;
    for(int h=threadIdx.x;h<hidden;h+=blockDim.x) {
        // Explicit casts preserve original weighted_assignment.to(x.dtype),
        // followed by BF16 index_add. Never fuse product with accumulation.
        O sum=from_float<O>(0.f);
        for(int k=0;k<topk;++k) {
            int a=token*topk+k;
            float value=__fmul_rn(as_float(returned[inverse[a]*hidden+h]),as_float(weight[a]));
            if constexpr(std::is_same<T,W>::value) value=as_float(from_float<T>(value));
            O product=from_float<O>(value);
            sum=from_float<O>(as_float(sum)+as_float(product));
        }
        out[token*hidden+h]=sum;
    }
}

template<int Bins> void launch_sort(const int64_t* ids,int n,int divisor,int64_t* perm,int64_t* inverse,int64_t* counts,int* positions,int* blocks,int* offsets,cudaStream_t stream) {
    if(n<=256){sort_small<Bins><<<1,256,0,stream>>>(ids,n,divisor,perm,inverse,counts);return;}
    int grid=(n+255)/256;
    count_positions<Bins><<<grid,256,0,stream>>>(ids,n,divisor,positions,blocks);
    prefix_counts<Bins><<<1,256,0,stream>>>(blocks,offsets,counts,grid);
    scatter_maps<Bins><<<grid,256,0,stream>>>(ids,n,divisor,positions,offsets,counts,perm,inverse);
}
// Runtime-bin path: warp match identifies equal expert IDs without a loop
// over all bins. Only warp leaders publish counts, so atomics are unnecessary.
__global__ void count_runtime(const int64_t* ids,int n,int divisor,int bins,bool masked,int* positions,int* blocks) {
    extern __shared__ int warps[];
    for(int j=threadIdx.x;j<bins*8;j+=256)warps[j]=0;
    __syncthreads();
    int i=blockIdx.x*256+threadIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int bin=i<n && (!masked || ids[i]>=0)?ids[i]/divisor:-1;
    assert(bin>=-1 && bin<bins);
    unsigned peers=__match_any_sync(0xffffffff,bin);
    if(bin>=0 && lane==__ffs(peers)-1)warps[bin*8+warp]=__popc(peers);
    __syncthreads();
    if(i<n){
        if(bin>=0){int before=0;for(int w=0;w<warp;++w)before+=warps[bin*8+w];positions[i]=before+__popc(peers&(lane==0?0:(0xffffffffu>>(32-lane))));}
        else positions[i]=-1;
    }
    for(int e=threadIdx.x;e<bins;e+=256){int total=0;for(int w=0;w<8;++w)total+=warps[e*8+w];blocks[e*gridDim.x+blockIdx.x]=total;}
}
__global__ void prefix_runtime(const int* blocks,int* offsets,int64_t* counts,int num_blocks) {
    __shared__ int scan[256];int e=blockIdx.x,carry=0;
    for(int base=0;base<num_blocks;base+=256){
        int b=base+threadIdx.x,own=b<num_blocks?blocks[e*num_blocks+b]:0;
        scan[threadIdx.x]=own;__syncthreads();
        for(int d=1;d<256;d*=2){int add=threadIdx.x>=d?scan[threadIdx.x-d]:0;__syncthreads();scan[threadIdx.x]+=add;__syncthreads();}
        if(b<num_blocks)offsets[e*num_blocks+b]=carry+scan[threadIdx.x]-own;
        carry+=scan[255];__syncthreads();
    }
    if(threadIdx.x==0)counts[e]=carry;
}
__global__ void scatter_runtime(const int64_t* ids,int n,int divisor,const int* positions,const int* offsets,const int64_t* counts,int64_t* perm,int64_t* inv){
    int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;
    if(positions[i]<0){inv[i]=-1;return;}
    int e=ids[i]/divisor,base=0;for(int b=0;b<e;++b)base+=counts[b];
    int dst=base+offsets[e*gridDim.x+blockIdx.x]+positions[i];perm[dst]=i;inv[i]=dst;
}
__global__ void small_runtime(const int64_t* ids,int n,int divisor,int bins,bool masked,int64_t* perm,int64_t* inv,int64_t* counts){
    extern __shared__ int warps[];
    for(int j=threadIdx.x;j<bins*8;j+=256)warps[j]=0;
    __syncthreads();
    int i=threadIdx.x,lane=i&31,warp=i>>5;
    int bin=i<n && (!masked || ids[i]>=0)?ids[i]/divisor:-1;
    assert(bin>=-1 && bin<bins);
    unsigned peers=__match_any_sync(0xffffffff,bin);
    if(bin>=0 && lane==__ffs(peers)-1)warps[bin*8+warp]=__popc(peers);
    __syncthreads();
    for(int e=i;e<bins;e+=256){int total=0;for(int w=0;w<8;++w)total+=warps[e*8+w];counts[e]=total;}
    __syncthreads();
    if(i<n){
        if(bin>=0){int base=0,before=0;for(int e=0;e<bin;++e)base+=counts[e];for(int w=0;w<warp;++w)before+=warps[bin*8+w];int dst=base+before+__popc(peers&(lane==0?0:(0xffffffffu>>(32-lane))));perm[dst]=i;inv[i]=dst;}
        else inv[i]=-1;
    }
}
void launch_runtime_sort(const int64_t* ids,int n,int divisor,int bins,bool masked,int64_t* perm,int64_t* inv,int64_t* counts,int* pos,int* blocks,int* offsets,cudaStream_t stream){
    if(n<=256){small_runtime<<<1,256,bins*8*sizeof(int),stream>>>(ids,n,divisor,bins,masked,perm,inv,counts);return;}
    int grid=(n+255)/256;
    count_runtime<<<grid,256,bins*8*sizeof(int),stream>>>(ids,n,divisor,bins,masked,pos,blocks);
    prefix_runtime<<<bins,256,0,stream>>>(blocks,offsets,counts,grid);
    scatter_runtime<<<grid,256,0,stream>>>(ids,n,divisor,pos,offsets,counts,perm,inv);
}
extern "C" void ep_sort_masked(const int64_t* ids,int n,int bins,int64_t* perm,int64_t* inv,int64_t* counts,int* pos,int* blocks,int* offsets,cudaStream_t stream){
    launch_runtime_sort(ids,n,1,bins,true,perm,inv,counts,pos,blocks,offsets,stream);
}
extern "C" void ep_sort(const int64_t* ids,int n,int divisor,int bins,int64_t* perm,int64_t* inverse,int64_t* counts,int* positions,int* blocks,int* offsets,cudaStream_t stream) {
    if(bins==2)launch_sort<2>(ids,n,divisor,perm,inverse,counts,positions,blocks,offsets,stream);
    else if(bins==3)launch_sort<3>(ids,n,divisor,perm,inverse,counts,positions,blocks,offsets,stream);
    else launch_runtime_sort(ids,n,divisor,bins,false,perm,inverse,counts,positions,blocks,offsets,stream);
}
#define EP_TYPES(OP) if(dtype==0) {OP(float);} else if(dtype==1) {OP(half);} else {OP(__nv_bfloat16);}
extern "C" void ep_pack(const void* x,const int64_t* ids,const int64_t* perm,void* out,
                        int64_t* local,int rows,int hidden,int topk,int experts,int dtype,cudaStream_t stream) {
    if(!rows)return;
#define PACK(T) pack_rows<<<rows,128,0,stream>>>((const T*)x,ids,perm,(T*)out,local,rows,hidden,topk,experts)
    EP_TYPES(PACK)
#undef PACK
}
extern "C" void ep_gather(const void* x,const int64_t* map,void* out,int rows,int hidden,int dtype,cudaStream_t stream) {
    if(!rows)return;
#define GATHER(T) gather_rows<<<rows,128,0,stream>>>((const T*)x,map,(T*)out,rows,hidden)
    EP_TYPES(GATHER)
#undef GATHER
}
template<class T,class W> void launch_combine(const void* x,const void* weights,const int64_t* map,void* out,int tokens,int hidden,int topk,int outtype,cudaStream_t stream) {
#define OUT(O) combine_rows<<<tokens,128,0,stream>>>((const T*)x,(const W*)weights,map,(O*)out,tokens,hidden,topk)
    if(outtype==0){OUT(float);}else if(outtype==1){OUT(half);}else{OUT(__nv_bfloat16);}
#undef OUT
}
extern "C" void ep_combine(const void* x,const void* weights,const int64_t* map,void* out,
                            int tokens,int hidden,int topk,int dtype,int wtype,int outtype,cudaStream_t stream) {
    if(!tokens)return;
#define COMB(T) if(wtype==0) launch_combine<T,float>(x,weights,map,out,tokens,hidden,topk,outtype,stream); else if(wtype==1) launch_combine<T,half>(x,weights,map,out,tokens,hidden,topk,outtype,stream); else launch_combine<T,__nv_bfloat16>(x,weights,map,out,tokens,hidden,topk,outtype,stream)
    EP_TYPES(COMB)
#undef COMB
}

// Fixed-capacity Decode: no host counts are needed to launch NCCL. Padding
// carries local_expert_id=-1; the receiving GPU computes valid expert counts.
template<class T> __global__ void pack_fixed_rows(const T* x,const int64_t* ids,const int64_t* perm,const int64_t* counts,
                  T* out,int64_t* local,int64_t* inverse,int hidden,int topk,int capacity,int experts) {
    int row=blockIdx.x,rank=row/capacity,slot=row%capacity;
    int start=0;for(int r=0;r<rank;++r)start+=counts[r];
    bool valid=slot<counts[rank];int src=valid?perm[start+slot]:0;
    for(int h=threadIdx.x;h<hidden;h+=blockDim.x)out[row*hidden+h]=valid?x[(src/topk)*hidden+h]:from_float<T>(0.f);
    if(threadIdx.x==0){local[row]=valid?ids[src]%experts:-1;if(valid)inverse[src]=row;}
}
template<class T> __global__ void scatter_fixed_rows(const T* x,const int64_t* inverse,T* out,int rows,int hidden) {
    int row=blockIdx.x;int src=inverse[row];
    for(int h=threadIdx.x;h<hidden;h+=blockDim.x)out[row*hidden+h]=src>=0?x[src*hidden+h]:from_float<T>(0.f);
}
extern "C" void ep_masked_sort(const int64_t* ids,int n,int bins,int64_t* perm,int64_t* inv,int64_t* counts,cudaStream_t stream){
    if(bins==2)sort_small<2,true><<<1,256,0,stream>>>(ids,n,1,perm,inv,counts);
    else if(bins==3)sort_small<3,true><<<1,256,0,stream>>>(ids,n,1,perm,inv,counts);
    else small_runtime<<<1,256,bins*8*sizeof(int),stream>>>(ids,n,1,bins,true,perm,inv,counts);
}
extern "C" void ep_pack_fixed(const void* x,const int64_t* ids,const int64_t* perm,const int64_t* counts,void* out,int64_t* local,int64_t* inv,int hidden,int topk,int capacity,int ranks,int experts,int dtype,cudaStream_t stream){
#define PACK_FIXED(T) pack_fixed_rows<<<ranks*capacity,128,0,stream>>>((const T*)x,ids,perm,counts,(T*)out,local,inv,hidden,topk,capacity,experts)
    EP_TYPES(PACK_FIXED)
#undef PACK_FIXED
}
extern "C" void ep_scatter_fixed(const void* x,const int64_t* inv,void* out,int rows,int hidden,int dtype,cudaStream_t stream){
#define SCATTER_FIXED(T) scatter_fixed_rows<<<rows,128,0,stream>>>((const T*)x,inv,(T*)out,rows,hidden)
    EP_TYPES(SCATTER_FIXED)
#undef SCATTER_FIXED
}
