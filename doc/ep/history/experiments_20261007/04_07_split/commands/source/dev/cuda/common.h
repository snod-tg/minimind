#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <float.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstddef>

// ---- 1. 类型转换：存储类型 <-> float ----
// as_float/from_float 都标 __host__，因为 CPU 参考也要用它转换 half/bf16。
template <typename T> __host__ __device__ float as_float(T v){return (float)v;}
template <> __host__ __device__ float as_float<half>(half v){return __half2float(v);}
template <> __host__ __device__ float as_float<__nv_bfloat16>(__nv_bfloat16 v){return __bfloat162float(v);}

template <typename T> __host__ __device__ T from_float(float v){return (T)v;}
template <> __host__ __device__ half from_float<half>(float v){return __float2half_rn(v);}
template <> __host__ __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float v){return __float2bfloat16_rn(v);}

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e = (call);                                                   \
        if (e != cudaSuccess) {                                                   \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(e));                                  \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

bool is_aligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}


float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, float* h_ptr, size_t count) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType), cudaMemcpyHostToDevice);
    free(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    return status;
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    CUDA_CHECK(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (int i = 0; i < num_elements; i++) {
        // Skip masked elements
        if(!isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    CUDA_CHECK(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    CUDA_CHECK(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        CUDA_CHECK(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        CUDA_CHECK(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        CUDA_CHECK(cudaEventRecord(stop, nullptr));
        CUDA_CHECK(cudaEventSynchronize(start));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float single_call;
        CUDA_CHECK(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    CUDA_CHECK(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}


// ---- GEMM 共用的分配、参考和计时工具 ----
#define CUBLAS_CHECK(call) do { auto s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s), __FILE__, __LINE__); exit(1); } } while (0)

template<class T> struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count;
    explicit DeviceBuffer(size_t n) : count(n) {
        // 空专家 / 空 token 仍分配一个元素，但不发起零 grid 的 kernel。
        CUDA_CHECK(cudaMalloc(&ptr, std::max(size_t(1), n) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const std::vector<T>& h) {
        if (h.size() != count) throw std::runtime_error("upload size mismatch");
        if (count) CUDA_CHECK(cudaMemcpy(ptr, h.data(), count*sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> h(count);
        if (count) CUDA_CHECK(cudaMemcpy(h.data(), ptr, count*sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
    void poison() { CUDA_CHECK(cudaMemset(ptr, 0xff, std::max(size_t(1), count)*sizeof(T))); }
};

struct BlasHandle {
    cublasHandle_t handle;
    BlasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, cublasMath_t(CUBLAS_DEFAULT_MATH |
            CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    }
    ~BlasHandle() { cublasDestroy(handle); }
};

template<class T> constexpr cudaDataType_t gemm_dtype() {
    if constexpr (std::is_same_v<T, float>) return CUDA_R_32F;
    else if constexpr (std::is_same_v<T, half>) return CUDA_R_16F;
    else return CUDA_R_16BF;
}

// 行主序 C[M,N] = A[M,K] B[K,N]。
// cuBLAS 为列主序：计算 C^T = B^T A^T，交换 A/B 和 M/N；无需实际转置。
template<class T>
void gemm_cublas(
    cublasHandle_t handle,
    T* c,
    const T* a,
    const T* b,
    int m,
    int n,
    int k
){
    if (!m) return;
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha, b, gemm_dtype<T>(), n, a, gemm_dtype<T>(), k,
        &beta, c, gemm_dtype<T>(), n,
        std::is_same_v<T,float> ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template<class T> std::vector<T> gemm_random(
    size_t n,
    unsigned seed,
    bool zeros = false
){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<T> h(n);
    for (auto& v : h) v = from_float<T>(zeros ? 0.f : dist(rng));
    return h;
}

template<class T> bool gemm_close(
    float got,
    double ref
){
    // 拒绝 NaN/Inf，不能让 fabs(NaN) > tolerance 的 false 掩盖错误。
    double atol = 2e-4;
    double rtol = std::is_same_v<T, float> ? 2e-4 : (std::is_same_v<T, half> ? 1e-3 : 8e-3);
    return std::isfinite(got) && std::isfinite(ref) && std::abs(got-ref) <= atol + rtol*std::abs(ref);
}

template<class T> bool gemm_validate(
    const std::vector<T>& got,
    const std::vector<T>& ref,
    const char* label
){
    double max_abs = 0.; size_t failures = 0;
    if (got.size() != ref.size()) return false;
    for (size_t i=0; i<got.size(); ++i) {
        float g = as_float(got[i]), r = as_float(ref[i]);
        max_abs = std::max(max_abs, double(std::abs(g-r)));
        if (!gemm_close<T>(g, r)) {
            if (failures++ < 3) fprintf(stderr, "%s mismatch[%zu]: got=%g ref=%g\n", label, i, g, r);
        }
    }
    printf("CHECK %s elements=%zu max_abs=%.7g failures=%zu\n", label, got.size(), max_abs, failures);
    return !failures;
}

// 小矩阵全量 CPU double 参考；大矩阵每个专家固定抽查 257 个输出。
// CPU 从实际量化后的 T 输入出发，参考输出也舍入到 T。
template<class T> bool gemm_cpu_check(
    const std::vector<T>& a,
    const std::vector<T>& b,
    const std::vector<T>& c,
    int m,
    int n,
    int k,
    size_t aoff = 0,
    size_t boff = 0,
    size_t coff = 0
){
    const size_t total = size_t(m)*n;
    const size_t samples = total*size_t(k) <= 8000000 ? total : std::min(total, size_t(257));
    for (size_t s=0; s<samples; ++s) {
        size_t idx = samples == total ? s : (s * (total-1) / std::max(size_t(1), samples-1));
        size_t row=idx/n, col=idx%n;
        double acc=0.;
        for (int p=0; p<k; ++p) acc += double(as_float(a[aoff+row*k+p])) * as_float(b[boff+size_t(p)*n+col]);
        float ref = as_float(from_float<T>(float(acc)));
        if (!gemm_close<T>(as_float(c[coff+idx]), ref)) {
            fprintf(stderr, "CPU double mismatch at (%zu,%zu): got=%g ref=%g\n", row,col,as_float(c[coff+idx]),ref);
            return false;
        }
    }
    return true;
}

// 预分配、预热后计时，默认热缓存；所有版本都用同一协议。
// gpu_ms 是 CUDA Event 区间，包括全部 kernel 和 host launch 导致的 stream 空档，
// 不等于所有 kernel duration 的简单求和（cuBLAS loop 为多个 launch）；wall_ms 是
// 连续 repeats 次调用再同步的摊销时间，不是单请求的 p50/p99 延迟。
struct GemmTiming { double gpu_ms, wall_ms; };
template<class F> GemmTiming gemm_benchmark(
    F launch,
    int repeats
){
    for (int i=0; i<5; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    auto begin=std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start));
    for (int i=0; i<repeats; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    auto end=std::chrono::steady_clock::now();
    float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return {ms/repeats, std::chrono::duration<double,std::milli>(end-begin).count()/repeats};
}

struct GemmOptions {
    int kernel=0, m=512, n=512, k=512, repeats=30, experts=8, seed=0;
    std::string dtype="fp32", counts, distribution="balanced";
    bool self_test=false, bench=true;
};
inline int gemm_integer(
    const char* text,
    int lo,
    int hi
){
    size_t pos=0; long long value=std::stoll(text,&pos);
    if (pos != std::strlen(text) || value < lo || value > hi) throw std::runtime_error("integer out of range");
    return int(value);
}
inline GemmOptions gemm_options(
    int argc,
    char** argv,
    int max_kernel = 5,
    int default_dimension = 512
){
    GemmOptions o;
    o.m = o.n = o.k = default_dimension;
    for (int i=1;i<argc;++i) {
        std::string key=argv[i];
        if (key=="--self-test") { o.self_test=true; continue; }
        if (key=="--no-bench") { o.bench=false; continue; }
        if (key=="--help") {
            printf("--kernel 0..%d (0=all) --m M --n N --k K --dtype fp32|fp16|bf16\n"
                   "--repeats R --seed S --self-test --no-bench\n"
                   "Grouped: --experts E --distribution balanced|skewed|empty --counts 0,1,17,...\n", max_kernel);
            exit(0);
        }
        if (++i==argc) throw std::runtime_error("missing option value");
        if (key=="--m") o.m=gemm_integer(argv[i],0,INT32_MAX);
        else if (key=="--n") o.n=gemm_integer(argv[i],1,16384);
        else if (key=="--k") o.k=gemm_integer(argv[i],1,16384);
        else if (key=="--kernel") o.kernel=gemm_integer(argv[i],0,max_kernel);
        else if (key=="--repeats") o.repeats=gemm_integer(argv[i],1,10000);
        else if (key=="--experts") o.experts=gemm_integer(argv[i],1,256);
        else if (key=="--seed") o.seed=gemm_integer(argv[i],0,1000000);
        else if (key=="--dtype") o.dtype=argv[i];
        else if (key=="--counts") o.counts=argv[i];
        else if (key=="--distribution") o.distribution=argv[i];
        else throw std::runtime_error("unknown option: "+key);
    }
    if (o.dtype!="fp32" && o.dtype!="fp16" && o.dtype!="bf16") throw std::runtime_error("unsupported dtype");
    // 教学 harness 限制一次测试的工作集，避免参数误输导致占满共享 GPU。
    if ((size_t(o.m)*o.k + size_t(o.m)*o.n + size_t(o.experts)*o.k*o.n)*4 > size_t(6)*1024*1024*1024)
        throw std::runtime_error("test working set exceeds 6 GiB");
    return o;
}
inline void gemm_device_info() {
    int device; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p,device));
    printf("DEVICE %s sm_%d%d; row-major; FP32 accumulation; alpha=1 beta=0; warmup=5; hot-cache\n",p.name,p.major,p.minor);
}

// ---- 动态 Grouped GEMM prefill/decode：SM80+ 共享原语 ----
// DeepGEMM 的分组布局/调度思想；SM89 使用 cp.async + mma.sync。
#if defined(__CUDACC__)
namespace grouped9 {
struct Metadata {int64_t offsets[17],tiles[17];int valid;};
inline bool target(int e,int n,int k) {
    return e>=1 && e<=16 && ((n==2432 && k==768)||(n==768 && k==2432)||(n==1024 && k==1024));
}
template<bool CG> __device__ __forceinline__ void gg9_copy16(void* dst,const void* src,bool valid) {
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dst));int bytes=valid?16:0;
    if constexpr(CG) asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address),"l"(src),"r"(bytes) : "memory");
    else asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(address),"l"(src),"r"(bytes) : "memory");
}
__device__ __forceinline__ void gg9_ldmatrix_a(
    unsigned (&fragment)[4],
    const void* ptr
){
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(fragment[0]),"=r"(fragment[1]),"=r"(fragment[2]),"=r"(fragment[3]) : "r"(address));
}
__device__ __forceinline__ void gg9_ldmatrix_b(
    unsigned (&fragment)[2],
    const void* ptr
){
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(fragment[0]),"=r"(fragment[1]) : "r"(address));
}
template<class T> __device__ __forceinline__ void gg9_mma_16x8x16(
    float (&acc)[4],
    const unsigned (&a)[4],
    const unsigned (&b)[2]
){
    if constexpr (std::is_same_v<T,half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
}
template<class T> __device__ __forceinline__ void gg9_store2(
    T* ptr,
    float x,
    float y
){
    if constexpr (std::is_same_v<T,half>) {
        half2 value=__floats2half2_rn(x,y);
        *reinterpret_cast<half2*>(ptr)=value;
    } else {
        __nv_bfloat162 value=__floats2bfloat162_rn(x,y);
        *reinterpret_cast<__nv_bfloat162*>(ptr)=value;
    }
}

template<int WIDTH>
__device__ __forceinline__ int gg9_shared_index(int row,int col) {
    return (row*WIDTH+col)^((row&7)*8);
}
template<class T,int BM,int BN,int BK,int N=1024,int K=1024,bool CG=true,int THREADS=128,bool REGULAR=false>
__device__ __forceinline__ void gg9_copy(
    const T* a,const T* b,int m,int row0,int col0,int base,T* sa,T* sb,unsigned amask=0xffffffff
){
    #pragma unroll
    for(int index=threadIdx.x;index<BM*BK/8;index+=THREADS) {
        int row=index/(BK/8),p=index%(BK/8)*8;
        bool valid=REGULAR?((amask>>(index/THREADS))&1):(row0+row<m && base+p<K);
        gg9_copy16<false>(sa+gg9_shared_index<BK>(row,p),valid?a+size_t(row0+row)*K+base+p:a,valid);
    }
    #pragma unroll
    for(int index=threadIdx.x;index<BK*BN/8;index+=THREADS) {
        int p=index/(BN/8),col=index%(BN/8)*8;
        bool valid=(REGULAR || base+p<K) && (N%BN==0 || col0+col<N);
        gg9_copy16<CG>(sb+gg9_shared_index<BN>(p,col),valid?b+size_t(base+p)*N+col0+col:b,valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template<class T,int BM,int BN,int BK,int WM,int WN>
__device__ __forceinline__ void gg9_fragments(
    unsigned (&ar)[WM/16][4],unsigned (&br)[WN/8][2],
    const T* sa,const T* sb,int wr,int wc,int p
){
    int lane=threadIdx.x&31;
    #pragma unroll
    for(int i=0;i<WM/16;++i)
        gg9_ldmatrix_a(ar[i],sa+gg9_shared_index<BK>(wr+i*16+(lane&15),p+(lane/16)*8));
    #pragma unroll
    for(int j=0;j<WN/8;j+=2) {
        // 一条 x4.trans 读取两个相邻的 16x8 B fragment，减少 ldmatrix 指令。
        unsigned x[4];unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(
            sb+gg9_shared_index<BN>(p+(lane&15),wc+j*8+(lane/16)*8)));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            :"=r"(x[0]),"=r"(x[1]),"=r"(x[2]),"=r"(x[3]):"r"(address));
        br[j][0]=x[0];br[j][1]=x[1];br[j+1][0]=x[2];br[j+1][1]=x[3];
    }
}

template<int GROUPS>
__device__ __forceinline__ void gg9_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(GROUPS) : "memory");
}
template<class Out> __device__ __forceinline__ void gg9_store(Out* c,float x,float y) {
    if constexpr(std::is_same_v<Out,float>)*reinterpret_cast<float2*>(c)=make_float2(x,y);
    else gg9_store2(c,x,y);
}
template<class T,int BM,int BN,int BK,int WM,int WN,int STAGES,int N=1024,int K=1024,class Out=T,bool DB=false,bool CG=true,bool EPI=false,bool UNROLL=false,bool CROSS=false,bool REGULAR=false>
__device__ __forceinline__ void gg9_tile(
    Out* c,
    const T* a,
    const T* b,
    int m,
    int row0,
    int col0,
    T* storage,
    int first_tile=0,
    int last_tile=K/BK
){
    constexpr int PA=BK,PB=BN,TILES=K/BK,THREADS=(BM/WM)*(BN/WN)*32;
    static_assert(STAGES>=2 && STAGES<=4 && K%BK==0 && N%16==0);
    auto& sa=*reinterpret_cast<T(*)[STAGES][BM][PA]>(storage);
    auto& sb=*reinterpret_cast<T(*)[STAGES][BK][PB]>(storage+STAGES*BM*PA);
    unsigned amask=0;
    if constexpr(REGULAR) {
        #pragma unroll
        for(int copy=0;copy<(BM*BK/8+THREADS-1)/THREADS;++copy)
            if(row0+(threadIdx.x+copy*THREADS)/(BK/8)<m)amask|=1u<<copy;
    }
    int warp=threadIdx.x/32,lane=threadIdx.x&31;
    int wr=warp/(BN/WN)*WM,wc=warp%(BN/WN)*WN;
    float acc[WM/16][WN/8][4]={};
    unsigned ar[2][WM/16][4],br[2][WN/8][2];
    // 预填 STAGES-1 块，等待第一块，保留最新一组拷贝在途。
    #pragma unroll
    for(int stage=0;stage<STAGES-1;++stage)
        gg9_copy<T,BM,BN,BK,N,K,CG,THREADS,REGULAR>(a,b,m,row0,col0,(first_tile+stage)*BK,&sa[stage][0][0],&sb[stage][0][0],amask);
    gg9_wait<STAGES-2>();__syncthreads();
    if constexpr(CROSS) {
        // 跨 shared stage 的寄存器预取：屏障后先加载下一块，再计算当前最后一组。
        // 最后一次 shared 读取发生在屏障前，屏障后的 MMA 只读寄存器。
        int pool=0;
        gg9_fragments<T,BM,BN,BK,WM,WN>(ar[0],br[0],&sa[0][0][0],&sb[0][0][0],wr,wc,0);
        for(int tile=first_tile;tile<last_tile;++tile) {
            int next=tile+STAGES-1,copy_pool=pool?pool-1:STAGES-1;
            if(next<last_tile)gg9_copy<T,BM,BN,BK,N,K,CG,THREADS,REGULAR>(a,b,m,row0,col0,next*BK,&sa[copy_pool][0][0],&sb[copy_pool][0][0],amask);
            int next_pool=pool+1==STAGES?0:pool+1;
            #pragma unroll
            for(int p=0;p<BK;p+=16) {
                int slot=(p/16)&1;
                if(p+16<BK)gg9_fragments<T,BM,BN,BK,WM,WN>(ar[slot^1],br[slot^1],&sa[pool][0][0],&sb[pool][0][0],wr,wc,p+16);
                else if(tile+1<last_tile) {
                    if(next<last_tile)gg9_wait<STAGES-2>();else gg9_wait<0>();
                    __syncthreads();
                    gg9_fragments<T,BM,BN,BK,WM,WN>(ar[slot^1],br[slot^1],&sa[next_pool][0][0],&sb[next_pool][0][0],wr,wc,0);
                }
                #pragma unroll
                for(int i=0;i<WM/16;++i) {
                    #pragma unroll
                    for(int j=0;j<WN/8;++j)gg9_mma_16x8x16<T>(acc[i][j],ar[slot][i],br[slot][j]);
                }
            }
            pool=next_pool;
        }
    } else {
    auto step=[&](int tile,int pool) {
        int next=tile+STAGES-1,copy_pool=pool?pool-1:STAGES-1;
        if(next<last_tile)
            gg9_copy<T,BM,BN,BK,N,K,CG,THREADS,REGULAR>(a,b,m,row0,col0,next*BK,&sa[copy_pool][0][0],&sb[copy_pool][0][0],amask);
        // DB 只切换操作数寄存器；累加仍按 K 的原顺序执行。
        if constexpr(DB) gg9_fragments<T,BM,BN,BK,WM,WN>(ar[0],br[0],&sa[pool][0][0],&sb[pool][0][0],wr,wc,0);
        #pragma unroll
        for(int p=0;p<BK;p+=16) {
            constexpr int STEPS=BK/16;
            int slot=DB?(p/16)%2:0;
            if constexpr(DB) {
                if(p+16<BK) gg9_fragments<T,BM,BN,BK,WM,WN>(ar[slot^1],br[slot^1],&sa[pool][0][0],&sb[pool][0][0],wr,wc,p+16);
            } else gg9_fragments<T,BM,BN,BK,WM,WN>(ar[0],br[0],&sa[pool][0][0],&sb[pool][0][0],wr,wc,p);
            #pragma unroll
            for(int i=0;i<WM/16;++i) {
                #pragma unroll
                for(int j=0;j<WN/8;++j) gg9_mma_16x8x16<T>(acc[i][j],ar[slot][i],br[slot][j]);
            }
        }
        if(tile+1<last_tile) {
            // 收尾必须排空：没有新提交时 wait_group 1 可能留下最后一块未完成。
            if(next<last_tile)gg9_wait<STAGES-2>();
            else gg9_wait<0>();
            __syncthreads();
        }
    };
    if constexpr(UNROLL) {
        // 按流水线深度展开：每段 shared pool 成为编译期常量，减少地址计算。
        for(int batch=first_tile;batch<last_tile;batch+=STAGES) {
            #pragma unroll
            for(int pool=0;pool<STAGES;++pool) if(batch+pool<last_tile)step(batch+pool,pool);
        }
    } else {
        int pool=0;
        for(int tile=first_tile;tile<last_tile;++tile){step(tile,pool);if(++pool==STAGES)pool=0;}
    }
    }
    if constexpr(EPI) {
        static_assert(std::is_same_v<Out,T> && STAGES*(BM+BN)*BK>=BM*BN);
        // 复用已消费完的 shared，重排 MMA 输出，使每次全局写入为连续16B。
        gg9_wait<0>();__syncthreads();
        #pragma unroll
        for(int i=0;i<WM/16;++i) {
            #pragma unroll
            for(int j=0;j<WN/8;++j) {
                int row=wr+i*16+lane/4,col=wc+j*8+(lane%4)*2;
                gg9_store2(storage+gg9_shared_index<BN>(row,col),acc[i][j][0],acc[i][j][1]);
                gg9_store2(storage+gg9_shared_index<BN>(row+8,col),acc[i][j][2],acc[i][j][3]);
            }
        }
        __syncthreads();
        for(int index=threadIdx.x;index<BM*BN/8;index+=THREADS) {
            int row=index/(BN/8),col=index%(BN/8)*8;
            if(row0+row<m && col0+col<N) *reinterpret_cast<uint4*>(c+size_t(row0+row)*N+col0+col)=
                *reinterpret_cast<const uint4*>(storage+gg9_shared_index<BN>(row,col));
        }
    } else {
    #pragma unroll
    for(int i=0;i<WM/16;++i) {
        #pragma unroll
        for(int j=0;j<WN/8;++j) {
            int row=row0+wr+i*16+lane/4,col=col0+wc+j*8+(lane%4)*2;
            if(row<m && col<N)gg9_store(c+size_t(row)*N+col,acc[i][j][0],acc[i][j][1]);
            if(row+8<m && col<N)gg9_store(c+size_t(row+8)*N+col,acc[i][j][2],acc[i][j][3]);
        }
    }
    }
}

// 每个 warp 独立读取最多16个GPU counts；无共享元数据或CPU回读。
// 非法 counts 时所有 warp 得到同一 valid，整块退出，不触碰矩阵。
__device__ __forceinline__ bool expert_bounds(const int64_t* counts,int experts,int m,int e,int& off,int& rows,bool masked=false,int capacity=INT32_MAX) {
    int lane=threadIdx.x&31;
    if(experts<=2) {
        int64_t a=__shfl_sync(0xffffffff,lane==0?counts[0]:int64_t(0),0);
        int64_t b=experts==2?__shfl_sync(0xffffffff,lane==1?counts[1]:int64_t(0),1):0;
        off=e?int(a):0;rows=int(e?b:a);
        bool valid=a>=0 && b>=0 && a<=m && b<=m && (!masked || (a<=capacity && b<=capacity));
        return valid && (masked?a+b<=m:a+b==m);
    }
    int64_t value=lane<experts?counts[lane]:0;
    int bad=value<0 || value>m || (masked && value>capacity);unsigned invalid=__ballot_sync(0xffffffff,bad);
    int64_t sum=bad?0:value;
    #pragma unroll
    for(int step=1;step<32;step*=2) {int64_t x=__shfl_up_sync(0xffffffff,sum,step);if(lane>=step)sum+=x;}
    int64_t total=__shfl_sync(0xffffffff,sum,31);
    off=e?int(__shfl_sync(0xffffffff,sum,e-1)):0;
    rows=int(__shfl_sync(0xffffffff,value,e));
    return !invalid && (masked?total<=m:total==m);
}
// 教学 CLI 共用参数解析、cuBLAS 对拍和 CPU double 抽查。
inline GemmOptions cli_options(int argc,char** argv,bool decode) {
    auto o=gemm_options(argc,argv,1,1024);bool has_m=false,has_dtype=false;
    for(int i=1;i<argc;++i){has_m|=std::string(argv[i])=="--m";has_dtype|=std::string(argv[i])=="--dtype";}
    if(!has_m)o.m=decode?16:2048;if(!has_dtype)o.dtype="bf16";
    if(o.dtype=="fp32")throw std::runtime_error("prefill/decode support fp16/bf16");return o;
}
template<class T,class Launch> int cli_run(const GemmOptions& o,bool decode,Launch launch) {
    std::vector<int64_t> counts;
    if(o.counts.empty()) {
        counts.resize(o.experts);for(int e=0;e<o.experts;++e)counts[e]=o.m/o.experts+(e<o.m%o.experts);
        if(o.distribution=="skewed"){counts.assign(o.experts,0);counts[0]=o.m;}
        else if(o.distribution=="empty")std::fill(counts.begin(),counts.end(),0);
        else if(o.distribution!="balanced")throw std::runtime_error("unknown distribution");
    }else {
        size_t start=0;while(start<o.counts.size()) {
            size_t end=o.counts.find(',',start);std::string text=o.counts.substr(start,end-start);
            counts.push_back(gemm_integer(text.c_str(),0,INT32_MAX));if(end==std::string::npos)break;start=end+1;
        }
        if(counts.empty()||o.counts.back()==',')throw std::runtime_error("invalid counts");
    }
    int64_t total=0;for(auto x:counts)total+=x;
    if(total>INT32_MAX || !target(int(counts.size()),o.n,o.k) || (decode && total>256))throw std::runtime_error("unsupported counts/shape");
    int m=int(total),e=int(counts.size());
    auto a=gemm_random<T>(size_t(m)*o.k,o.seed+17),b=gemm_random<T>(size_t(e)*o.k*o.n,o.seed+43);
    DeviceBuffer<T> da(a.size()),db(b.size()),dc(size_t(m)*o.n),dr(dc.count);DeviceBuffer<int64_t> ct(counts.size());
    DeviceBuffer<Metadata> meta(1);DeviceBuffer<float> partial(decode?size_t(8)*m*o.n:0);
    da.upload(a);db.upload(b);ct.upload(counts);dc.poison();BlasHandle blas;size_t offset=0;
    for(int i=0;i<e;++i){gemm_cublas(blas.handle,dr.ptr+offset*o.n,da.ptr+offset*o.k,db.ptr+size_t(i)*o.k*o.n,int(counts[i]),o.n,o.k);offset+=counts[i];}
    auto run=[&](){launch(dc.ptr,da.ptr,db.ptr,ct.ptr,meta.ptr,partial.ptr,e,m,o.n,o.k);};run();auto out=dc.download();
    if(!gemm_validate(out,dr.download(),decode?"kernel9_decode":"kernel9_prefill"))throw std::runtime_error("cuBLAS mismatch");offset=0;
    for(int i=0;i<e;++i){if(!gemm_cpu_check(a,b,out,int(counts[i]),o.n,o.k,offset*o.k,size_t(i)*o.k*o.n,offset*o.n))throw std::runtime_error("CPU double mismatch");offset+=counts[i];}
    if(o.bench && m){auto result=gemm_benchmark(run,o.repeats);printf("TIME complete_call_us=%.4f M=%d N=%d K=%d E=%d\n",result.gpu_ms*1000,m,o.n,o.k,e);}
    puts("ALL CHECKS PASSED");return 0;
}
} // namespace grouped9
#endif // __CUDACC__
