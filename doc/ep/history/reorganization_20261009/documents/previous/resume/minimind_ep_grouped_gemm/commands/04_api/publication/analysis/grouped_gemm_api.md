# Kernel9：PyTorch 推理接口

## 注册与构建结果
- `my_cuda_kernels::grouped_gemm(x, packed_b, counts) -> y`：自动分配输出/临时工作区。
- `my_cuda_kernels::grouped_gemm_out(x, packed_b, counts, out, metadata, partial) -> None`：调用者预分配；三个写入张量均在 schema 中声明可变，检查它们不与输入或彼此重叠。
- CUDA implementation 用输入设备的当前 stream，CUDAGuard 恢复原设备；输入 BF16/FP16、CUDA 连续、地址对齐，禁止 requires_grad；没有新增主机同步和 GPU counts 读回。
- Python 包装含 FakeTensor、明确报错的 backward 注册、同步调试 counts 检查、工作区与专家权重缓存。

当前环境缺 Ninja。R02/R05 改用安装好的 GCC12.4/nvcc12.4 直接编译，链接 torch 2.9.0+cu128 的 C++11 ABI=true。最终 R05 库 SHA256：`4154552b563c947f708254d30da9284a9a6b18b230affcca07746e91b31ac6b3`。实际三条编译/链接命令由构建器记录，未安装新依赖；更换 PyTorch ABI 后重建。

metadata 大小为 280 字节；普通路径 partial 为空，小 M Split-K 的 partial 为 FP32。M=1 down 的八分区工作区为 24576 字节，M=16 gate/up 的四分区工作区为 622592 字节；另有输出张量。自动调用的临时张量由 PyTorch 当前 stream 分配器管理；预分配工作区由调用者持有，不能在无依赖的并发 stream 中复用。

## 正确性与接口边界
R06 的 100 条成功检查和 23 条预期拒绝全部通过：

- 72 个输入案例，每个调用自动/预分配两种接口，共 144 次逐元素对拍。覆盖 FP16/BF16、三组 N/K、E=1/2/16、M=0～65536、单专家 M 至 32769、空专家、集中分配与尾块；部分案例另做 CPU double 抽查。容差沿用阶段 03：atol=2e-4，rtol=0.001/0.008，参考 FP32 matmul 关闭 TF32 后舍入。
- 同一 Graph 中改变 GPU counts，两个总 M、两种接口、四种分配，共 16 次结果验证通过。M 保持固定，改变总 M 需重新准备输出/工作区或捕获。
- 非默认 stream、显式跨 stream 等待、输入设备与当前设备不同、调用后设备恢复均通过。
- functional opcheck 的 schema、autograd_registration、faketensor、aot_dispatch_dynamic 四项 SUCCESS；out opcheck 的 schema/faketensor 两项 SUCCESS。
- `torch.compile(backend="eager", fullgraph=True, dynamic=True)` 的 M=1/65/129 验证通过。仅证明该接口的 Dynamo 动态形状路径，没有验证 Inductor 优化、完整模型编译或训练。
- CPU 输入、device/dtype/shape/stride 错误、输入地址不对齐、输出与输入 alias、工作区不足/不对齐、梯度请求等均按预期拒绝。counts 的负值、总数不匹配和极大值仅在显式同步调试选项中报错。
- fast path 的非法 GPU counts 触发原 kernel 防访问分支，预先填充的 NaN 输出保持不变；未承诺对非法 counts 同步报 Python 错误。输入数值合法性仍是正常调用契约。
- profiler 的预分配调用没有 aten::item、_local_scalar_dense 或 memcpy 事件。另在当前 stream 排入延迟，host 调用仅约 29.0 μs，返回时 GPU event 尚未完成；证明本次测试没有等待 GPU 计算完成。

## 权重缓存
以同维度的模拟 FP32 专家模块验证缓存。指定专家顺序 [1,0] 的打包结果正确，FP32 源权重未改动；每层两专家三投影缓存为 22413312 字节，即 21.375 MiB。显式 invalidate 后访问拒绝；重建反映更新值；进入训练或替换参数对象后访问拒绝。

原地权重值更新必须显式失效，参数身份/地址/device/dtype 的检查无法发现所有原地更新，不依赖 inference tensor 的版本号。尚未把此缓存接到真实模型加载/迁移回调，阶段 05/06 继续验证。

## 接口内存检查
R08 Compute Sanitizer memcheck：5 个输入案例、两种 API、非默认 stream，均数值通过，ERROR SUMMARY 为 0。覆盖 BF16 两组模型维度、Split-K、常规路径、FP16 E=16 和 M=0。未改共享内存计算核心，阶段 03 已完成的 memcheck/racecheck/synccheck 共 22 组结果仍作为计算核心验证；本轮不声称重新执行了全部 22 组。

## 同轮接口性能
五个负载使用相同随机矩阵、counts、dtype 和 kernel9 默认路径。Native 是阶段 03 的 C ABI 基准入口；三个路径都含 GPU metadata 和完整计算，不是 CUTLASS 对比。seed=20261007，RTX 4090D 物理 GPU 1，不锁频、预热和热缓存。

CUDA Graph：每图 64 次调用，7 轮交替，每轮 10 次 replay，中位数。捕获和 Python/主机属性检查不在 replay 计时内；自动分配的 Graph 私有工作区在捕获时建立。

| 负载 | Native C ABI μs | PyTorch 预分配 μs | PyTorch 自动分配 μs |
| --- | ---: | ---: | ---: |
| Decode M=1，down | 7.118 | 7.130 | 7.107 |
| Decode M=16，gate/up 单次 | 6.398 | 6.427 | 6.437 |
| Prefill M=2912，down | 100.338 | 100.395 | 100.437 |
| Prefill M=22463，gate/up 单次 | 657.978 | 657.446 | 660.150 |
| E=16，FP16 M=2048 | 47.723 | 47.678 | 47.579 |

Eager wall：预热后连续调用 64 次，末尾同步，7 轮交替，中位数摊到每次。包含 CPU 提交、接口检查、自动路径的张量分配和 GPU 执行；仅在批次末尾同步，不代表逐个请求独立同步的单请求延迟。Event 数据亦包含 CPU 提交空隙，不能用作纯计算耗时。

| 负载 | Native C ABI μs | PyTorch 预分配 μs | PyTorch 自动分配 μs |
| --- | ---: | ---: | ---: |
| Decode M=1，down | 12.026 | 9.366 | 13.540 |
| Decode M=16，gate/up 单次 | 11.671 | 9.197 | 13.368 |
| Prefill M=2912，down | 101.422 | 101.402 | 101.408 |
| Prefill M=22463，gate/up 单次 | 662.433 | 664.985 | 661.276 |
| E=16，FP16 M=2048 | 49.190 | 48.990 | 49.287 |

Graph 下接口路径与 native/full 接近；小 M eager 下预分配路径约 9.2～9.4 μs，自动分配约 13.4～13.5 μs。两条路径的检查和分配行为不同，差值不能全部归因于 allocator。大 M 时计算成本占主导，几个入口的小幅差异不足以证明某入口更快；不跨轮与阶段 03 的 CUTLASS 数值拼接加速比。

![接口调用成本](../figures/grouped_gemm_api.png)


## 构建与调用示例

在安装当前 PyTorch/CUDA 的 Linux GPU 环境中执行，GCC12 路径按本机环境设置。构建入口显式读取当前 torch ABI/include/lib，不依赖 Ninja。

```bash
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
export CUDA_HOME=/usr/local/cuda
python script/build_grouped_gemm_api.py --source . --output result/build/grouped_gemm_api
```

```python
import sys,torch
sys.path.insert(0,'dev/pytorch')
from grouped_gemm_api import load_library,grouped_gemm,GemmWorkspace
load_library('result/build/grouped_gemm_api/my_cuda_kernel9.so')
with torch.inference_mode():
    x=torch.randn(64,2432,device='cuda',dtype=torch.bfloat16)
    b=torch.randn(2,2432,768,device='cuda',dtype=torch.bfloat16)
    counts=torch.tensor([31,33],device='cuda',dtype=torch.int64)
    y=grouped_gemm(x,b,counts)
    workspace=GemmWorkspace.allocate(x,b)
    y_preallocated=workspace.run(x,b,counts)
```

输入在当前 stream 上就绪，跨 stream 先建立依赖；权重更新后显式使缓存失效。当前数据属于独立投影，完整专家 MLP 和 EP 模型收益另行测量。
