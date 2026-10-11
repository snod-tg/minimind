# 04 PyTorch 接口：执行指令

## 目的
构建并加载 kernel9 的 PyTorch dispatcher 扩展，验证属性、数值、stream/设备、Graph、缓存生命周期及接口耗时。

[说明](../docs/04_PyTorch接口.md) · [结果](../results/04_PyTorch接口.md) · [环境](../测试环境.md)。

## 执行方案：完整复现
执行主机 `Server-Internet`，绝对阶段目录 `/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04`。数值/设备测试使用物理 GPU 1、2；性能及 memcheck 只用物理 GPU 1。未替换 MiniMind 的 EP 实现。

本机同步与连接：

```bash
cd /Users/tongleiwen/Documents/resume
rsync -a --exclude __pycache__ minimind_ep_grouped_gemm/commands/04_api/ Server-Internet:/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/
ssh Server-Internet
```

服务器环境与构建：

```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
export CC=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-cc
export CUDA_HOME=/usr/local/cuda
export OMP_NUM_THREADS=1
"$CXX" --version
"$CUDA_HOME/bin/nvcc" --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
python build_api.py --source source --output results/REPRO_api_build > results/REPRO_api_build.console.log 2>&1
export K9_API_LIBRARY=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/REPRO_api_build/my_cuda_kernel9.so
tail -3 results/REPRO_api_build.console.log
```

[build_api.py](04_api/build_api.py) 直接执行当前 GCC/nvcc，不需要 Ninja。编译 `.cpp`、`.cu` 两个对象再链接，使用当前 torch 的 include/lib 路径、C++11 ABI=true、sm_89、C++17、O3、共享 cudart、cublas 与 torch/c10 CUDA 库。脚本输出可复制的完整参数数组、build.sh、编译结果，并用 load_library 试加载；成功标志 `BUILD PASSED`。输出目录必须尚不存在。扩展依赖当前 PyTorch ABI，切换 PyTorch/设备架构后需重建。

接口测试：

```bash
CUDA_VISIBLE_DEVICES=1,2 python verify_api.py \
  --library "$K9_API_LIBRARY" --source source --output results/REPRO_api_validation \
  > results/REPRO_api_validation.console.log 2>&1
tail -8 results/REPRO_api_validation.console.log
```

[verify_api.py](04_api/verify_api.py) 验证两种接口、72 个数值案例、16 个 Graph 内动态分配、设备/stream、opcheck、动态 eager 编译、错误拒绝和缓存生命周期。成功标志 `ALL API CHECKS PASSED`。每个数值案例调用普通及预分配两种接口，各自对 FP32 舍入参考逐元素检查。梯度拒绝在 inference_mode 之外另测。

实际使用示例：

```bash
CUDA_VISIBLE_DEVICES=1 K9_API_LIBRARY="$K9_API_LIBRARY" python - <<'PY'
import os,sys,torch
sys.path.insert(0,'source/dev/pytorch')
from grouped_gemm_api import load_library,grouped_gemm,GemmWorkspace
load_library(os.environ['K9_API_LIBRARY'])
with torch.inference_mode():
    x=torch.randn(64,2432,device='cuda',dtype=torch.bfloat16)
    b=torch.randn(2,2432,768,device='cuda',dtype=torch.bfloat16)
    counts=torch.tensor([31,33],device='cuda',dtype=torch.int64)
    y=grouped_gemm(x,b,counts)
    workspace=GemmWorkspace.allocate(x,b)
    y_preallocated=workspace.run(x,b,counts)
    torch.testing.assert_close(y,y_preallocated)
    print(y.shape)
PY
```

调用时 x、b、counts 必须已在当前 stream 上就绪；跨 stream 数据通过 wait_stream/event 建立依赖。预分配工作区不在并发 stream 间同时复用。GPU counts 数值检查为契约，`validate_counts=True` 是显式同步调试选项。

同轮性能：

```bash
CUDA_VISIBLE_DEVICES=1 python benchmark_api.py \
  --library "$K9_API_LIBRARY" \
  --native /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/results/R13_release_build/kernel9_bridge.so \
  --source source --output results/REPRO_api_benchmark \
  > results/REPRO_api_benchmark.console.log 2>&1
```

[benchmark_api.py](04_api/benchmark_api.py) 使用相同矩阵、counts、dtype 和 kernel9 默认路径，比较 C ABI native/full、预分配、自动分配。Graph 每图 64 次调用、7 轮交替、每轮 10 次 replay；eager 每轮批量调用 64 次、末尾同步，另记录 Event 与 wall。eager Event 包含 CPU 提交空隙，不等于纯 kernel 时间；native C ABI 也有自己的 Python/ctypes 成本，不能将两个入口的差值直接解释为 C++ dispatcher 单独耗时。

工作区内存检查与绘图：

```bash
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 97 \
  python check_api_sanitizer.py --library "$K9_API_LIBRARY" --source source \
  > results/REPRO_api_memcheck.log 2>&1
grep 'ALL API MEMORY CHECKS PASSED' results/REPRO_api_memcheck.log
grep 'ERROR SUMMARY: 0' results/REPRO_api_memcheck.log
python plot_api.py --results results/REPRO_api_benchmark/benchmark.json --output results/figures/REPRO_grouped_gemm_api.png
```

[check_api_sanitizer.py](04_api/check_api_sanitizer.py) 检查 BF16 两组模型维度、Split-K、常规路径及 FP16 E=16、M=0；均包含自动与预分配调用、非默认 stream 与数值对拍。新增接口 memcheck 是补充检查，阶段 03 对相同 CUDA 数学核心的 22 组 mem/race/sync 检查仍有效。

## 已执行指令与结果
R01～R05 的构建环境与上述相同，构建输出独立。

| 轮次 | 具体指令/修订 | 结果 |
| --- | --- | --- |
| R01_build | 初版 build_api.py 使用 torch.utils.cpp_extension.load，`--source source --output results/R01_build` | Ninja 缺失，构建入口失败 |
| R02_manual_build | 改为直接 GCC/nvcc，`python build_api.py --source source --output results/R02_manual_build` | BUILD PASSED |
| R03_validation | `CUDA_VISIBLE_DEVICES=1,2 python verify_api.py --library results/R02_manual_build/my_cuda_kernel9.so --source source --output results/R03_validation` | 测试对空 tensor 的零指针断言错误 |
| R04_validation_fixed | 同一库，`--output results/R04_validation_fixed` | 数值、Graph、设备通过；动态编译触发手写 Autograd 对 symbolic numel 的访问 |
| R05_dispatch_build | 修复分发，`python build_api.py --source source --output results/R05_dispatch_build` | BUILD PASSED |
| R06_dispatch_validation | `CUDA_VISIBLE_DEVICES=1,2 python verify_api.py --library results/R05_dispatch_build/my_cuda_kernel9.so --source source --output results/R06_dispatch_validation` | 完整验收通过 |
| R07_api_benchmark | 上述 benchmark 指令，library 使用 R05，output 为 results/R07_api_benchmark | 五个同输入负载测量完成 |
| R08_api_memcheck | 上述 memcheck 指令，library 使用 R05，输出 results/R08_api_memcheck.log | 零错误 |

实际 R05 构建封装：[run_build.sh](04_api/run_build.sh)。R07/R08 和绘图封装：[run_measure.sh](04_api/run_measure.sh)。这些记录使用已存在目录；重新执行使用 REPRO 或新轮次目录。

## 调整闭环
Ninja 缺失时复用现有编译器；空 tensor 验收改为按 shape/numel 判断；FakeTensor 与动态编译问题按分发层修复，未修改 CUDA 数学核心。每次接口修订后完整重跑数值和边界，最终 R06、R07、R08 分别作为正确性、性能、内存验收依据。

汇总生成：

```bash
cd /Users/tongleiwen/Documents/resume
python3 minimind_ep_grouped_gemm/commands/04_api/build_report.py
```
