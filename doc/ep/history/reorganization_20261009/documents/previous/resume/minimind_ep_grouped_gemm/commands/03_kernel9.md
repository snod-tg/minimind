# 03 kernel9：执行指令

## 目的与结果
在 RTX 4090D 上复现动态 M 的 kernel9，完成正确性、安全性、同输入 CUTLASS 对照和优化闭环。R09 数值验证、R14 默认路径测量、R15/R16 安全检查、R17 CLI 回归已完成；R18 NCU 因性能计数器权限不足未获得数据。

[阶段说明](../docs/03_kernel9.md) · [结果](../results/03_kernel9.md) · [测试环境](../测试环境.md)。

## 主机、目录与输入
执行主机为 `Server-Internet`。阶段绝对目录是 `/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03`，使用物理 GPU 1，Python 使用 TileLang venv；GCC 显式使用 SCVideo 的 12.4 编译器，无需切换 Python 到 SCVideo。

计算源码为 `source/dev/cuda/moe_grouped_gemm_forward.cu`，基准/安全适配器在 `source/script/`。三组 `(K,N)` 为 `(768,2432)`、`(2432,768)`、`(1024,1024)`；完整分布由实际脚本定义。真实输入来自阶段 01 的 R04 快照，checkpoint 固定 SHA256，见环境记录。

## 最终版本完整复现
以下命令使用新的输出目录，避免覆盖已经完成的轮次；可先复制最终源码与封装脚本。执行这些命令会重新测量，数值可能随频率/负载波动。

在本机执行：

```bash
cd /Users/tongleiwen/Documents/resume
rsync -a --exclude __pycache__ minimind_ep_grouped_gemm/commands/03_kernel9/source/ Server-Internet:/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/source/
rsync -a minimind_ep_grouped_gemm/commands/03_kernel9/build_final.sh minimind_ep_grouped_gemm/commands/03_kernel9/run_ncu.sh Server-Internet:/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/
ssh Server-Internet
```

在服务器上执行环境检查与编译：

```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export K9_CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
"$K9_CXX" --version
/usr/local/cuda/bin/nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
nvidia-smi --query-gpu=index,name,uuid,memory.used,pstate,clocks.sm,power.limit --format=csv
sha256sum /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth
sha256sum /data/user4/tongleiwen/moe-open-baselines/bridge.so /data/user4/tongleiwen/moe-open-baselines/bridge_host.so /data/user4/tongleiwen/moe-open-baselines/bridge_extra.so
export K9_BUILD=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/results/REPRO_build
bash build_final.sh
```

[build_final.sh](03_kernel9/build_final.sh) 封装最终编译步骤，含 sm_89、GCC12、C++17、共享 CUDA runtime 与 native harness 的 rpath。编译成功应得到 `kernel9_bridge.so`、`grouped`、`check_kernel9` 和三份编译记录。脚本不启用 fast-math。

完整数值与动态 counts 验证：

```bash
python source/script/benchmark_kernel9.py \
  --library "$K9_BUILD/kernel9_bridge.so" \
  --validate-only --output results/REPRO_validation \
  > results/REPRO_validation.log 2>&1
tail -5 results/REPRO_validation.log
```

通过条件：`ALL CHECKS PASSED`；输出 JSON 分别包含 102 组合成输入、108 次真实分组投影与 168 次 Graph 动态 counts 验证。输出逐元素阈值及 CPU double 抽样规则见结果文档。

从头扫描 CUTLASS 并测量最终默认路径：

```bash
python source/script/benchmark_kernel9.py \
  --library "$K9_BUILD/kernel9_bridge.so" \
  --benchmark-only --default-policy \
  --output results/REPRO_scan \
  > results/REPRO_scan.log 2>&1
tail -5 results/REPRO_scan.log
```

快速重复 R14 的同轮比较，可复用 R07 已筛选的 CUTLASS 配置：

```bash
python source/script/benchmark_kernel9.py \
  --library "$K9_BUILD/kernel9_bridge.so" \
  --benchmark-only --default-policy \
  --cutlass-selection results/R07_tuning_fixed/benchmark.json \
  --output results/REPRO_default_benchmark \
  > results/REPRO_default_benchmark.log 2>&1
```

前者重新扫描；后者只重新测已选出的 CUTLASS 配置。两者均实际调用 kernel9 的自动策略 `config=-1`，并检查双方输出；矩阵、输出和工作区预分配。正式计时每图 64 次调用、7 轮轮换、每轮 10 次 replay。CUTLASS 描述符预先建立，吞吐只按纯计算比较。

安全检查：

```bash
CHECK_BINARY="$K9_BUILD/check_kernel9" \
SANITIZER_OUT=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/results/REPRO_sanitizer \
BENCH_GPU=1 bash source/script/sanitize_kernel9.sh \
  > results/REPRO_sanitizer.log 2>&1
tail -5 results/REPRO_sanitizer.log
```

通过条件为 22 组检查完成，native harness 均输出 `ALL CHECKS PASSED`；memcheck/synccheck 为零错误，racecheck 为零 hazards、零错误、零 warnings。脚本用 error-exitcode=97，并按不同工具核对汇总。

最终 CLI 与旧 kernel8 回归：

```bash
"$K9_BUILD/grouped" --kernel 9 --dtype bf16 --n 2432 --k 768 --counts 32768,32768 --no-bench
"$K9_BUILD/grouped" --kernel 9 --dtype bf16 --n 768 --k 2432 --counts 0,1 --no-bench
"$K9_BUILD/grouped" --kernel 8 --dtype fp16 --n 1024 --k 1024 --counts 128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128 --no-bench
```

绘图：

```bash
python source/script/plot_kernel9.py \
  --results results/R14_default_benchmark/benchmark.json \
  --output results/figures/grouped_gemm_kernel9.png
```

NCU 的实际尝试指令封装在 [run_ncu.sh](03_kernel9/run_ncu.sh)：

```bash
bash run_ncu.sh
cat results/R18_ncu/profile.log
```

R18 返回 `ERR_NVGPUCTRPERM`，没有有效计数器报告；不把未采集数据当作零 bank conflict 或已确认的 occupancy。

## 已执行轮次与调整指令
各轮独立输出目录。R01～R07 为六个常规配置，R08 起增加 Split-K，R11/R13 增加默认选择与 CLI 修正。失败轮次仅记录调试过程，不作为最终验收结果。

| 轮次 | 具体执行与改变 | 实际结果 |
| --- | --- | --- |
| R01_build | 编译初版 bridge 与 grouped；编译参数同最终构建流程的前两项 | 成功 |
| R02_smoke | `python source/script/benchmark_kernel9.py --library results/R01_build/kernel9_bridge.so --smoke --output results/R02_smoke` | 24 组合成冒烟、144 次动态检查通过 |
| R03_validation | 初版脚本完整验证，`--library results/R01_build/kernel9_bridge.so --validate-only --output results/R03_validation` | CPU 抽样索引越界，未通过 |
| R04_validation_fixed | 整数索引修复后，`--library results/R01_build/kernel9_bridge.so --validate-only --output results/R04_validation_fixed` | 102 组合成、108 次真实投影、144 次动态检查通过 |
| R05_tuning | `--library results/R01_build/kernel9_bridge.so --benchmark-only --output results/R05_tuning` | 不支持 CUTLASS 候选留下 CUDA 错误导致中断 |
| R06_build | 修复 launch 错误处理；native harness 使用 `--cudart shared` 与 rpath | 编译成功 |
| R07_tuning_fixed | `--library results/R06_build/kernel9_bridge.so --benchmark-only --output results/R07_tuning_fixed` | 完整候选扫描完成，发现小 M down 偏慢 |
| R08_split_build | 新增 Split-K4/8、FP32 partial/reduce，重新编译三项 | 编译成功 |
| R09_split_validation | `--library results/R08_split_build/kernel9_bridge.so --validate-only --output results/R09_split_validation` | 102 组合成、108 次投影、168 次动态检查通过 |
| R10_split_benchmark | `--library results/R08_split_build/kernel9_bridge.so --benchmark-only --cutlass-selection results/R07_tuning_fixed/benchmark.json --output results/R10_split_benchmark` | 小 M down 改善，确定默认规则 |
| R11_final_build | 加入 `moe_kernel9_default_config`，编译默认调用路径 | 成功；未作为正式性能版本 |
| R13_release_build | common.h 的 M 与 grouped counts 解除原固定 32768 限制，再编译三项 | 最终实现编译成功；没有执行 R12 |
| R14_default_benchmark | `--library results/R13_release_build/kernel9_bridge.so --benchmark-only --default-policy --cutlass-selection results/R07_tuning_fixed/benchmark.json --output results/R14_default_benchmark` | 10 个负载最终同轮测量完成 |
| R15_sanitizer | 最终 native harness，三种工具检查 | 12 个 memcheck 通过；首个 racecheck 为零错误但摘要解析误判 |
| R16_sanitizer_remaining | 修复摘要解析；`SANITIZER_TOOLS='racecheck synccheck'` 复测 | 剩余 10 个检查通过 |
| R17_cli | 上述三条 CLI，执行对象为 `results/R13_release_build/grouped` | 三项通过 |
| R18_ncu | `bash run_ncu.sh`，M=2912 的 down | 性能计数器权限不足 |

表中省略 `python source/script/benchmark_kernel9.py` 的参数行均使用该实际入口。历史脚本与初版计算源保留对应修订快照；最终复现使用本篇的完整命令。

R15/R16 的实际环境参数：

```bash
CHECK_BINARY=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/results/R13_release_build/check_kernel9 \
SANITIZER_OUT=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/results/R15_sanitizer \
BENCH_GPU=1 bash source/script/sanitize_kernel9.sh

CHECK_BINARY=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/results/R13_release_build/check_kernel9 \
SANITIZER_OUT=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/results/R16_sanitizer_remaining \
SANITIZER_TOOLS='racecheck synccheck' BENCH_GPU=1 bash source/script/sanitize_kernel9.sh
```

这些历史目录已经存在，脚本会阻止覆盖；重新检查应改用新的输出目录。

## 汇总与复核
在本机执行最终报告生成与数量核查：

```bash
cd /Users/tongleiwen/Documents/resume
python3 minimind_ep_grouped_gemm/commands/03_kernel9/build_report.py
```

[build_report.py](03_kernel9/build_report.py) 读取实测 JSON 与 sanitizer 摘要，要求 10 个默认路径负载、108 次真实投影、22 组唯一安全检查，汇总性能及 ptxas 资源记录。循环的最终结论以 [结果文档](../results/03_kernel9.md) 为准。
