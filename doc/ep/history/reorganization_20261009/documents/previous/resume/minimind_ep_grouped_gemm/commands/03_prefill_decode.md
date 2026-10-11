# 第三轮闭环：执行指令

## 目的
复现最终 prefill / decode 的构建、数值、CUTLASS 对比、兼容路径与安全检查。

## 执行方案
本机同步：
```bash
cd /Users/tongleiwen/Documents/resume
rsync -a --exclude __pycache__ --exclude publication minimind_ep_grouped_gemm/commands/03_split/ Server-Internet:/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03_split/
ssh Server-Internet
```
服务器：
```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03_split
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
export SPLIT_BUILD=results/REPRO_release_build
bash build.sh

taskset -c 28 python source/script/bench_split.py --library "$SPLIT_BUILD/split_bridge.so" --mode validate --output results/REPRO_validation
taskset -c 28 python source/script/bench_split.py --library "$SPLIT_BUILD/split_bridge.so" --mode benchmark --output results/REPRO_default_benchmark
taskset -c 28 python source/script/bench_layout.py --library "$SPLIT_BUILD/split_bridge.so" --output results/REPRO_layout
taskset -c 28 python source/script/audit_precision.py --output results/REPRO_precision

CHECK_OUT=results/REPRO_sanitizer PREFILL_CONFIGS=-1,0,1,2,3,4,5,6,7 DECODE_CONFIGS=-1,0,1,2,3,4,5,6 bash sanitize.sh

taskset -c 28 python source/script/check_compat.py --output results/REPRO_native_compat
taskset -c 28 python source/script/bench_compat.py --output results/REPRO_compat_latency
python source/script/plot_split.py --results results/REPRO_default_benchmark/benchmark.json --output results/figures/REPRO_grouped_gemm_prefill_decode.png
```
Python环境已包含torch，使用TileLang venv；CUDA构建使用SCVideo的GCC12.4，nvcc12.4，sm89/O3/C++17，不启用fast-math。输出目录必须新建，避免覆盖旧轮次。bench_split会按形状重新扫描 CUTLASS，有162有效组合、6不支持组合；默认用既定正常选路计时，不能把扫描到的手动最优配置当作正式默认结果。

两个独立入口示例：
```bash
"$SPLIT_BUILD/grouped_prefill" --dtype bf16 --counts 1480,1432 --n 2432 --k 768 --no-bench
"$SPLIT_BUILD/grouped_decode" --dtype bf16 --counts 64,73 --n 768 --k 2432 --no-bench
"$SPLIT_BUILD/grouped_decode" --dtype fp16 --counts 0,1 --n 1024 --k 1024 --no-bench
```
内部名字是kernel9_prefill/kernel9_decode，文件后缀直接prefill/decode。CLI共用common.h参数和验证，不需要gemm_common.cuh。

实际NCU尝试：
```bash
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/ncu --set full --kernel-name regex:moe_grouped_gemm_forward_kernel9_prefill --launch-count 1 --export results/REPRO_ncu_prefill results/R31_release_build/grouped_prefill --counts 1480,1432 --n 768 --k 2432 --no-bench
```

## 结果
最终正式构建 R31_release_build；R32完整FP32参考验证通过；R33同轮性能完成；R34布局成本、R35数值争议完成；R38三种sanitizer共34组通过；R42两算子独立CLI14项通过；R43 NCU被计数器权限拒绝。原生BF16兼容最终342次真实投影逐位通过；R45单独eager计时，R46补充独立autocast。未改变默认BF16归约设置。

## 调整闭环
R01～R07修正动态共享内存启动与参考权重连续性；R08～R24扫描Tile、流水线、缓存、寄存器双缓冲、共享输出写回、跨stage预取；R25编译模板参数错误后在R28修正并重跑；R28～R31收敛8/7配置、简化默认选路与共享helper；R32～R38最终数值/性能/布局/安全检查。R39因checkpoint存储FP16而模型参数实际FP32的测试构造差异失败，R40按实际加载方式修正；R44～R46减少重复autocast开销并复测。失败轮次没有计作验收通过。
