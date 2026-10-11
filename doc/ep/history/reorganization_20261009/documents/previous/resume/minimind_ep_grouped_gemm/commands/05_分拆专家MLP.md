# 分拆专家MLP执行指令

## 目的
连接gate、SiLU、up、乘法、down完整专家MLP，验证真实输入和容量复用，判断单算子收益是否能传递到MLP。

## 执行方案
Server-Internet，下面变量在同一shell设置；REPRO目录必须使用未存在的新名字。最终构建R16；计时使用R01，其CUDA/C++源码完全相同。

```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04_07_split_20261009
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 BENCH_CPU_BASE=28
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
LIB=results/R16_release_build/my_cuda_grouped_split.so
CKPT=/data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth
BASE=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007
```

```bash
CUDA_VISIBLE_DEVICES=4 python source/script/replay_split_mlp.py --library "$LIB" --checkpoint "$CKPT" --baseline "$BASE/results/R04_baseline" --extra "$BASE/stage05/results/R01_extra_capture" --output results/REPRO_mlp
CUDA_VISIBLE_DEVICES=4 python source/script/benchmark_mlp.py --library "$LIB" --checkpoint "$CKPT" --output results/REPRO_mlp_benchmark
```

## 结果
最终R14：100真实MLP、300投影FP32对拍/CPU double抽查，完整FP32链bad=0；15项补充含Graph动态counts、容量边界和KN直接打包。对原默认BF16另有3份MLP未通过原误差门槛，未扩大阈值。R11使用真实权重和合成输入，Graph32节点、7轮轮换×10重放；eager每轮64调用。

| local M / counts | 原 MLP Graph μs | CUDA MLP Graph μs | 原 MLP eager μs | CUDA MLP eager μs |
| --- | ---: | ---: | ---: | ---: |
| 1 / [0, 1] | 17.530 | 15.526 | 57.114 | 60.067 |
| 16 / [8, 8] | 31.914 | 16.694 | 111.120 | 59.635 |
| 137 / [64, 73] | 50.886 | 31.667 | 115.616 | 55.897 |
| 2912 / [1480, 1432] | 290.906 | 287.782 | 298.487 | 292.444 |


## 调整闭环
R03的测试导入根路径不正确，修正后R03b通过；随后改为每stream容量复用、权重record_stream和一次版本检查，最终R14再次完整回放通过。数值闭环保持原BF16与FP32两份参考；3份原结果差异如实保留。M1 Graph虽变快，eager稍慢，继续以调用粒度决定使用范围。

[说明](../docs/05_分拆专家MLP.md) · [指令](../commands/05_分拆专家MLP.md) · [结果](../results/05_分拆专家MLP.md)
