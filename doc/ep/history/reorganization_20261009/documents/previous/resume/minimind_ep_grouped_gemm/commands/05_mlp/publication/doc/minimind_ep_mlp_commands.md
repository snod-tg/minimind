# 05_真实输入回放：具体指令

## 目的

回放真实完整 MLP，核对原输出和 FP32/CPU double 参考，实测完整链并检查安全和生命周期。

[阶段说明](../docs/05_真实输入回放.md) · [结果](../results/05_真实输入回放.md)

## 主机与执行方案

Server-Internet。工作目录 /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage05。补采使用 GPU1、2；回放 / 计时 / memcheck 使用 GPU1。Python 固定为 TileLang venv。checkpoint 与模型 snapshot 沿用基线；不修改 MiniMind 生产模型。

在本地 /Users/tongleiwen/Documents/resume 执行同步：

```bash
rsync -a --exclude __pycache__ minimind_ep_grouped_gemm/commands/05_mlp/ Server-Internet:/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage05/
```

随后在服务器执行。每次 --output 必须为新的目录；正式轮次已存在，重现时使用 REPRO 名称。

```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage05
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 实际轮次指令与结果

R01：补采；8 个 Rank-case 记录、64 个完整 MLP，核对输出 bitwise 一致。

```bash
bash run_capture.sh
```

R02 / R03 / R04：初版使用 revisions/R02 的类，R02 使用该目录的 replay_mlp.py，R03/R04 使用 revisions/R03 的回放脚本。下列是当轮实际执行的入口，当时入口文件与上述快照相同；不要把当前类运行成初版历史结果。

```bash
K9_RUN=R02_mlp_validation bash run_replay.sh --validate-only
K9_RUN=R03_error_audit bash run_replay.sh --validate-only --audit-errors
K9_RUN=R04_initial_benchmark bash run_replay.sh --benchmark-only
```

R02 预期在第 18 例失败，不是通过命令；R03 完整审计不把旧阈值失败隐藏；R04 只对所选 9 个正确性已核对的场景计时。

R05：缓冲复用 / 缓存校验一次 / M=0 调整后的 100 例审计，结果与 R03 完全相同。

```bash
K9_RUN=R05_optimized_audit bash run_replay.sh --validate-only --audit-errors
```

R06：CUTLASS 全候选选择和正式同轮计时。类和回放 / CUTLASS 脚本快照为 revisions/R06；两投影形状 18 次选择，2916 有效候选全输出通过，108 资源不支持组合。Graph / eager 都包含完整三次 GEMM 与激活。

```bash
K9_RUN=R06_final_benchmark bash run_replay.sh --benchmark-only --audit-errors --cutlass-root /data/user4/tongleiwen/moe-open-baselines
```

R07：完整链 memcheck 初轮。

```bash
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 97 python check_mlp_sanitizer.py --source source --library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/R05_dispatch_build/my_cuda_kernel9.so --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth --output results/R07_memory_lifecycle/summary.json > results/R07_memory_lifecycle.console.log 2>&1
```

R08：最终类追加缓存源身份、空输入无梯度检查后，重新核对 100 个回放。

```bash
K9_RUN=R08_release_audit bash run_replay.sh --validate-only --audit-errors
```

R09：使用最终类和新增拒绝案例再次检查；5 组完整 MLP、4 组 Graph counts、失效 / 训练 / 形状 / cache 源 / 空输入无梯度检查通过，memcheck 零错误。

```bash
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 97 python check_mlp_sanitizer.py --source source --library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/R05_dispatch_build/my_cuda_kernel9.so --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth --output results/R09_release_memory/summary.json > results/R09_release_memory.console.log 2>&1
```

## 最终版本直接重现

当前 replay_mlp.py 显式断言新链 vs FP32 舍入链通过；--audit-errors 保留对原 BF16 输出的旧阈值差异，不改变阈值。分配 / 预分配输出 bitwise 一致、单投影 FP32 与 CPU double 仍为失败即退出。

```bash
K9_RUN=REPRO_release_audit bash run_replay.sh --validate-only --audit-errors
K9_RUN=REPRO_full_mlp_benchmark bash run_replay.sh --benchmark-only --audit-errors --cutlass-root /data/user4/tongleiwen/moe-open-baselines
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 97 python check_mlp_sanitizer.py --source source --library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/R05_dispatch_build/my_cuda_kernel9.so --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth --output results/REPRO_memory/summary.json > results/REPRO_memory.console.log 2>&1
python plot_mlp.py --results results/R06_final_benchmark/benchmark.json --output results/figures/grouped_expert_mlp.png
tail -5 results/REPRO_release_audit.console.log
tail -5 results/REPRO_full_mlp_benchmark.console.log
tail -5 results/REPRO_memory.console.log
```

## 文档核对与生成

在本地结果同步结束后执行：

```bash
cd /Users/tongleiwen/Documents/resume
rsync -a --exclude '*.pt' Server-Internet:/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage05/results/ minimind_ep_grouped_gemm/results/05_mlp/
python3 minimind_ep_grouped_gemm/commands/05_mlp/build_report.py --root minimind_ep_grouped_gemm
```

## 调整闭环

| 轮次 | 证据 / 问题 | 调整 | 结果与下一步 |
| --- | --- | --- | --- |
| R01 | 原 36 个 MLP 回放未覆盖大 Prefill 和单条 Decode 的完整 8 层 | 用原 EP 模型补采 64 个 MLP，核对输入、counts 和完整模型输出 | 共 100 个真实 MLP；补采与基线输入/counts 相同，模型输出 bitwise 一致 |
| R02 | 旧逐元素阈值在第 18 个案例中首次失败 | 保留严格阈值；停止“全部通过”的结论，进入数值审计 | 前 17 例通过，发现 prefill_medium top1 / Rank1 / 层7 的 6 个差异 |
| R03 | 需要区分 GEMM 计算错误与 BF16 舍入链的差异 | 增加逐投影 FP32、CPU double、完整 FP32 舍入链及争议点抽查 | 300 投影 / 8976 double 抽查通过；新链 vs 舍入参考 100/100；原输出严格标准仍有 3 例、72 个差异 |
| R04 | 初版完整 MLP 的小 M eager 仍有明显提交开销，大 M 退化 | 在同轮控制组和 profiler 中检查缓存、分配与调用次数 | 原热态路径未出现 `_to_copy`，不把收益归为每步权重转换；3 次 cache.get 和激活分配可减少 |
| R05 | 分配和重复缓存验证增加主机开销 | 每次前向只校验一次全缓存；预分配 SiLU / product 缓冲；M=0 直接 clone | 100 例数值审计与 R03 完全一致，保留原 BF16 激活舍入边界 |
| R06 | 需要完整 MLP 的强开源参照，不能拼接不同轮的单 GEMM 数据 | 对两种投影形状扫描 CUTLASS device/host/extra；gate/up 共享形状配置，down 单独选择；5 条路径同轮计时 | 2916 有效配置 / 108 资源不支持组合；小 M eager 有收益，大 M 不应默认替换 |
| R07 | 新增输出缓冲与 M=0 分支需检查完整链内存和生命周期 | 5 组 counts、4 组 Graph 动态分配、形状/失效/训练状态检查 | memcheck 零错误；共享缓存失效后两种入口都拒绝运行 |
| R08 | 发布类补充了缓存源身份和空输入无梯度约束 | 对最终源码重新回放全部 100 例，显式断言新链与 FP32 舍入参考通过 | 与 R03/R05 审计完全一致，源码 SHA 校验通过 |
| R09 | 新增拒绝逻辑需验证 | 用最终类检查错配缓存源、空输入 requires_grad，再复测完整链内存 | 所有检查通过，memcheck 零错误；模型级数值影响留到 06 追踪 |
