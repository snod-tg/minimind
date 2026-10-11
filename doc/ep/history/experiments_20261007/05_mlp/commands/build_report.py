"""Verify completed rounds and generate stage explanations, commands and results."""
import argparse,hashlib,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root;data=root/'results/05_mlp';code=root/'commands/05_mlp'
read=lambda f:json.loads((data/f).read_text())
v=read('R08_release_audit/validation.json');b=sorted(read('R06_final_benchmark/benchmark.json'),key=lambda r:r['m'])
assert len(v)==100 and len(b)==9 and v==read('R05_optimized_audit/validation.json')==read('R03_error_audit/validation.json')
assert all(r['original_exact'] and r['cached_error']['exact'] and r['new_vs_oracle']['element_failures']==0 and r['new_vs_oracle']['relative_rms_pass'] for r in v)
assert sum(len(r['projections']) for r in v)==300
assert sum(p['cpu_samples'] for r in v for p in r['projections'])==8976
fail=[r for r in v if r['preallocated_error']['element_failures']]
assert len(fail)==3 and sum(r['preallocated_error']['element_failures'] for r in v)==72
memory=read('R09_release_memory/summary.json');assert memory['passed'] and len(memory['cases'])==5 and memory['graph_distributions']==4
assert 'ERROR SUMMARY: 0 errors' in (data/'R09_release_memory.console.log').read_text()
assert all(all(e['element_failures']==0 and e['relative_rms_pass'] for e in r['errors'].values()) for r in b)
env=read('R08_release_audit/environment.json')
assert env['mlp_source_sha256']==hashlib.sha256((code/'source/dev/pytorch/grouped_expert_mlp.py').read_bytes()).hexdigest()
selections=[json.loads(f.read_text()) for f in (data/'R06_final_benchmark').glob('*_cutlass_selection.json')]
assert len(selections)==9 and sum(len(r['candidates']) for r in selections)==2916 and sum(len(r['unsupported']) for r in selections)==108

names={'decode_single_top1':'Decode 单条 top1','decode_single_top2':'Decode 单条 top2',
       'decode_batch_top1':'Decode 批量 top1','decode_batch_top2':'Decode 批量 top2',
       'prefill_small_top1':'Prefill 小 top1','prefill_medium_top1':'Prefill 中 top1',
       'prefill_medium_top2':'Prefill 中 top2','prefill_large_top1':'Prefill 大 top1','prefill_large_top2':'Prefill 大 top2'}
keys=['original_fp32_autocast','grouped_preallocated','cutlass_static_preallocated']
def table(metric):
    s='| 场景 | 本地总 M / 两专家 counts | 原逐专家 PyTorch | kernel9 完整 MLP | 调参 CUTLASS 完整 MLP |\n| --- | --- | ---: | ---: | ---: |\n'
    for r in b:s+=f'| {names[r["case"]]} | {r["m"]} / {r["counts"]} | '+ ' | '.join(f'{r[metric][k]:.3f}' for k in keys)+' |\n'
    return s
fail_table='| 场景 | Rank / 层 | 本地总 M | 超阈值输出数 | 相对 RMS |\n| --- | --- | ---: | ---: | ---: |\n'
for r in fail:fail_table+=f'| {names.get(r["case"],r["case"])} | {r["rank"]} / {r["layer"]} | {r["m"]} | {r["preallocated_error"]["element_failures"]} | {100*r["preallocated_error"]["relative_rms"]:.4f}% |\n'
loop='''| 轮次 | 证据 / 问题 | 调整 | 结果与下一步 |
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
'''
scope='''GPU 为物理 GPU1 的 RTX 4090D，BF16 输入/权重/中间输出、FP32 GEMM 累加，专家 MLP 维度 768→2432→768；每 Rank 两个本地专家。使用真实 checkpoint 和基线采集输入。top2 是路由压力实验，不代表该 checkpoint 使用 top2 训练。

完整链执行 gate GEMM→原 BF16 SiLU→up GEMM→原 BF16 乘法→down GEMM。每个非空前向的 kernel9 含三次 GPU 元数据处理；权重、GPU counts、工作区准备在计时外。原实现沿用 FP32 Parameter 和 BF16 autocast；单个 autocast 上下文贯穿热态调用，控制组保留原模块并预缓存 BF16 权重。

Graph：每图 16 次完整 MLP，7 轮轮换，每轮 5 次 replay，CUDA Event 中位数。Eager：每批 32 次，批末同步，7 轮轮换，wall time 中位数；包含 Python / dispatcher / 分配 / 提交空隙。热缓存、不锁频；两种计时分开解释。路由、排序、EP 通信、counts 生成与 H2D、权重打包、prepare 和模型其它部分不在区间内。

CUTLASS 使用相同 BF16 输入、packed 权重及原激活，预建静态 counts 描述符、预分配所有输出。其 C ABI 提交路径与 kernel9 的 PyTorch dispatcher 不同；Graph 是完整执行链对照，eager 不用于归因纯计算速度。动态 counts 的描述符重建成本没有测量，因此不能据此声称 CUTLASS 已可直接接入动态 EP 路由。
'''
numerical='''数值审计已完成，但与原低精度模型的兼容性门槛仍未完全通过。

- 原模型回放和原布局 BF16 权重控制组：100/100 bitwise 一致。
- 分配版 / 复用版：100/100 bitwise 一致；最终源码 R08 与 R03/R05 审计记录完全相同。
- 单投影校验：300 次通过 FP32 matmul 后舍入 BF16 的全输出校验（atol=2e-4、rtol=0.008），8976 个 CPU double 抽查通过。
- 完整新链 vs FP32 舍入链：100/100 通过 atol=0.02、rtol=0.02 及相对 RMS≤0.5%，最大相对 RMS 为 0.0648%。舍入参考保留每个投影、SiLU、乘法的 BF16 输出边界，并非全链 FP32 后只舍入一次。
- 完整新链 vs 原 BF16 autocast：97/100 通过同一逐元素阈值；3 例、72 个输出超阈值。所有 100 例相对 RMS≤0.5%，最大 0.4100%。保留失败，不调整阈值。

'''+fail_table+'''
FP32 舍入链对原输出也在上述同三例出现 73 个超阈值元素；争议点取样 58 个，其中 57 个与实际新中间值的 CPU double 下投影舍入 bitwise 相同。证据支持 BF16 舍入链差异会传播，但不能据此证明模型行为没有变化。下一阶段需检查逐层 hidden、路由专家、logits 和下一 token。

R09 memcheck 的 5 个完整链案例 [0,0] / [1,0] / [1,1] / [31,33] / [0,129] 均通过、零内存错误。另在同一 Graph 中更新 4 组 M=65 分配 [0,65] / [65,0] / [31,34] / [1,64] 并对拍；形状不匹配、共享缓存失效、训练模式、错配缓存源、空输入 requires_grad 均拒绝运行。未将 memcheck 扩写成完整链 racecheck / synccheck 的本轮结论。
'''
interpret='''M=1、2、18、29 的 eager 延迟由 73.498 / 132.248 / 116.309 / 143.708 μs 降至 56.018 / 56.079 / 46.126 / 56.145 μs，分别减少 23.8% / 57.6% / 60.3% / 60.9%。小 Prefill 的 M=137 从 142.541 降至 79.697 μs，减少 44.1%。这些数字只针对资源已准备好的本地专家 MLP。

Graph 下 M=2 / 18 / 29 对原实现为 1.91× / 1.44× / 1.39×；M=1 却由 15.008 增至 18.704 μs，M=137 由 53.214 增至 74.584 μs。M=1 只有一个专家活跃，没有多专家合并收益，kernel9 仍付出元数据和 Split-K 归约开销；M=137 暴露了当前总 M 选路及分块的效率不足。不能把 eager 的主机收益推广为所有小 M 的 GPU 计算收益。

M=2912～22463 的 kernel9 eager 比原实现慢 1.7%～8.7%，Graph 比原实现慢约 5.8%～8.6%。对调参 CUTLASS 的完整链 Graph 吞吐，M=1 / 2 为 118.9% / 105.7%，M=18 / 29 为 98.7% / 92.6%；M=137 仅 40.7%，大 M 为 86.2%～91.9%。这是本轮所测静态参照，不能称为所有开源实现的最优。

5 次小 M 前向的 profiler 中，原路径没有 `aten::_to_copy`；原预缓存 BF16 控制组也没有明显普遍加速。M=18 时原逐专家 SiLU / mul 各 10 次，分组链各 5 次；复用版省去自动 GEMM 的 45 次显式 `aten::empty`。这些证据说明专家调用数与提交/分配成本可减少，不支持“原路径每步重转权重”假设。缓存校验、激活缓冲复用的调整保持输出一致，未融合激活或改变 BF16 舍入边界。
'''
decision='''06 阶段先提供可切换后端并追踪数值，原后端保留为基线。优先验证资源可复用的 Decode 及小 M eager 场景；本轮 M≥2912 的 Prefill 保留原路径作为候选回退，M=137 也不能因 eager 更快就承诺 Graph 收益。未测区间、不同专家倾斜和后续层的分布仍需验证，不能用本表直接制定普适 M 阈值。

模型里的总 M 会变化，预分配调用需要在形状变更时 prepare；其成本未计入本轮，应在 06/07 单独测量并纳入端到端结果。每层 packed 权重额外约 21.375 MiB，两本地专家 / 8 层约 171 MiB；四个 [M,2432] BF16 中间缓冲及 [M,768] 输出另占 20992×M bytes，再加元数据与 Split-K 工作区。工作区不能同时被多个 stream 复用，返回结果会在下一次前向被覆盖，需要立即交给后续消费或 clone。

模型权重原位修改、加载新权重、设备/精度/专家映射变化后需显式 invalidate 并重建 cache。当前只支持前向，不支持训练。gate/up 合并、激活融合、完整模型 Graph、训练反向和默认生产替换均未验证。
'''
analysis='''# MiniMind-EP：真实专家 MLP 的 Grouped GEMM 回放

2026年10月7日，完整本地专家 MLP 的实现、数值审计和同轮性能对照已完成。当前 kernel9 的收益集中在资源已准备好的小 M eager 调用；大 Prefill 退化，原低精度模型的数值兼容性仍有 3 个案例需在模型级追踪。

## 从问题到实现

原 EP 已把收到的 Token 按本地专家分组，仍逐专家运行三个线性层与激活，再拼接输出。本阶段复用 kernel9 API，将本地专家 MLP 组织为三个 Grouped GEMM，保持 SiLU 与乘法的 BF16 舍入边界。预打包权重只在初始化执行，复用 GEMM / 激活工作区，每次前向校验一次缓存；空 M 直接返回空输出。

## 测量约定

'''+scope+'''
## 正确性与模型兼容性

'''+numerical+'''
## 最终版本对照

单位 μs，越小越好；三条曲线均为完整专家 MLP。

CUDA Graph：

'''+table('graph_us')+'''
Eager：

'''+table('eager_wall_us')+'''
![完整专家 MLP 对照](../figures/grouped_expert_mlp.png)

CUTLASS 为现有 SM80 TensorOp grouped adapter：device / host 各 25 配置，extra 6 配置，CTA 数取 0 / 114 / 228。每案例 gate/up 同形状与 down 共两组选择；2916 组有效候选全输出数值通过，108 组不支持的资源组合单独计数。先按短 Graph 扫描，再各取前 3 名用 16 次调用图复测；最终 5 条控制路径按 7 轮轮换测量，而非拿调参最低单点充当正式结果。

## 结果解释

'''+interpret+'''
## 接入判断

'''+decision
publication=code/'publication';(publication/'analysis').mkdir(parents=True,exist_ok=True)
(publication/'analysis/grouped_expert_mlp.md').write_text(analysis)
docs='''# 05_真实输入回放

## 目的

验证 kernel9 的单 GEMM 收益能否传到真实完整本地专家 MLP；区分数学正确性、原模型兼容性与可接入的性能范围。

## 执行方案

1. 复用原 36 个 MLP，并用原模型补采大 Prefill / 单条 Decode 全 8 层的 64 个 MLP；核对输入、counts、完整模型输出。
2. 从同一 checkpoint 构建原逐专家前向、原布局 BF16 缓存控制、三个 Grouped GEMM 加原激活；保持每个 BF16 舍入边界。
3. 严格对拍原输出，逐投影检查 FP32 / CPU double，完整链检查 FP32 舍入参考；出现差异保留审计，不放宽旧标准。
4. 给非空路径复用全部输出缓冲；减少重复 cache 校验，M=0 跳过计算；验证形状变化、cache 失效和 Graph 动态 counts。
5. 给 9 个真实分布扫描 CUTLASS 参照，5 条路径同轮轮换测 Graph / eager，profiler 在正式计时后执行；最后检查 memcheck 并记录接入判断。

可执行脚本及逐轮指令：[执行方案](../commands/05_真实输入回放.md)。

## 结果

本阶段回放与性能审计完成；没有判定原模型兼容性“全部通过”。100 个真实 MLP、300 次投影、8976 个 CPU double 抽查通过，新链 vs FP32 舍入链 100/100 通过；原 BF16 输出严格逐元素标准通过 97/100，仍有 3 例 / 72 个差异。最终 memcheck 零错误。

小 M eager 完整 MLP 减少 23.8%～60.9% 延迟；M=137 减少 44.1%，但该形状 Graph 退化。M=2912～22463 eager 慢 1.7%～8.7%。CUTLASS 同轮完整链参照已实测；本阶段没有获得模型端到端加速结论。

完整数据、阈值与阶段判断：[实测结果](../results/05_真实输入回放.md)。

## 调整闭环

'''+loop+'''
## 下一阶段的进入条件

先以可切换后端验证数值传播，检查各层 hidden、专家路由、logits 与下一 token；把真实形状变化的 prepare 成本纳入测量。大 Prefill 保留原路径，不能用只含 GPU/本地 MLP 的数字承诺整模型收益。
'''
(root/'docs/05_真实输入回放.md').write_text(docs)
results='''# 05_真实输入回放：结果

[阶段说明](../docs/05_真实输入回放.md) · [具体指令](../commands/05_真实输入回放.md)

## 目的与执行范围

验证完整本地专家 MLP 的数值、开销与接入范围。具体轮次 R01～R09，2026年10月7日，均已完成；R02 因原阈值对拍失败退出，其余数值审计保留该失败。执行条件如下。

'''+scope+'''
## 数值结果

'''+numerical+'''
## 正式性能：R06

单位 μs，中位数。

CUDA Graph：

'''+table('graph_us')+'''
Eager wall time：

'''+table('eager_wall_us')+'''
原布局 BF16 控制组和分配版作为诊断控制，不代替最终复用版性能结论：

| M | BF16 控制 Graph | 分配版 Graph | BF16 控制 Eager | 分配版 Eager |
| ---: | ---: | ---: | ---: | ---: |
'''
for r in b:results+=f'| {r["m"]} | {r["graph_us"]["per_expert_bf16_cached"]:.3f} | {r["graph_us"]["grouped_allocating"]:.3f} | {r["eager_wall_us"]["per_expert_bf16_cached"]:.3f} | {r["eager_wall_us"]["grouped_allocating"]:.3f} |\n'
results+='''
![完整 MLP 对照](05_mlp/figures/grouped_expert_mlp.png)

## 判读与调整闭环

'''+interpret+'\n'+loop+'''
## 最终判断

'''+decision+'''
## 证据核对

R08 [数值审计](05_mlp/R08_release_audit/validation.json)、R06 [性能](05_mlp/R06_final_benchmark/benchmark.json)、R09 [完整链生命周期](05_mlp/R09_release_memory/summary.json) 对应上述数量与表格。R06 的计时类快照保留在指令脚本的 revisions/R06；R08/R09 最终类仅追加缓存源身份 / 空输入无梯度检查，非空前向计时体未修改，R08 复测输出完全一致。
'''
(root/'results/05_真实输入回放.md').write_text(results)

commands='''# 05_真实输入回放：具体指令

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

'''+loop
(root/'commands/05_真实输入回放.md').write_text(commands)
status=root/'docs/00_总安排.md';s=status.read_text();old='01 基线测量、02 输入约定、03 kernel9 与 04 PyTorch 接口已完成。算子及接口已通过数值、动态 counts、设备/stream、Graph、内存与性能验证；05 完整 MLP 回放、06 EP 接入、07 整体模型评估待执行。'
new='01～04 已完成。05 完整 MLP 回放、数值审计及 CUTLASS 同轮性能对照完成：100 个真实 MLP，新链对 FP32 舍入参考 100/100 通过，原 BF16 输出严格标准仍有 3 例 / 72 个差异；小 M eager 有收益、大 Prefill 退化。06 先以可切换后端追踪模型级数值影响；06 EP 接入与 07 整体模型评估尚未执行，未默认替换生产模型。'
assert old in s or new in s;status.write_text(s.replace(old,new))
envfile=root/'测试环境.md';s=envfile.read_text()
mark='## 阶段 05：真实完整 MLP，2026年10月7日'
if mark in s:s=s[:s.index(mark)].rstrip()+'\n'
s+='\n'+mark+'''

补采使用物理 GPU1/2、EP=2；全部单 Rank 回放、正式性能和 memcheck 使用物理 GPU1 的 RTX 4090D。仍用 TileLang venv 的 Python 3.13.5、torch 2.9.0+cu128、CUDA runtime 12.8，未安装依赖。模型和 checkpoint SHA 与阶段 01 相同。

复用阶段 04 R05 dispatcher 扩展（SHA256 `4154552b563c947f708254d30da9284a9a6b18b230affcca07746e91b31ac6b3`），本轮未重编计算核心。真实输入为 36 原回放加 64 补采，合计 100 个，M=0～22463，8 层、两 Rank、top1/top2；真实模型参数 FP32，autocast BF16，packed 权重与激活 BF16、GEMM 累加 FP32。top2 为路由压力实验。

独立 MLP 使用显式非默认 CUDA stream。Graph 每图 16 次完整 MLP、7 轮轮换、每轮 5 次 replay；eager 每批 32 次调用、末尾同步、7 轮轮换。原热态参考使用普通 Parameter，在单个 autocast 上下文中测量；未把每步 BF16 权重转换当作已证实瓶颈。计时包含三投影、两激活操作、kernel9 GPU 元数据；counts/H2D、权重缓存与工作区准备、路由、通信在区间外。工作区变形成本留到模型接入测量。

CUTLASS 仍使用阶段 03 记录的三个固定 SHA 二进制，预建静态 counts 描述符；每个 gate/up 或 down 形状扫描 56 配置×3 CTA 数，162 有效 / 6 不支持，合计 18 次选择。9 个场景、5 条控制路径同轮正式计时。其 C ABI 与 PyTorch dispatcher 的主机成本不同，不把 eager 差值视为纯 kernel 效率。

Compute Sanitizer memcheck R07 / R09 完整链均零错误；最终类另覆盖 Graph 更新 counts、空 M、形状不匹配、cache 失效、训练模式、错配源与 requires_grad 空输入拒绝。没有在本阶段重复全部计算核心 racecheck/synccheck，也没有获得 ncu 计数器数据。绘图沿用服务器 Matplotlib 3.10.0。

最终 Python 类 R08 SHA256：`'''+env['mlp_source_sha256']+'''`。R06 计时类快照的非空前向与发布类相同；发布类仅追加初始化 cache 源检查和空输入无梯度检查，R08 对全部 100 回放的结果与 R03/R05 完全一致。

[阶段 05 说明](docs/05_真实输入回放.md) · [指令](commands/05_真实输入回放.md) · [结果](results/05_真实输入回放.md)。
'''
envfile.write_text(s)
print('Verified 100 MLP / 300 projection / 8976 double / 2916 CUTLASS / 0 memcheck errors; compatibility failures retained.')
