"""Verify numerical gates, occupancy, paired statistics and build stages 6/7 reports."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
six=root/'results/06_ep';seven=root/'results/07_model';code=root/'commands/06_ep'
load=lambda f:json.loads(f.read_text())
v=six/'R03_conservative_validation';full=[];isolated=[];boundaries=[];greedy=[];growing=[]
for rank in [0,1]:
    assert load(v/f'summary_rank{rank}.json')['completed']
    full+=load(v/f'full_model_rank{rank}.json');isolated+=load(v/f'isolated_ep_rank{rank}.json')
    boundaries+=load(v/f'boundaries_rank{rank}.json');greedy+=load(v/f'greedy_rank{rank}.json')
    assert load(v/f'training_rank{rank}.json')['passed']
    assert load(six/f'R04_fast_validation/summary_rank{rank}.json')['passed']
    g=load(six/f'R05_growing_validation/summary_rank{rank}.json')
    assert g['last_layer_strict_pass'] and g['last_layer_tokens_changed']==0
    growing+=load(six/f'R05_growing_validation/growing_rank{rank}.json')
assert len(full)==60 and len(isolated)==480 and len(boundaries)==36 and len(greedy)==24
assert all(r['oracle_error']['strict_pass'] for r in boundaries)
adapter=code/'source/dev/pytorch/minimind_ep_backend.py';sha=hashlib.sha256(adapter.read_bytes()).hexdigest()
assert load(six/'R04_fast_validation/summary_rank0.json')['adapter_sha256']==sha
measurement=load(seven/'analysis.json');repeat=load(seven/'repeat_analysis.json')
assert len(measurement['model'])==10 and len(measurement['ep'])==10 and len(measurement['growing_decode'])==4
assert len(measurement['profiles'])==24 and len(repeat['model'])==6
monitor_counts={}
for run in ['R01_model_timing','R03_small_repeat']:
    directory=seven/run
    env=[load(directory/f'environment_rank{i}.json') for i in [0,1]];pids={str(e['pid']) for e in env}
    assert env[0]['cpu_affinity']==[28] and env[1]['cpu_affinity']==[29]
    samples=[json.loads(s) for s in (directory/'gpu_monitor.jsonl').read_text().splitlines()]
    assert not any(app[1] not in pids for row in samples for app in row['apps'])
    assert all(load(directory/f'summary_rank{i}.json')['adapter_sha256']==sha for i in [0,1])
    monitor_counts[run]=len(samples)
safe=[r for r in full if r['backend']=='last_layer']
assert len(safe)==20 and all(r['logits']['strict_pass'] and r['next_tokens_changed']==0 for r in safe)
assert all(z['routing']['changed_tokens']==0 for r in safe for z in r['layers'])

def num_table():
    text='| 后端 | logits 严格通过 | 最大相对 RMS | 路由变化 Token-层次数 | 固定输入下一 token 变化 | 自建 KV 生成变化 |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for backend in ['kernel9','hybrid','last_layer']:
        rows=[r for r in full if r['backend']==backend];g=[r for r in greedy if r['backend']==backend]
        text+=f'| {backend} | {sum(r["logits"]["strict_pass"] for r in rows)}/20 | {100*max(r["logits"]["relative_rms"] for r in rows):.4f}% | {sum(z["routing"]["changed_tokens"] for r in rows for z in r["layers"])} | {sum(r["next_tokens_changed"] for r in rows)} | {sum(r["changed_tokens"] for r in g)}/160 |\n'
    return text

def latency_table(rows,all_paths=False):
    text='| 场景 | 原实现中位 / P95 | last_layer 中位 / P95 | 延迟减少 |'
    if all_paths:text+=' kernel9 中位 | hybrid 中位 |'
    text+='\n| --- | ---: | ---: | ---: |'+(' ---: | ---: |' if all_paths else '')+'\n'
    for r in rows:
        old=r['paths']['original']['wall'];new=r['paths']['last_layer']['wall'];label=r.get('case',f'top{r.get("topk")} B={r.get("batch")}，8 步')
        text+=f'| {label} | {old["median_ms"]:.3f} / {old["p95_ms"]:.3f} | {new["median_ms"]:.3f} / {new["p95_ms"]:.3f} | {r["latency_reduction_percent"]["last_layer"]:+.3f}% |'
        if all_paths:text+=f' {r["paths"]["kernel9"]["wall"]["median_ms"]:.3f} | {r["paths"]["hybrid"]["wall"]["median_ms"]:.3f} |'
        text+='\n'
    return text

loop6='''| 轮次 | 证据 / 问题 | 调整 | 结果与判断 |
| --- | --- | --- | --- |
| R01 | GPU1/2 起初空闲，随后另一任务占用约 20 GiB，大 Prefill OOM | 保留中断；改用 GPU4/5，不处理其他任务 | 此轮不用于性能结论 |
| R02 | 全量和 M≤256 hybrid 的误差传播到 logits；人工低幅值输入相对 RMS 超 0.5% 使检查停止 | 保留原阈值失败；增加完整 FP32 舍入链边界参考 | 局部正确并不意味着整个模型可直接替换 |
| R03 | 早层低精度差异会影响后续路由 | 增加最后一层且 M≤256 的策略；保持通信/逆置换；完整 10 场景、8 层与训练回退审计 | last_layer 20/20 logits 通过、路由不变；全量和 hybrid 不准入 |
| R04 | 正式快速路径须与审计路径一致；最后一层策略前 7 层可直接原 forward | 比较 10 场景×3 后端×2 Rank 的快速/审计输出 | 60 组 logits/hidden bitwise 相同；最终源码 SHA 核对 |
| R05 | 07 连续 Decode 使用 B16、原 KV 起点，超出 R03 的 B1/B4 自建 KV 范围 | 对同一原 KV、B16、top1/top2、8 步另做数值审计 | last_layer 32 次逐步 logits 对照通过，512 个采样 token 一致 |
'''
loop7='''| 轮次 | 证据 / 问题 | 执行 / 调整 | 结果与判断 |
| --- | --- | --- | --- |
| R01 | 局部 MLP 收益不能推成模型加速 | GPU4/5 同轮原实现、kernel9、hybrid、last_layer；完整模型/第7层EP各60次，增长KV每路径20个8步块 | last_layer 固定输入收益约 -0.15%～0.78%；连续 Decode 约 -0.46%～0.03%，准备成本没有排除 |
| R02 | 需要定位收益被何处限制 | 正式计时后独立 profiler，3场景×4后端×2Rank×3前向，按 CPU 注释/相关ID关联GPU事件 | 专家提交开销可减少；路由、计数/排序、dispatch/combine仍保留，不能相加推算模型时间 |
| R03 | 小于1%的效果和部分块波动需复核 | 独立进程再测小Prefill、单条/批量Decode，6场景、4后端各60次 | 复测收益约 -0.27%～0.89%；top1单条仍退化，部分小场景有可重复的小幅收益，不能宣称普遍加速 |
'''
scope='''2026年10月7日，Server-Internet，GPU4/5 两张 RTX 4090D、EP=2；互联 NODE，无 NVLink，CPU Rank0/1 固定到 28/29（NUMA1）。TileLang venv Python 3.13.5、torch 2.9.0+cu128；BF16 autocast、FP32 参数与 GEMM 累加，TF32 关闭，native EP 通信。沿用原 checkpoint / 模型 snapshot 和输入构造；top2 为原 top1 checkpoint 的路由压力实验。

五个负载均为每 Rank：Prefill B1×128 / B8×340 / B32×340；Decode B1×1 / B16×1，固定原 128-token KV 快照；top1/top2 共 10 场景。Prefill / 固定 Decode 保留全部位置 logits 与新 KV 输出，不包含 tokenizer、采样和持续调度。

四条路径：original；全层 kernel9；hybrid（本地总 M≤256 才用 kernel9）；last_layer（仅最后一层且本地 M≤256）。256 是阶段 05 小 M 结果形成的本次候选策略，未宣称适用于所有模型/形状。默认 original。
'''
evidence6='''严格标准沿用阶段 05：逐元素 atol=0.02 / rtol=0.02，同时相对 RMS≤0.5%。原模型重复执行、带审计执行位级一致，输入与基线逐元素一致、counts 与原采集相同。路由变化按每层每 Token 的专家集合比较，topk 的顺序变化不作为专家集合变化。

'''+num_table()+'''
以上路由数字是 Token-层观察次数，不是唯一 Token 数。全层 kernel9 的 logits 相对 RMS 最高 5.1442%，固定输入下一 token 有 3 次变化；hybrid 的中/大 Prefill 回退原实现，可通过 8/20，但小场景仍不满足标准。两者自建 KV 的 160 个生成 token 均有 2 个变化。

last_layer 的中/大 M 回退保持原输出，小 M 则只修改最后一次专家计算，避免影响后续层的路由。20/20 Rank-场景通过，最大 logits 相对 RMS 0.1785%、逐元素超阈值 0、路由变化 0；自建 KV 的 B1/B4、top1/top2、8 步合计 160 个 token 全部一致。另以 B16 / 原 KV 起点测试 32 次逐步 logits、512 个采样 token，也全部通过/一致。这些是有限样本验证，不表示任意输入下都与原模型等价。

同输入隔离 EP 回放共 10场景×8层×2Rank×3后端=480 次：kernel9 / hybrid / last_layer 对原 BF16 输出严格标准分别 151/160、157/160、158/160；路由专家与 counts 全部一致，所有相对 RMS≤0.5%。未把局部的逐元素失败写成全部通过。

6类人工路由×2Rank×3后端=36 次边界检查，覆盖零接收Rank、同Rank top2、跨Rank top2、偏斜/尾块、某Rank零源输入；分配数守恒、专家顺序和输出形状均通过，完整 FP32 舍入链 EP 参考 36/36 通过。对原 BF16 逐元素+相对RMS标准为21/36，其中低幅值输入的相对误差超限保留记录。训练回退两 Rank 的实际 forward/backward、输入梯度和有梯度参数均 bitwise 一致，训练调用清理对应缓存。
'''
interpret7='''首轮 last_layer 固定前向的延迟变化为 -0.152%～+0.785%，独立复测为 -0.273%～+0.890%。小 Prefill top1 的 0.785%→0.890%，单条 Decode top2 的 0.426%→0.373%，批量 Decode top2 的 0.613%→0.540%，在两窗口有小幅收益；top1 单条 Decode 两轮均略慢。批量 top1 的两轮结果由 +0.039% 到 +0.637%，块间也跨零，不做稳定加速承诺。

全层 kernel9 有些静态前向约快 2.4%～7.1%，大 Prefill 约慢1.0%～1.5%；它没有通过模型数值门槛，速度只能作为诊断，不作为可用性能收益。hybrid 的小 M 效果也因数值失败而不准入。

增长 KV 的 last_layer 8步块耗时变化为 -0.456% / -0.189% / -0.201% / +0.028%（top1 B1/B16、top2 B1/B16）。20个块中每Rank分别发生160 / 160 / 120 / 140次工作区 prepare，均在前向内计时；不能拿稳态无prepare的微基准推断连续 Decode。

独立 profiler 的 Decode B16 top1 / Rank0，8层专家区域 CPU inclusive 时间 original 2.521ms、全量kernel9 0.953ms、last_layer 2.353ms；对应 GPU duration sums 0.413 / 0.369 / 0.408ms。路由CPU约1.696ms、dispatch约1.402ms、combine约1.039ms，计数拆分/计数统计和排序也保留。区域含 profiler、提交与等待，不是正式模型时间分解；NCCL GPU duration 含等待，不能看作纯链路传输。

分组专家路径省掉的是本地 counts.tolist，send/recv splits 的 CPU 同步仍在；PyTorch bincount 本身也可能在其动态输出处理中同步。代码没有改变路由/通信算法，不能仅凭 GEMM 优化消除这些开销。safe策略只覆盖1/8层，收益上限进一步缩小。

R01 的 prefill_small_top2 某个轮换块出现 +15.785% 的相对变化，而总中位数只有+0.230%；该异常波动保留，图中单独标注。两轮正式GPU占用采样未发现其它计算进程共用GPU4/5，整台服务器仍有其他GPU任务、频率未锁定；不能把小幅比例泛化为稳定吞吐承诺。
'''
timing='''正式模型和第7层完整EP：每后端预热20次，5轮轮换，每轮12次，合计60次；切换后端后再预热3次以恢复各自稳态。barrier与计时前同步在区间外，区间为开始Event/墙钟→实际forward→结束Event同步；先按同编号样本取两Rank较慢的wall time，再计算中位数与P95。Event区间含CPU提交空隙，单独保留；不将GPU Event当作纯计算时间。

模型区间包括路由、count exchange、dispatch、原专家或kernel9、元数据、激活、形状变化prepare、combine、attention、全部logits和新KV构建。权重缓存初始化在计时外，2Rank分别8.197/8.066ms；本测量初始化了8层cache，每Rank约171MiB。另记录每场景首次清空工作区的单次前向，不能把单点当分布。

增长KV：从相同原模型128-token KV与首token起点，连续8步，包含argmax与KV推进、logits_to_keep=1，5轮轮换×4块=20个8步块/后端；不含prefix prefill。其算法范围与固定Decode不同，分别呈现。整模型CUDA Graph没有尝试，动态EP已有CPU splits，不宣称可以直接捕获。
'''

docs6='''# 06_EP接入验证

## 目的

把新算子接到真实双卡 EP，检验通信语义、模型误差传播和训练回退，形成数值准入策略。

## 执行方案

通过原 forward 的 AST 保留路由/dispatch/combine/逆置换，仅将本地专家段替换为可切换后端。GPU bincount 直接交给kernel9；变化的M在前向内prepare。训练及 inference_mode 外的调用走原方法并在训练时失效缓存。分别检查同输入EP、逐层EP/hidden、完整logits、下一token、连续生成与人工路由边界。

'''+scope+'''
具体可执行命令：[指令](../commands/06_EP接入验证.md)。

## 结果

'''+evidence6+'''
最终快速/审计路径60组logits和hidden位级相同。接入实现与审计已完成，original保留默认；last_layer仅作为已测样本满足门槛的可选后端，kernel9/hybrid拒绝默认替换。

## 调整闭环

'''+loop6+'''
[详细结果](../results/06_EP接入验证.md)。
'''
result6=docs6.replace('# 06_EP接入验证','# 06_EP接入验证：结果',1).replace('[指令](../commands/06_EP接入验证.md)','[指令](../commands/06_EP接入验证.md)').replace('[详细结果](../results/06_EP接入验证.md)。','[阶段说明](../docs/06_EP接入验证.md)。')
result6+='\n## 最终实现身份\n\n适配器 SHA256：`'+sha+'`；复用阶段04扩展 SHA256 `4154552b563c947f708254d30da9284a9a6b18b230affcca07746e91b31ac6b3`，未修改 CUDA 核心。生产模型源码没有更改。\n'
(root/'docs/06_EP接入验证.md').write_text(docs6);(root/'results/06_EP接入验证.md').write_text(result6)

docs7='''# 07_整体性能评估

## 目的

确认本地算子收益能否传到EP与模型，计入动态工作区、通信和实际生成成本，且性能结论须满足数值门槛。

## 执行方案

同卡、同权重、同输入、同精度比较4种后端；每次先取两Rank较慢完成时间，再统计中位/P95。正式计时不带数据抓取/profiler；独立诊断时间线，不相加推算模型耗时。对小幅结果另开进程复测。

'''+scope+timing+'''
[完整执行命令](../commands/07_整体性能评估.md)。

## 结果

单位ms。通过模型门槛的last_layer与原实现比较：

'''+latency_table(measurement['model'])+'''
连续8步增长KV：

'''+latency_table(measurement['growing_decode'])+'''
'''+interpret7+'''
## 调整闭环

'''+loop7+'''
## 最终判断与后续优先级

阶段06/07实现、验证、模型/EP/生成测量及复测已完成；默认original。当前可选last_layer在已测样本通过数值门槛，部分静态小场景仅获得约0.x%的模型延迟减少，连续Decode基本持平或退化。不能把阶段05本地MLP的最高60.9%延迟减少写成模型收益。

下一轮优先：1）用容量复用或有限形状缓存替代每M重建workspace，并纳入连续生成再测；2）明确低精度舍入与router margin的数值策略，不能靠放宽阈值开放全层替换；3）评估counts/排序的设备侧准备及EP同步/通信，保留当前路由语义；4）改善kernel9大M与M≈137的分块选路。训练反向与整模型Graph仍为独立范围。

[实测结果](../results/07_整体性能评估.md)。
'''
(root/'docs/07_整体性能评估.md').write_text(docs7)
result7='''# 07_整体性能评估：结果

## 目的与执行方案

'''+scope+timing+'''
## 正式结果：R01

单位ms。kernel9/hybrid没有通过模型数值门槛，作为诊断列保留。

'''+latency_table(measurement['model'],True)+'''
完整第7层EP，同一原输入，包含路由、计数交换、两次特征All-to-All及本地专家/prepare：

'''+latency_table(measurement['ep'],True)+'''
增长KV的8步块：

'''+latency_table(measurement['growing_decode'],True)+'''
## 小幅场景独立复测：R03

'''+latency_table(repeat['model'],True)+'''
## 时间线与判断

'''+interpret7+'''
![模型级对照](07_model/figures/minimind_ep_model.png)

## 调整闭环

'''+loop7+'''
## 核对结果

主测10个完整模型和10个EP场景，4后端各60次；增长KV4配置，4后端各20个8步块；24条独立profiler配置。复测6个模型场景、4后端各60次。正式GPU占用采样R01为'''+str(monitor_counts['R01_model_timing'])+'，R03为'+str(monitor_counts['R03_small_repeat'])+'''，均未发现其它进程共用测试卡。

最终默认original；last_layer是有限验证范围的可选策略。整体收益与生成成本已经实测，后续方向为工作区容量复用、数值路由稳定性和EP准备/同步优化。

[阶段说明](../docs/07_整体性能评估.md) · [具体指令](../commands/07_整体性能评估.md)。
'''
(root/'results/07_整体性能评估.md').write_text(result7)

commoncmd='''在本地 /Users/tongleiwen/Documents/resume 同步脚本，随后在 Server-Internet 执行。Python 用 TileLang venv；GPU4/5，CPU28/29；必须使用新的输出轮次，原轮次已存在。
'''
cmd6='''# 06_EP接入验证：具体指令

## 目的

验证真实EP、模型数值传播、原实现回退及最终快速路径。

## 执行方案

'''+commoncmd+'''
```bash
rsync -a --exclude __pycache__ minimind_ep_grouped_gemm/commands/06_ep/ Server-Internet:/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage06/
```

服务器通用环境：

```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage06
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=4,5
export BENCH_CPU_BASE=28
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 实际轮次与结果

R01在GPU1/2运行，外部任务占卡后OOM；R02改卡后保留模型数值失败，并在人工低幅值参考的相对RMS检查处中断。R01/R02快照在revisions下；下面命令是当轮运行入口，不把当前程序重跑当作旧失败轮。

```bash
BENCH_GPUS=1,2 K9_RUN=R01_ep_validation bash run_validation.sh
BENCH_GPUS=4,5 K9_RUN=R02_ep_validation bash run_validation.sh
```

R03使用revisions/R03的类与验证器版本，完整保留原阈值差异、增加last_layer及人工EP的FP32舍入参考；R04最终类只将last_layer的前7层快速调用直接回退原forward，其数学与审计路径相同；R05补齐B16连续Decode范围。

```bash
K9_RUN=R03_conservative_validation bash run_validation.sh
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 check_fast_path.py --source source --library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/R05_dispatch_build/my_cuda_kernel9.so --repo /data/user4/tongleiwen/minimind --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth --baseline /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/results/R04_baseline --output results/R04_fast_validation > results/R04_fast_validation.console.log 2>&1
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 check_growing_decode.py --source source --library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/R05_dispatch_build/my_cuda_kernel9.so --repo /data/user4/tongleiwen/minimind --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth --output results/R05_growing_validation > results/R05_growing_validation.console.log 2>&1
```

## 最终程序直接重现

```bash
K9_RUN=REPRO_ep_validation bash run_validation.sh
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 check_fast_path.py --source source --library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/R05_dispatch_build/my_cuda_kernel9.so --repo /data/user4/tongleiwen/minimind --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth --baseline /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/results/R04_baseline --output results/REPRO_fast > results/REPRO_fast.console.log 2>&1
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 check_growing_decode.py --source source --library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/R05_dispatch_build/my_cuda_kernel9.so --repo /data/user4/tongleiwen/minimind --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth --output results/REPRO_growing > results/REPRO_growing.console.log 2>&1
tail -4 results/REPRO_ep_validation.console.log
tail -4 results/REPRO_fast.console.log
tail -4 results/REPRO_growing.console.log
```

## 结果与通过条件

严格误差不放宽：所有输出有限、shape/counts/路由边界通过，人工完整EP的FP32舍入链通过；全量/hybrid的旧标准失败保留。last_layer须20个模型对照通过、下一token不变、自建KV160token一致、B16原KV512token一致；快速/审计60组bitwise一致；训练反向bitwise一致。[详细结果](../results/06_EP接入验证.md)。

## 调整闭环

'''+loop6
(root/'commands/06_EP接入验证.md').write_text(cmd6)
cmd7='''# 07_整体性能评估：具体指令

## 目的

完成模型/EP/连续Decode的同轮计时、独立诊断和小幅效果复测。

## 执行方案

'''+commoncmd+'''
```bash
rsync -a --exclude __pycache__ minimind_ep_grouped_gemm/commands/07_model/ Server-Internet:/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage07/
```

以下均在服务器执行。前后读取GPU占用，run_measure另每秒记录只读占用；不处理其他任务。切换后的原实现基线与新后端同轮，不与GPU1/2历史速度拼接计算比例。

```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage07
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export BENCH_GPUS=4,5
export BENCH_CPU_BASE=28
nvidia-smi -i 4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
K9_RUN=R01_model_timing bash run_measure.sh
K9_RUN=R02_model_profile bash run_measure.sh --profile-only
K9_RUN=R03_small_repeat bash run_measure.sh --workloads prefill_small,decode_single,decode_batch --no-ep --no-growing
python analyze_model.py --input results/R01_model_timing --profile results/R02_model_profile --output results/analysis.json
python analyze_model.py --input results/R03_small_repeat --output results/repeat_analysis.json
python plot_model.py --results results/analysis.json --output results/figures/minimind_ep_model.png
```

正式轮次已有结果；重现时将3个K9_RUN改成REPRO_model / REPRO_profile / REPRO_repeat，并让分析参数对应改名。每条命令必须在前一轮进程结束后执行，profiler及数值验证不能与正式计时重叠。

## 结果核对

```bash
tail -4 results/R01_model_timing.console.log
tail -4 results/R02_model_profile.console.log
tail -4 results/R03_small_repeat.console.log
nvidia-smi -i 4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

模型10场景、EP10场景各后端60次；增长KV4配置各后端20个8步块；profiler24配置；复测6模型场景各后端60次。先按样本取两个Rank的max，不能取两个Rank中位数的max代替；P95是对应配对分布的观测分位数。[实际数据](../results/07_整体性能评估.md)。

在本地结果同步后生成/验证文档：

```bash
cd /Users/tongleiwen/Documents/resume
rsync -a --exclude '*.pt' Server-Internet:/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage06/results/ minimind_ep_grouped_gemm/results/06_ep/
rsync -a --exclude '*_top*_rank*.json' Server-Internet:/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage07/results/ minimind_ep_grouped_gemm/results/07_model/
python3 minimind_ep_grouped_gemm/commands/07_model/build_report.py --root minimind_ep_grouped_gemm
```

## 调整闭环

'''+loop7
(root/'commands/07_整体性能评估.md').write_text(cmd7)
public='''# MiniMind-EP：Grouped GEMM 接入与整体模型评估

2026年10月7日，双卡EP接入、数值传播审计及模型/EP/连续Decode同轮性能评估完成。全层kernel9与hybrid没有通过模型兼容性门槛；最后一层小M策略通过已测输入的标准，整体收益主要是部分静态小场景约0.x%，连续Decode基本持平或略慢。默认保持original。

## 从问题到接入

原模型已把Token按本地专家分组，专家段仍逐专家运行3个投影。新适配器保留原路由、dispatch/combine、逆置换，只替换专家段；将GPU counts直接交给kernel9，在前向中按实际M准备工作区，缓存权重并保留原激活的BF16舍入边界。训练与inference_mode外走原实现，训练时失效缓存。

'''+scope+'''
## 数值验证决定后端范围

'''+evidence6+'''
## 模型前向结果

'''+timing+'\n'+latency_table(measurement['model'])+'''
增长KV，单位为8步块的ms：

'''+latency_table(measurement['growing_decode'])+'''
![整体模型与测量波动](../figures/minimind_ep_model.png)

## 结果解释与闭环

'''+interpret7+'''
当前项目闭环为：独立算子正确性/开源对照→可调用API→真实MLP→模型误差传播→保守接入→整体计时与复测。局部MLP最高60.9%的延迟减少没有直接变成模型级收益，数值门槛限制了覆盖范围，动态prepare和EP提交/同步限制了连续Decode效果。

后续优先按容量复用工作区，再测增长KV；评估counts/排序及同步/通信；对全层接入先制定并验证低精度与路由稳定性方案。当前数据不足以支持新算子反向、整模型Graph或任意输入的等价性结论。

## 调用与缓存约定

当前适配器绑定本次 MiniMind forward snapshot；更换模型版本后需重新核对语句结构。先完成分布式初始化、checkpoint加载和eval，加载阶段04构建的扩展，再安装适配器：

```python
from grouped_gemm_api import load_library
from minimind_ep_backend import MiniMindEPBackend
load_library(extension_path)
backend = MiniMindEPBackend(model, backend='original', threshold=256)
backend.initialize()  # 一次性权重准备；当前入口准备8层
backend.backend = 'last_layer'
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    output = model(input_ids, past_key_values=cache, use_cache=True)
backend.backend = 'original'
```

加载或原位更新专家权重后显式调用 `backend.invalidate(reason='weights changed')`，再initialize或在下次专家调用重建；设备、精度、专家映射变化同样要重建。工作区每模型只支持一次并发调用，多个stream需要独立实例。训练会自动回退原方法并失效对应cache，回退检查不表示kernel9实现了反向。
'''
pub=root/'commands/07_model/publication/analysis';pub.mkdir(parents=True,exist_ok=True);(pub/'minimind_ep_integration.md').write_text(public)
status=root/'docs/00_总安排.md';s=status.read_text();start=s.index('## 当前状态');end=s.index('\n[测试环境]',start)
s=s[:start]+'''## 当前状态
01～07 的实现、审计和测量已完成。全量kernel9/hybrid出现模型级误差与路由变化，不准入默认替换；last_layer仅在最后一层且本地M≤256使用kernel9，已测20个模型对照及生成检查通过。整体静态小场景有约0.x%的小幅收益，增长KV基本持平或略慢，默认original；后续重点为工作区容量复用、数值路由稳定性及EP准备/同步/通信。
'''+s[end:];status.write_text(s)
envfile=root/'测试环境.md';s=envfile.read_text();marker='## 阶段 06/07：双卡 EP 与模型整体评估，2026年10月7日'
if marker in s:s=s[:s.index(marker)].rstrip()+'\n'
s+='\n'+marker+'\n\n'+scope+'''
GPU4 UUID `GPU-290c4b5b-88d6-fffd-344e-0763e6e6d22a`；GPU5 UUID `GPU-cbae796b-9a3d-0154-69c2-2fa382798a4d`。R01阶段06使用GPU1/2时遭外部任务占用OOM，后续阶段06验证和阶段07计时全部换至4/5；两对卡均为NODE拓扑，但正式性能只在同一对4/5做同轮对比。未安装依赖或重编CUDA核心，复用阶段04 dispatcher扩展。

阶段07主测与复测分别有'''+str(monitor_counts['R01_model_timing'])+'和'+str(monitor_counts['R03_small_repeat'])+'''个GPU占用采样，未发现其它计算进程共用4/5；服务器其它GPU任务未处理，未锁频。模型来源SHA/checkpoint SHA与前阶段相同，最终adapter SHA为`'''+sha+'`。\n\n'+timing+'''
阶段06覆盖60个模型候选对照、480个同输入EP对照、36个人工EP边界参考、两Rank训练回退、60个快速/审计一致性对照、自建KV160token/后端和原KV起点B16的512token/后端。原低精度输出的数值失败保留；last_layer在测试范围通过模型准入。未宣称新算子支持反向或整个EP CUDA Graph。

[阶段06](docs/06_EP接入验证.md) · [阶段07](docs/07_整体性能评估.md)。
''';envfile.write_text(s)
print('Stages6/7 verified and documented; conservative numerical gate passed, rejected variants retained, model benefit limited.')
