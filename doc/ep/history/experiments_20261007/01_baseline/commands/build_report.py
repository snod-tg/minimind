import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'results/01_baseline'
analysis = json.loads((DATA/'R05_analysis/analysis.json').read_text())
new = [json.loads((DATA/f'R06_timing/summary_rank{r}.json').read_text()) for r in (0,1)]
old = [json.loads((DATA/f'R04_baseline/summary_rank{r}.json').read_text()) for r in (0,1)]

cases=[]
for a,b,c in zip(new[0],new[1],analysis['cases']):
    assert a['case']==b['case']==c['case']
    for rank,x in enumerate((a,b)):
        original=next(y for y in old[rank] if y['case']==x['case'])
        assert x['input']['input_sha256']==original['input']['input_sha256']
    samples=sorted(max(x,y) for x,y in zip(a['e2e']['wall_samples_ms'],b['e2e']['wall_samples_ms']))
    row=dict(c)
    row['final_timing']=dict(samples=60,median_ms=statistics.median(samples),
                             p95_ms=samples[int(.95*(len(samples)-1))],
                             min_ms=min(samples),max_ms=max(samples))
    cases.append(row)
for rank in (0,1):
    x=json.loads((DATA/f'R04_baseline/environment_rank{rank}.json').read_text())
    y=json.loads((DATA/f'R06_timing/environment_rank{rank}.json').read_text())
    for k in ('source_sha256','checkpoint_sha256','autocast_dtype','parameter_dtype','torch','visible_devices'):
        assert x[k]==y[k]

final=dict(timing_run='R06_timing',diagnostic_run='R04_baseline',analysis_run='R05_analysis',cases=cases)
audit_path = DATA/'R07_audit/verification.json'
audit_row = ''
if audit_path.exists():
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'passed'
    final['integrity_verification'] = audit
    audit_row = f"| R07 | 核对测量数据完整性 | 逐项检查文件大小与 SHA256 | {audit['files']} 个文件全部匹配 |\n"
(DATA/'final_summary.json').write_text(json.dumps(final,ensure_ascii=False,indent=2))

rows=['| 负载（每 Rank） | Top-K | 模型前向中位数 ms | P95 ms | 本地总 M 范围 | 单专家 M 范围 |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
for c in cases:
    totals=[r['total_m'] for rank in c['counts'] for r in rank]
    ms=[m for rank in c['counts'] for r in rank for m in r['counts']]
    label=f"Prefill B={c['batch_per_rank']}, S={c['seq_len']}" if not c['context_len'] else f"Decode B={c['batch_per_rank']}, KV=128"
    t=c['final_timing']
    rows.append(f"| {label} | {c['topk']} | {t['median_ms']:.3f} | {t['p95_ms']:.3f} | {min(totals)}～{max(totals)} | {min(ms)}～{max(ms)} |")
timing_table='\n'.join(rows)

rows=['| 负载 | Top-K | 本地专家 MLP 中位数 ms | 完整 EP 层中位数 ms |',
      '| --- | ---: | ---: | ---: |']
for c in cases:
    if not c['micro']: continue
    label=f"Prefill B={c['batch_per_rank']}, S={c['seq_len']}" if not c['context_len'] else f"Decode B={c['batch_per_rank']}, KV=128"
    mlp=[m['mlp']['median_ms'] for m in c['micro']]
    ep=[m['ep']['median_ms'] for m in c['micro']]
    rows.append(f"| {label} | {c['topk']} | {min(mlp):.3f}～{max(mlp):.3f} | {min(ep):.3f}～{max(ep):.3f} |")
micro_table='\n'.join(rows)

rows=['| Top-1 负载 | 专家段 GPU 工作 ms | EP 范围 NCCL kernel ms | local counts CPU 区间 ms |',
      '| --- | ---: | ---: | ---: |']
for c in cases:
    if c['topk']!=1: continue
    d=c['diagnoses'][0]
    nccl=sum(r['duration_ms'] for r in d['kernel_top'] if 'nccl' in r['name'].lower())
    rows.append(f"| {c['case']} | {d['stages']['experts']['gpu_ms']:.3f} | {nccl:.3f} | {d['stages']['expert_counts_to_cpu']['cpu_ms']:.3f} |")
profile_table='\n'.join(rows)

report=f'''# 01 基线测量：结果

## 目的与范围
建立现有 MiniMind-EP 专家前向、完整 EP 和模型前向的参考，决定 kernel9 的实际支持范围。本阶段测量现有实现，没有新增算子加速结果。

## 测量条件
环境：[测试环境](../测试环境.md)。RTX 4090D 物理 GPU 1、2，2 Rank，4 专家、每 Rank 2 专家，hidden=768、intermediate=2432、8 层。现有 pretrain_ep checkpoint；FP32 参数与 BF16 autocast，eval/inference_mode，native EP 通信。

数据集前 512 条文本编码后拼接为无 padding 输入，经现有模型实际路由。每个负载使用一个固定输入快照重复测量，不代表生产请求的统计分布。Top-1 是原 checkpoint 配置；Top-2 为同权重路由实验。两者使用不同固定文本快照，Decode Token 也来自对应模型的 argmax，因此不能把差异完全归因于 Top-K；后续算子比较复用同一组合输入。

## 正式模型计时：R06
预热 20 次，正式 60 次。每次 barrier 在计时外，结束时同步；按同次两个 Rank 的 wall latency 最大值统计中位数和 P95。P95 使用排序后 floor(0.95×(n-1)) 项。所有样本保留。

Prefill 包含全位置 logits 和 KV cache 构建；Decode 固定 128-token KV cache、单步 S=1，包含输出 cache 的构建，不含采样、tokenizer 或连续长序列生成。无 CUDA Graph、无 profiler、无数据抓取。

{timing_table}

本地 M 范围来自 R04 中 8 层、2 Rank 的对应快照，并非每次计时重新随机路由。R04/R06 的每 Rank 输入哈希、模型源码、权重、精度和设备已经比对一致。

## 完整专家 MLP 与 EP 层回放：R04
采集第 0/3/7 层，分别测量本地逐专家三个 Linear、激活、输出拼接，以及整个 EP 层。回放均先通过位级输出检查，再预热 3 次、计时 30 次；按同轮较慢 Rank 统计。CPU 到 GPU 的输入传输在计时外；权重仍为 FP32，BF16 autocast。该表是独立层的缓存与启动条件，不将其直接乘以 8 来推算模型比例。

{micro_table}

## 诊断时间线：R04 抓取、R05 校正
每个组合单独抓取 3 次标记前向；以下为 Rank 0、跨 8 层的每前向平均值。专家段 GPU 工作包含三个投影、激活与拼接；NCCL 列是实际 NCCL kernel 时长之和，包含等待，不能解释为链路纯传输时间；CPU 区间含 profiler 开销与等待。

{profile_table}

这些值来自标记运行，可能重叠，不能相加得到 R06 模型耗时，也不能据此直接预测加速百分比。小 Prefill 的 NCCL 长等待尤其说明 profiler 与双 Rank 执行会扰动时间线。

代码与时间线共同确认：rank counts 与 local expert counts 有 CPU 转换；bincount 路径也出现 DtoH 与同步；原前向逐专家调用三次 Linear。Grouped GEMM 的接入需要同时测量元数据和完整 MLP，不能只优化预分配任务表后的纯计算。

## 输入与正确性
- 10 组合 × 2 Rank 的标记前后 logits/hidden_states 位级一致，共 20 个比较。
- 36 个完整 MLP 回放、36 个完整 EP 层回放全部位级一致。
- 160 个层/Rank counts 记录；逐层全局 assignment 数为两 Rank 输入 Token 数 × Top-K，全部守恒。
- 专家输入、输出为 BF16，权重参数为 FP32；输入行 stride 为 [768,1]。
- 本地总 M 实际出现 0～22463，单专家 M 出现 0～12526。Decode Top-1 记录了空专家和 Rank 零接收。
- 计算语义为 X[M_i,K] @ W_i[N,K].T；gate/up 为 K=768、N=2432，down 为 K=2432、N=768。

## 调整闭环
| 轮次 | 问题 | 调整 | 结果 |
| --- | --- | --- | --- |
| R01 | SCVideo 的 GCC 激活脚本与 nounset 冲突 | 根据用户偏好核实并采用可用 venv | 尚未进入 GPU 测量，无性能数据 |
| R02 | 测试脚本动态模块未注册 sys.modules | 修复加载流程 | R03 通过 |
| R03 | 仅冒烟，预热与样本不足 | 扩为 10 组合并增加 MLP/EP 回放 | R04 全部一致性通过 |
| R04→R05 | profiler CPU/GPU 同名注释被重复统计 | 从实际 Chrome 事件重新关联与统计 | 排除伪 kernel 与重复注释，正式计时不受该汇总错误影响 |
| R04→R06 | 小批量计时长尾明显 | 独立计时、20 次预热、60 次样本 | 形成最终计时表，仍如实保留波动 |
{audit_row}
R04→R06 改变的是测量安排，模型算法未改变。小负载中位数有明显变化，不能将差异记为优化收益；后续必须采用同轮同协议比较新旧后端。

## 下一步依据
kernel9 首先适配 E=2 的本地专家计算、BF16、上述两组 N/K，允许动态总 M 和专家 M，包括零行与尾块。保留 E=16、1024×1024 作为算子验证负载。较大 Prefill 的计算成本明确增加；Decode 的完整 EP 层远长于本地 MLP，元数据、Python 启动、排序和通信也应纳入验证范围。当前证据支持继续实现与接入，但不能预先承诺端到端收益。

## 数据与指令
[完整执行指令](../commands/01_基线测量.md) · [最终汇总数据](01_baseline/final_summary.json) · [校正诊断](01_baseline/R05_analysis/analysis.json)。逐组合数据和时间线以相同组合名组织，详情由这些汇总引用。
'''
(ROOT/'results/01_基线测量.md').write_text(report)

doc=f'''# 01 基线测量

## 目的
定位当前 MiniMind-EP 专家前向的真实开销，采集后续 kernel9 的输入范围。

## 执行方案
1. 核实服务器源码、checkpoint、tokenizer、数据集、双卡与 Python 环境，并保存版本和哈希。
2. 使用现有预训练权重和文本输入，在双卡上测 5 种负载 × Top-1/Top-2；完整模型正式计时使用原 forward。
3. 单独添加范围标记，先对拍原输出，再采集 counts、真实回放和 CPU/GPU 时间线。
4. 分别测完整本地专家 MLP、完整 EP 层和模型；排除 profiler 统计中的重复注释。
5. 针对小批量波动，以独立进程、20 次预热、60 次样本复测同输入；记录中位数和长尾。

[逐轮精确指令](../commands/01_基线测量.md) · [环境](../测试环境.md)。

## 结果
状态：已完成，2026年10月7日。

- 标记前后 20 个模型结果一致；36 个 MLP 与 36 个 EP 层回放一致；assignment 守恒。
- 真实专家输入为 BF16，FP32 权重，投影为 768→2432→768，每 Rank 2 专家。
- 本地总 M 出现 0～22463，单专家 M 出现 0～12526；空专家与 Rank 零接收已实际观察。
- Top-1：Prefill B=8/S=340 模型中位数 13.296 ms，B=32/S=340 为 35.755 ms；Decode B=1/16 为 11.864/12.862 ms。
- 原实现存在多处 CPU counts 转换与同步；独立层回放表明不能将纯 GEMM 收益直接视为完整 EP 收益。

[完整结果与局限](../results/01_基线测量.md)。

## 调整闭环
| 轮次 | 证据与问题 | 调整 | 复测结论 |
| --- | --- | --- | --- |
| R01 | Conda 激活 nounset 失败 | 用户允许时改用已验证 venv | R02 成功导入依赖 |
| R02 | 动态模块加载错误 | 注册 sys.modules | R03 冒烟通过 |
| R03→R04 | 小范围验证不足 | 扩大负载与回放，保持原计算顺序 | 全部一致性和守恒通过 |
| R04→R05 | CPU/GPU 注释重复计数 | 使用实际 kernel 和 CPU annotation 关联 | 诊断口径校正 |
| R04→R06 | 小批量长尾与测量混杂 | 独立计时、增加预热和样本 | 最终表采用 R06，波动保留 |
{audit_row}
进入 02 输入约定：以 BF16、两组模型维度、动态 M/E 与 GPU 元数据为首版目标；训练反向仍为后续范围。Grouped GEMM 的整体收益由后续同轮对比决定。
'''
(ROOT/'docs/01_基线测量.md').write_text(doc)
print('Final JSON, result report and stage description generated.')
