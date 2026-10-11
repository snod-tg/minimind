from pathlib import Path
import json
import shutil

M=Path('/Users/tongleiwen/Code/minimind');K=Path('/Users/tongleiwen/Code/my_cuda_kernels')
run=M/'runs/ep6_top2_acceptance_20261011';doc=M/'doc/ep'
summary=json.loads((run/'03_benchmark_v2/summary.json').read_text())
validation=[json.loads((run/f'01_validation/rank{i}.json').read_text()) for i in [0,1]]
migration=json.loads((run/'02_migration.json').read_text())
assert summary['passed'] and summary['occupancy_clean'] and summary['rank_fallbacks']==[0,0]
assert all(x['passed'] for x in validation) and migration['output_comparison_passed']
backup=doc/'history/acceptance_documentation_20261011';assert not backup.exists();backup.mkdir(parents=True)
def put(path,text):
    if path.exists():
        root=M if path.is_relative_to(M) else K
        old=backup/root.name/path.relative_to(root);old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,old)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
def page(title,purpose,plan,result,loop):
    return f'# {title}\n\n## 目的\n{purpose}\n\n## 执行方案\n{plan}\n\n## 结果\n{result}\n\n## 调整闭环\n{loop}\n'
names={'decode_b1':'Decode B1','decode_b16':'Decode B16','prefill_m128':'Prefill M128','prefill_m2720':'Prefill M2720','prefill_m10880':'Prefill M10880'}
table='| 场景 | Original ms | Compact ms | 耗时下降 | 峰值显存 Original / Compact GiB |\n| --- | ---: | ---: | ---: | ---: |\n'
for x in summary['cases']:
    table+=f"| {names[x['case']]} | {x['original_ms']:.3f} | {x['compact_ms']:.3f} | {x['latency_reduction_pct']:.2f}% | {x['memory']['original']/2**30:.3f} / {x['memory']['compact']/2**30:.3f} |\n"
setting='6专家Top-2、2 Rank、每Rank3专家，hidden768、8层、专家中间维度2432；RTX4090D物理GPU5/7，PyTorch2.9.0+cu128。FP16导出权重加载为FP32模型参数，BF16 autocast；专家矩阵乘仍使用原PyTorch。'
common='''在MiniMind根目录、训练所用PyTorch环境执行。先用nvidia-smi检查GPU5/7空闲；改变GPU时两张必须来自前四卡或后四卡的同一组。

```bash
MODEL=out/ep6_top2_compile_20261009/pretrain_ep6_top2_compile_20261009_768_moe.pth
CONFIG=out/ep6_top2_compile_20261009/training_config.json
RUN=runs/ep6_top2_acceptance_20261011
```

重复执行时选择新的输出目录，不覆盖已完成记录。
'''
step1='''完整模型219个checkpoint张量全部有限、8层×6专家×3投影的NK形状完整。两Rank各585个恢复文件张量有限；epoch=1（从0计）、step=19848、world_size=2，AdamW状态已实际load_state_dict。恢复文件模型值与完整导出的相应子集逐元素相同，共享Embedding/LM Head保持同一参数。未执行优化器更新。

每Rank完成输入长度1/17/128/340的Prefill及一次缓存Decode，输出有限；3个提示各生成16个Token。首轮未加训练BOS时出现单词重复；补齐BOS后恢复连续续写。哈尔滨相关回答仍出现地理错误，这些探针不替代验证集质量评估。'''
command1=common+'''
```bash
CUDA_VISIBLE_DEVICES=5,7 OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module tests.ep.validate_trained_checkpoint \
  --checkpoint "$MODEL" --training-config "$CONFIG" --resume-dir checkpoints \
  --output "$RUN/validation_new" --prepend-bos
```
'''
loop1='首轮生成探针漏加训练使用的BOS，补测输入格式后单词重复消失；不能据缺BOS的探针断定训练失败。模型存在常识错误，后续单独评估验证集loss和标准生成；本轮不更改权重或训练。'
step2=f'''正式文件201份，其中197份是新增或更新；覆盖前逐文件校验，原服务器版本已保留。旧模型、新完整权重及Rank恢复文件的大小和修改时间均未变化。

my_cuda_kernels以editable包安装在服务器同级项目目录，模型使用普通import。已有GCC12和nvcc编译EP库，通过动态加载；通用包3项、模型5项CPU检查全部通过。迁移后original/compact分别与首轮参考比较，4组Prefill、4组缓存Decode及3组16步生成均逐元素/逐Token相同；带BOS的original/compact探针也相同，两Rankcompact回退为0。'''
command2=common+'''
先按部署清单校验文件SHA256，保留旧版，确认正式模型核心文件与原版相同，再安装和构建。工具链选择已有SCVideo GCC12：

```bash
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
export CC=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-cc
export CUDA_HOME=/usr/local/cuda
python -m pip install --no-index --no-deps --no-build-isolation -e ../my_cuda_kernels
python ../my_cuda_kernels/script/build.py ep
python -m unittest tests.ep.test_local_contracts -v
(cd ../my_cuda_kernels && python -m unittest tests.test_package -v)

# 与带BOS的original参考对拍
CUDA_VISIBLE_DEVICES=5,7 OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module tests.ep.validate_trained_checkpoint \
  --checkpoint "$MODEL" --training-config "$CONFIG" --resume-dir checkpoints \
  --output "$RUN/migration_compact_new" --routing compact --prepend-bos \
  --reference "$RUN/02_bos_original"
```
'''
loop2='先验证原模型源码哈希相同，再迁移；包安装不增加或升级依赖。新库编译后以模型输出对拍作为切换验收。迁移未引入专家GEMM后端，也不宣称验证了新的训练收敛；历史运行保留原配置。'
step3=setting+'\n\n'+table+'''
5组隐藏状态和logits均完全相同。每组8次预热、31轮original/compact交替测量，先对同轮两Rank取较大值，再取中位数；同步CUDA，前置barrier不计入耗时。GPU占用监测20次，仅有本次测量进程，compact非预期回退为0。

M是每Rank输入Token数；Top-2展开后的专家行数动态分配。Decode是KV上下文128下的一次next-token前向；Prefill和Decode均保留KV，仅输出最后位置logits。模型推理未启用torch.compile，两种路径一致；本轮是整模型EP收益，不是单GEMM或编译收益。

显存为单Rank的PyTorch峰值allocated，取两Rank较大值，不是nvidia-smi总占用。M10880的compact工作区最大约1.592GiB，说明提速伴随工作区占用代价。'''
command3=common+'''
```bash
python -m scripts.ep.benchmark_trained_ep --gpus 5,7 \
  --checkpoint "$MODEL" --training-config "$CONFIG" \
  --output "$RUN/benchmark_new" --rounds 31 --warmup 8
```

缺库或GPU占用不符合要求会报错。库默认使用my_cuda_kernels/build/ep/current；可传--library覆盖。计时脚本保存两Rank样本，整体结果按同轮较慢Rank计算；吞吐按两个源Rank实际输入Token数汇总。
'''
loop3='首轮计算/数值对拍完成，但结束统计把主动禁用优化EP的original基线调用计作回退，触发严格断言。已分开记录intentional_original_bypasses和compact非预期回退，保留首轮并完整重测。第二轮两Rank主动基线绕过各1656次，compact真正回退均为0；只使用第二轮作为最终结果。后续若优化大M显存，应单独设计跨层工作区复用并重新对拍/测量；本轮不继续优化算子。'
steps=[('10_新模型验收','最终模型与恢复状态验收','确认正式训练产物能恢复、数值有限且能够Prefill/Decode。',command1,step1,loop1),
       ('11_服务器迁移','服务器项目迁移与输出对拍','把两个项目切换到独立正式入口并保持原模型数值。',command2,step2,loop2),
       ('12_6选2EP基线','六专家Top-2的EP基线复测','用同一训练模型确定compact EP的整模型收益与显存代价。',command3,step3,loop3)]
for name,title,purpose,command,result,loop in steps:
    for kind in ['design','commands','results']:
        plan=command if kind=='commands' else f'[完整操作指令](../commands/{name}.md)\n\n'+setting
        put(doc/kind/(name+'.md'),page(title,purpose,plan,result,loop))
analysis=page('六专家Top-2：验收、迁移与EP整模型收益','验证独立新模型和两个项目的接入闭环，并公平对照原EP和compact EP。',setting+'\n\n[验收](../doc/ep/design/10_新模型验收.md) · [迁移](../doc/ep/design/11_服务器迁移.md) · [性能](../doc/ep/design/12_6选2EP基线.md)',step1+'\n\n'+step2+'\n\n'+step3+'\n\n![六专家Top-2对照](../figures/ep6_top2_baseline_20261011.png)',loop1+'\n\n'+loop3)
put(M/'analysis/ep6_top2_validation_and_baseline.md',analysis)
p=doc/'README.md';text=p.read_text()
for name,_,_,_,_,_ in steps:text+=f'| {name} | [design](./design/{name}.md) | [commands](./commands/{name}.md) | [results](./results/{name}.md) |\n'
put(p,text)
p=M/'readme_ep.md';text=p.read_text();text+='''
## 2026-10-11：6选2模型验收与EP复测

最终模型与两Rank恢复状态通过加载、有限性和输出验收；服务器正式包切换后，original/compact的隐藏状态、logits和生成Token一致。GPU5/7、31轮配对测量，compact整模型耗时下降8.65%～20.88%；M10880峰值allocated显存约1.51→2.96GiB。推理两路径均为eager、原PyTorch专家、BF16 autocast。BOS补齐后可连续续写，常识准确性仍需单独评估。

[分析与图表](analysis/ep6_top2_validation_and_baseline.md) · [验收](doc/ep/design/10_新模型验收.md) · [迁移](doc/ep/design/11_服务器迁移.md) · [性能指令](doc/ep/commands/12_6选2EP基线.md)
''';put(p,text)
build=json.loads((run/'02_native_build.json').read_text())
put(K/'doc/results/ep_server_integration_20261011.md',page('EP算子包服务器接入验收','验证正式算子包可被MiniMind普通import并通过原生精度模型对拍。','使用服务器已有Python3.13.5 / PyTorch2.9.0+cu128、GCC12与nvcc；python script/build.py ep构建，随后执行MiniMind的最终权重验收。',f"EP库构建和加载成功，架构{build['arch']}、C++11 ABI={build['abi']}。通用包3项CPU检查通过，MiniMind侧5项通过；6选2完整模型对拍通过。当前只构建EP库，未接入手写专家GEMM。\n\n[模型侧结果](../../../minimind/analysis/ep6_top2_validation_and_baseline.md)",'构建先完成，再切换current；实际模型数值不变后才测量。original基线绕过计数与优化EP非预期回退需分开。'))
env_path=doc/'environment.md';env=env_path.read_text();env+='''
## 2026-10-11 服务器验收与基线

物理GPU5/7，两张RTX4090D，当前测量进程独占这两张卡；Python3.13.5、PyTorch2.9.0+cu128，已有SCVideo GCC12构建EP库。已安装正式editable包，默认库在my_cuda_kernels/build/ep/current。数值对拍与生成带训练BOS；6专家Top-2，FP16导出→FP32参数，BF16 autocast。性能时不启用模型compile；实际时序/显存定义见第12步，避免与训练速度或其他模型的旧数据混算。
''';put(env_path,env)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10.5,4.7));x=np.arange(5)
original=[r['original_ms'] for r in summary['cases']];compact=[r['compact_ms'] for r in summary['cases']]
ax.bar(x-.18,original,.36,color='#4e79a7',label='Original PyTorch EP')
bars=ax.bar(x+.18,compact,.36,color='#f28e2b',label='Compact EP')
for bar,row in zip(bars,summary['cases']):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.2,f"-{row['latency_reduction_pct']:.1f}%",ha='center',fontsize=10)
ax.set_xticks(x,['Decode B1\nKV 128','Decode B16\nKV 128','Prefill\nM128','Prefill\nM2720','Prefill\nM10880'])
ax.set_ylabel('Full-model median latency (ms)');ax.set_title('6 experts, Top-2: original vs compact EP')
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper left');ax.set_ylim(0,max(original)*1.15)
fig.text(.5,.015,'RTX 4090D x 2, GPU5/7 | native PyTorch experts, BF16 autocast | 31 alternating paired rounds\nMedian of per-pair max across ranks; last-position logits + KV cache; input M is per rank.',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.10,1,1));figure=M/'figures/ep6_top2_baseline_20261011.png'
assert not figure.exists();fig.savefig(figure,dpi=180,bbox_inches='tight');plt.close(fig)
print('Nine stage documents, model analysis, operator integration result, environment and figure written')
