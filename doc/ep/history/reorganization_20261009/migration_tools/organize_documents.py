"""Explicit, recoverable documentation and experiment ownership migration."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil

K=Path('/Users/tongleiwen/Code/my_cuda_kernels')
M=Path('/Users/tongleiwen/Code/minimind')
W=Path('/Users/tongleiwen/Documents/resume')
S=W/'minimind_ep_grouped_gemm'
B=M/'doc/ep/history/reorganization_20261009/documents'
assert not B.exists()
moves=[];mapping={};writes={}
def move(source,target):
    assert source.exists() and not target.exists(),(source,target)
    moves.append((source,target))
    if source.is_dir():
        mapping.update({p:target/p.relative_to(source) for p in source.rglob('*') if p.is_file()})
    else:mapping[source]=target
def put(path,text):writes[path]=text
def owner(name):
    return K if name.startswith(('02','03')) or name in ('04_PyTorch接口.md','04_分拆算子封装.md') else M
def docroot(root):return root/'doc' if root==K else root/'doc/ep'
stages={'01_baseline':M,'02_contract':K,'03_kernel9':K,'03_split':K,'03_weak':K,
        '04_api':K,'04_weights':M,'04_07_split':M,'05_mlp':M,'06_ep':M,'07_model':M,'08_ep':M,'09_compile':M}
for name,root in stages.items():
    move(S/'commands'/name,docroot(root)/'history/experiments_20261007'/name/'commands')
    move(S/'results'/name,(root/'result/history/experiments_20261007' if root==K else root/'runs/history/experiments_20261007')/name)
for folder,category in [('docs','design'),('commands','commands'),('results','results')]:
    for path in sorted((S/folder).glob('*.md')):
        move(path,docroot(owner(path.name))/category/path.name)
move(S/'测试环境.md',M/'doc/ep/environment_history_20261007.md')
model_analyses=['ep_routing_optimization.md','ep_routing_api.md','grouped_expert_mlp.md',
                'minimind_ep_integration.md','minimind_kn_checkpoint.md','grouped_gemm_split_integration.md']
for name in model_analyses:move(K/'analysis'/name,M/'analysis'/name)
for name in ['ep_routing_optimization.png','grouped_expert_mlp.png','minimind_ep_model.png','grouped_gemm_split_integration.png']:
    move(K/'figures'/name,M/'figures'/name)
for name in ['06_EP接入验证.md','07_整体性能评估.md','minimind_ep_mlp_commands.md']:
    move(K/'doc'/name,M/'doc/ep/history'/name)
move(K/'guide.md',K/'doc/history/guide_before_20261009.md')
move(K/'dev/pytorch/README_grouped_split.md',K/'doc/history/README_grouped_split_before_20261009.md')

# Repair Markdown links against the destination of each moved artifact.
link=re.compile(r'(!?\[[^\]]*\]\()([^\s)]+)(\))')
def repair(text,old,new):
    def target(match):
        value=match[2]
        if '://' in value or value.startswith(('#','mailto:','data:')):return match[0]
        path,sep,anchor=value.partition('#')
        original=(old.parent/path).resolve()
        destination=mapping.get(original)
        if destination is None:return match[0]
        return match[1]+os.path.relpath(destination,new.parent)+(sep+anchor if sep else '')+match[3]
    return link.sub(target,text)
for old,new in mapping.items():
    if old.suffix=='.md':
        text=repair(old.read_text(),old,new)
        if S in old.parents:
            text='> 历史实验记录：保留该轮配置和结果；当前执行入口见项目 README。\n\n'+text
        put(new,text)
for root in (K/'analysis',K/'doc'):
    for path in root.glob('*.md'):
        if path in mapping:continue
        text=repair(path.read_text(),path,path)
        if text!=path.read_text():put(path,text)

put(K/'doc/design/project_structure.md','''# 算子项目组织与职责

## 目的
保留可读的 CUDA 实践代码，并提供一个模型项目能稳定导入的通用算子包。

## 执行方案
| 目录 | 职责 |
| --- | --- |
| dev/cuda | CUDA 实现、Torch C++ 绑定、共用 common.h；唯一正式 CUDA 源码 |
| dev/tilelang | TileLang 实践 |
| dev/pytorch/my_cuda_kernels | GEMM、EP 张量算子、FakeTensor/autograd 注册、工作区和库加载 |
| script | 编译、单算子 benchmark 和绘图入口 |
| tests | 单算子正确性、边界和包结构检查 |
| analysis / figures | 算子分析和配图 |
| doc/design / commands / results | 目的与闭环、可执行命令、实际结果 |
| build | 按算子与构建版本分开保存的编译产物，current 指向成功版本 |
| result | 单算子实验结果 |

通用包接受张量、counts 和工作区，不访问 MiniMind 模型对象、训练配置或 checkpoint。模型按普通 Python 依赖调用它，详见 [安装与测试命令](../commands/local_workflow.md)。

## 结果
模型路由/NCCL、专家模块缓存、完整 MLP 及 KN 转换已归到 MiniMind；正式构建只读取 dev/cuda。历史源码快照不参与默认构建或导入。

## 调整闭环
先做模块边界和接口检查，再在 CUDA 环境运行数值/安全检查，最后同轮测量 CUTLASS。数值契约、输入形状与库 ABI 变化时重新编译和验证；目录迁移不作为性能改进。
''')
put(K/'doc/commands/local_workflow.md','''# 安装、编译与验证命令

## 目的
用同一 Python 环境构建和加载通用算子，不依赖模型阶段目录。

## 执行方案
在项目根目录执行，先激活已有 PyTorch 环境。GPU 构建需要 CUDA PyTorch、nvcc 和匹配的 C++ 编译器；CXX 可指定 SCVideo 的 GCC12。

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m unittest tests.test_package -v
python script/build.py ep --dry-run
python script/build.py grouped_gemm --dry-run
python script/build.py kernel9 --dry-run

# CUDA 机器上执行实际编译
python script/build.py ep
python script/build.py grouped_gemm
python script/build.py kernel9

# EP 张量算子：输出目录每轮独立
python -m tests.ep.validate_functional --library build/ep/current/my_cuda_ep.so --output result/ep_functional.json
python -m tests.ep.validate_configuration --library build/ep/current/my_cuda_ep.so --output result/ep_configuration.json
python -m tests.ep.validate_ops --library build/ep/current/my_cuda_ep.so --output result/ep_ops.json

# 最新 Prefill/Decode API；GPU 测试会创建输出目录
python -m tests.verify_grouped_gemm --library build/grouped_gemm/current/my_cuda_grouped_split.so --output result/grouped_gemm_validation
```

EP 的 --output 是 JSON 文件，先创建其父目录；Grouped GEMM 的 --output 是新目录。工作区、counts、BF16 舍入及支持形状见 [Grouped GEMM API](../../dev/pytorch/README.md)。MY_CUDA_KERNELS_BUILD 可选择其他编译产物根目录，或显式向 load_library 传库路径。

## 结果
本地包安装、CPU 检查和三种编译命令预览已通过；本轮没有进行 GPU 编译/benchmark。

## 调整闭环
构建失败不切换 current；修正工具链后生成新版本。每次性能调整先验证数值，再复测；具体结果写到对应结果文档。
''')
put(K/'dev/pytorch/README.md','''# my_cuda_kernels Python API

在项目根目录安装 `python -m pip install --no-deps --no-build-isolation -e .`，同一解释器中的模型直接导入。

```python
from my_cuda_kernels.grouped_gemm import load_library, grouped_gemm, GemmWorkspace
load_library()  # build/grouped_gemm/current/my_cuda_grouped_split.so
y = grouped_gemm(x, packed_b, counts)
workspace = GemmWorkspace.allocate(x, packed_b, capacity=4096)
y = workspace.run(x, packed_b, counts)
```

x 是按专家打包的 [M,K]，packed_b 是连续 [E,K,N]；counts 是 CUDA int64 [E]，非负且总和为 M。输入/权重/输出为 BF16 或 FP16，FP32 累加；支持 E=1～16、(K,N)=(768,2432)/(2432,768)/(1024,1024)。M≤256 默认 Decode，其余 Prefill；动态专家分配来自 GPU counts。只提供前向，要求 inference_mode；FP32 累加的 BF16 输出与原 PyTorch BF16 中间归约有不同数值契约。

debug_counts=True 会同步检查 counts 值。工作区输出是复用视图，长期保留需 clone；独立 stream 需独立工作区和明确依赖。FakeTensor 和分配接口可编译，完整模型数值/通信不能由单算子测试替代。

```python
from my_cuda_kernels.ep_ops import load_library
from my_cuda_kernels.ep import Workspace, combine_inference
load_library()  # build/ep/current/my_cuda_ep.so
```

EP 包提供稳定分桶、pack/gather/combine 的 CUDA 张量操作及训练 autograd；Router、NCCL 和原专家运行由 MiniMind 编排。专家模块权重缓存与完整 MLP 位于 MiniMind 的 model/backends。

旧 kernel9 接口通过 my_cuda_kernels.kernel9 使用。构建与验证见 [命令](../../doc/commands/local_workflow.md)；模型接入见 [MiniMind](../../../minimind/readme_ep.md)。
''')
put(K/'guide.md','''# My CUDA Kernels 工作约定

- CUDA 源码统一放 dev/cuda，复用 common.h；练习版本保留可读的 native、开源对照与优化 kernel。
- 通用 Python API 放 dev/pytorch/my_cuda_kernels；不得导入 MiniMind 模型或读取模型 checkpoint。
- 编译和测量脚本放 script，单算子测试放 tests，构建输出放 build，测量输出放 result。
- 分析放 analysis，配图统一放 figures；每轮说明包含目的、执行方案、结果、调整闭环。
- 具体指令写 doc/commands，执行结果写 doc/results，方案写 doc/design；先验证数值和边界，再比较同条件耗时。
- 新脚本使用项目根目录相对路径或参数，不写死服务器用户名、日期阶段目录。正式构建只读取 dev/cuda，不复制 source 树。
- MiniMind 的训练、通信编排、权重保存、整模型测试归 MiniMind；两个项目用 editable Python 包连接，不修改 sys.path。

[项目组织](doc/design/project_structure.md) · [操作命令](doc/commands/local_workflow.md) · [目录迁移前的工作记录](doc/history/guide_before_20261009.md)
''')
old=(K/'readme.md').read_text()
table=old[old.index('| 实现 |'):old.index('各报告注明')]
table='\n'.join(line for line in table.split('\n') if not any(n in line for n in model_analyses))
put(K/'readme.md','''# My CUDA Kernels

实践 RMSNorm、Fused Residual RMSNorm、GEMM、MoE Grouped GEMM 与 EP 张量算子。CUDA 实现和共用 common.h 位于 dev/cuda；通用 Torch 包位于 dev/pytorch/my_cuda_kernels。

## 目录与入口

| 内容 | 入口 |
| --- | --- |
| CUDA / TileLang 实现 | dev/cuda / dev/tilelang |
| 通用 Python API | [API 说明](dev/pytorch/README.md) |
| 编译 / benchmark | script/build.py / script |
| 单算子测试 | tests |
| 方案 / 指令 / 结果 | [组织约定](doc/design/project_structure.md) · [命令](doc/commands/local_workflow.md) · [整理验收](doc/results/organization_20261009.md) |
| 分析 / 图表 | analysis / figures |
| 构建 / 测量 | build / result |

'''+table+'''
## 编译与调用

在项目根目录执行，使用已安装 PyTorch 的解释器。

```bash
python -m pip install --no-deps --no-build-isolation -e .
python script/build.py ep
python script/build.py grouped_gemm
python -m unittest tests.test_package -v

mkdir -p build/practice
nvcc -O3 -std=c++17 -arch=sm_89 dev/cuda/gemm_forward.cu -lcublas -o build/practice/gemm
./build/practice/gemm --dtype fp16 --kernel 9
nvcc -O3 -std=c++17 -arch=sm_89 dev/cuda/moe_grouped_gemm_forward.cu -lcublas -o build/practice/grouped_gemm
./build/practice/grouped_gemm --m 2048 --experts 16 --n 1024 --k 1024 --dtype fp16 --kernel 8
```

CUTLASS 对照追加 --expt-relaxed-constexpr -DUSE_CUTLASS -I "$CUTLASS_ROOT/include"。4090D 使用 sm_89。最新 Prefill/Decode 的数值、性能和适用范围见 [分析](analysis/grouped_gemm_prefill_decode.md) 与 [Decode 优化](analysis/grouped_gemm_decode.md)。

## 与 MiniMind 的关系

MiniMind 通过 `from my_cuda_kernels...` 调用通用包。Router/NCCL、完整专家 MLP、专家权重缓存、KN checkpoint、训练和整模型评估都在 [MiniMind](../minimind/readme_ep.md)。通用算子不依赖模型仓库。训练目前使用原 PyTorch 专家与 compact EP；CUDA GEMM 全层推理仍需独立数值准入。
''')

put(M/'doc/ep/design/project_structure.md','''# MiniMind 与算子项目的关联

## 目的
让模型实现、训练入口和实验结论都能在 MiniMind 内找到，稳定调用独立算子包。

## 执行方案
| 目录 | 职责 |
| --- | --- |
| model/model_minimind.py | 模型、原 Router/专家、EP 模型定义 |
| model/ep/routing.py | MiniMind 路由、NCCL、负载均衡损失和专家调用 |
| model/backends | 专家权重缓存、完整 MLP、可选 Grouped GEMM 适配 |
| model/checkpoint_kn.py | 模型 KN 权重转换/加载/保存 |
| trainer | 实际训练器 |
| configs/ep | 独立可编辑的专家数/Top-K 和训练配置 |
| scripts/ep | 训练/tmux、benchmark、绘图和权重转换入口 |
| tests/ep | 模型、分布式梯度与配置边界测试 |
| analysis / figures | EP、MLP、模型分析与配图 |
| doc/ep/design / commands / results | 目的方案闭环、具体指令、结果 |
| runs/<run_id> | config、logs、checkpoints/full、checkpoints/resume、state |

两个仓库分别为同级 minimind 和 my_cuda_kernels。用训练所用解释器安装后者的 editable 包；调用通过普通 import，库默认由包解析，也可用 library 参数覆盖。算子仓库不导入模型，MiniMind 不复制 CUDA 源码或通过 sys.path 加载阶段文件。

训练路径：原 Router → compact CUDA pack/gather → NCCL → 原 PyTorch 专家 → NCCL → 融合 combine → 原辅助损失/反向/梯度同步。每层每步重新路由；工作区只缓存空间。推理 Grouped GEMM 是独立的显式后端选项，不混入本次训练入口。

## 结果
当前 API、模型适配、脚本和文档已按职责归位。历史阶段快照只用于理解对应实验；当前入口不导入快照。

## 调整闭环
目录修改先检查包导入、模型参数键和权重布局，再跑 CPU 契约测试；CUDA 环境补充算子与多卡训练回归。模型全层 BF16 误差问题继续保留为独立准入项；本轮不改变数值算法或服务器运行版本。
''')
put(M/'doc/ep/commands/local_workflow.md','''# MiniMind EP 操作命令

## 目的
在 MiniMind 根目录完成依赖安装、验证、独立训练与评估。

## 执行方案
以下命令在 MiniMind 根目录使用同一个已有 PyTorch 环境执行。

```bash
python -m pip install --no-deps --no-build-isolation -e ../my_cuda_kernels
python ../my_cuda_kernels/script/build.py ep
python -m unittest tests.ep.test_local_contracts -v

# 仅查看启动配置，不执行训练
python -m scripts.ep.train --run-id ep6_top2_new --gpus 2,3 --tmux --dry-run

# 先查看 GPU 占用；从 0..3 或 4..7 同一组内选择两张
nvidia-smi
python -m scripts.ep.train --run-id ep6_top2_new --gpus 2,3 --tmux
tmux attach -t ep6_top2_new

# 单独验证可配置专家数与 Top-K；选择实际空闲卡
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --standalone --nproc_per_node=2 --module tests.ep.validate_configurations \
  --experts 6 --topks 1 2 3 --library ../my_cuda_kernels/build/ep/current/my_cuda_ep.so --output runs/ep_config_validation

# 模型同输入基线对照；不传 checkpoint 则使用初始化模型
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --standalone --nproc_per_node=2 --module tests.ep.validate_model \
  --experts 4 --checkpoint out/pretrain_ep_768_moe.pth \
  --library ../my_cuda_kernels/build/ep/current/my_cuda_ep.so --output runs/ep_model_validation

python -m scripts.ep.run_benchmark --gpus 2,3 --experts 4 --checkpoint out/pretrain_ep_768_moe.pth \
  --library ../my_cuda_kernels/build/ep/current/my_cuda_ep.so --output runs/ep_benchmark

# 转换完整模型参数，显式提供该模型的配置；不转换优化器恢复状态
python -m scripts.ep.convert_checkpoint_kn --input out/pretrain_ep_768_moe.pth \
  --output out/pretrain_ep_768_moe_kn.pth --config path/to/training_config.json --report runs/kn_conversion.json
```

默认训练配置是 configs/ep/pretrain_6_top2.json。调整专家数和 Top-K 可复制配置后传 --config；总专家数必须能被两 Rank 整除。新的 run-id 独立保存模型和两 Rank 恢复状态，已存在的运行目录拒绝覆盖。CXX/CC 使用已选工具链，可指定 SCVideo GCC12；启动器通过 tmux 传递该选择。缺库或配置非法直接报错，无训练回退。

每次运行包含 config、logs、checkpoints/full、checkpoints/resume、state。训练结束检查 state/training_exit_code.txt=0、完整模型目录中 training_summary.json 的 completed_epochs=2 和所有 Rank routing_fallbacks=0，另确认完整模型及两份 Rank 恢复文件存在。退出成功与模型质量验证分开。

## 结果
本轮通过本地 CPU 契约与训练命令预览；未启动新训练，正在运行的服务器训练沿用其启动版本。

## 调整闭环
修改路由后先跑算子和多卡边界测试，再跑真实文本短训练及恢复测试，最后独立正式训练。更改动态通信/计数路径必须重新测量模型整体耗时；编译边界变化先验证梯度与参数保存。
''')
environment='''# 本轮整理环境

## 目的
区分本地目录/导入验证和服务器 CUDA 数值/性能验证。

## 执行方案
本地：macOS，Python 3.13.12，PyTorch 2.10.0，Transformers 5.3.0；解释器为 /opt/homebrew/Caskroom/miniconda/base/envs/py313/bin/python。使用已有环境，editable 安装不安装额外依赖。

服务器历史实验：RTX 4090D、CUDA PyTorch 2.9.0+cu128、Python 3.13.5，CUDA Toolkit 12.4，SCVideo GCC12.4。服务器编译选择 CXX，部署时必须使用训练所用解释器重新构建并验证 ABI。

## 结果
本轮完成 CPU 导入、权重转换/缓存与配置测试以及编译/训练命令预览。本地没有 CUDA GPU，本轮不计新增 CUDA 编译、安全或性能测试结果。

## 调整闭环
更换解释器、PyTorch ABI、CUDA 或源码后重新构建；GPU 回归单独记录。详细历史条件与当轮测量数据一起保留，避免跨条件比较。
'''
put(K/'doc/environment.md',environment)
put(M/'doc/ep/environment.md',environment+'\n[历史测试环境](environment_history_20261007.md)\n')
api=(K/'analysis/ep_routing_api.md').read_text().replace('    python_dir="/absolute/path/my_cuda_kernels/dev/pytorch",\n','')
put(M/'analysis/ep_routing_api.md',api)
readme=(M/'readme_ep.md').read_text()
readme=readme[:readme.index('## 17. 可选 Prefill / Decode Grouped GEMM 推理')]
readme+='''## 本地开发入口

模型相关修改统一在本项目，CUDA 张量操作通过独立 my_cuda_kernels 包调用。

| 内容 | 入口 |
| --- | --- |
| 项目职责和目录 | [组织说明](doc/ep/design/project_structure.md) |
| 安装、验证、训练与 tmux | [具体指令](doc/ep/commands/local_workflow.md) |
| 本轮整理验收 | [结果](doc/ep/results/organization_20261009.md) |
| 测试环境 | [环境记录](doc/ep/environment.md) |
| 路由实现 | model/ep/routing.py |
| 训练配置 / 启动 | configs/ep/pretrain_6_top2.json / scripts/ep/train.py |
| 模型测试 | tests/ep |
| EP 数值和性能 | [EP 优化](analysis/ep_routing_optimization.md) · [API](analysis/ep_routing_api.md) |
| Grouped GEMM 模型接入 | [接入分析](analysis/grouped_gemm_split_integration.md) |
| 权重布局 | [KN 分析](analysis/minimind_kn_checkpoint.md) |

```bash
python -m pip install --no-deps --no-build-isolation -e ../my_cuda_kernels
python ../my_cuda_kernels/script/build.py ep
python -m unittest tests.ep.test_local_contracts -v
python -m scripts.ep.train --run-id ep6_top2_new --gpus 2,3 --tmux --dry-run
```

默认训练：6 专家 Top-2、hidden768、8层、batch32/Rank、seq340、BF16、LR5e-4、累积8、裁剪1、2epochs，原数据从头预训练。使用原 PyTorch 专家与 compact EP，没有训练失败后的自动回退。专家数和 Top-K 可在配置文件调整；缺库或非法配置直接报错。

专家及模型张量计算可 compile；动态 counts、NCCL splits 与专家选择保留明确边界。每轮使用独立 runs/<run_id>，保存配置、日志、完整模型、两 Rank 恢复状态和退出码。

## 可选 Grouped GEMM 推理

model/backends 提供完整专家 MLP 与 CUDA Prefill/Decode 适配，model/checkpoint_kn.py 提供转换/加载。默认 expert_backend=original；native_bf16 保持原专家结果，cuda_split_last_layer 是受限推理路径，cuda_split 全层仍是实验选项。FP32 累加并输出 BF16 与原 BF16 中间归约存在差异，完整模型数值准入未通过。

```bash
python ../my_cuda_kernels/script/build.py grouped_gemm
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --standalone --nproc_per_node=2 eval_llm.py \
  --use_ep --weight pretrain_ep --hidden_size 768 --num_hidden_layers 8 \
  --num_experts 4 --num_experts_per_tok 1 --model_dtype bf16 --expert_backend cuda_split_last_layer
```

库默认由已安装的算子包解析；可传 grouped_gemm_library 覆盖。普通 NK 权重会在推理缓存阶段打包；显式 kn_checkpoint 加载转换后权重。专家 GEMM 只有前向，当前正式训练仍用原 PyTorch 专家。
'''
readme=readme.replace('## 1. 项目概述','[当前开发与训练入口](#本地开发入口) · [项目组织](doc/ep/design/project_structure.md)\n\n## 1. 项目概述',1)
put(M/'readme_ep.md',readme)
put(S/'README.md','''# MiniMind / CUDA 项目入口

本地正式代码、脚本和实验文档已按职责迁入两个项目。

| 项目 | 入口 | 职责 |
| --- | --- | --- |
| my_cuda_kernels | [项目说明](/Users/tongleiwen/Code/my_cuda_kernels/readme.md) | CUDA 算子、Python API、单算子验证与性能 |
| minimind | [EP 项目说明](/Users/tongleiwen/Code/minimind/readme_ep.md) | 模型、路由/NCCL、MLP、权重、训练与整体评估 |

使用同一 Python 环境安装 my_cuda_kernels 的 editable 包，MiniMind 通过普通 import 调用。服务器当前训练继续使用已启动版本。
''')
workspace=(W/'README.md').read_text()
workspace=workspace.replace('## 常用入口','## 常用入口\n\n- [MiniMind / CUDA 项目入口](minimind_ep_grouped_gemm/README.md)：两个项目的代码、命令和实验文档按职责归位。',1)
put(W/'README.md',workspace)

# Preserve originals of changed docs before moving; explicit inventory is reviewable.
B.mkdir(parents=True)
inverse={new:old for old,new in mapping.items()}
for path in writes:
    original=inverse.get(path,path)
    if original.exists():
        root=K if original.is_relative_to(K) else M if original.is_relative_to(M) else W
        previous=B/'previous'/root.name/original.relative_to(root)
        previous.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,previous)
for source,target in moves:
    target.parent.mkdir(parents=True,exist_ok=True);shutil.move(source,target)
for path,text in writes.items():
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
record=dict(moves=[dict(source=str(a),target=str(b)) for a,b in moves],
            writes=[str(p) for p in writes], file_destinations={str(a):str(b) for a,b in mapping.items()})
(B/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
(B/'cuda_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (K/'dev/cuda').iterdir() if p.is_file()},indent=2)+'\n')
print('DOCUMENTS ORGANIZED',len(moves),'recoverable moves;',len(mapping),'mapped files;',len(writes),'updated documents')
