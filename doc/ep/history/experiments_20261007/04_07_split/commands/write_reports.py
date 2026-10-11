"""Write the latest phase reports from measured JSON; no historical latency mixing."""
import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2];work=Path(__file__).resolve().parent;r=root/'results/04_07_split'
read=lambda p:json.loads((r/p).read_text())
perf=read('R15_clean_model/summary_paired.json');mlp=read('R11_mlp_benchmark/benchmark.json');api=read('R06_api_benchmark/benchmark.json');build=read('R16_release_build/build.json')
assert len(list((r/'R17_actual_cli').glob('cli_rank*.json')))==2
cli=[x for p in (r/'R17_actual_cli').glob('cli_rank*.json') for x in json.loads(p.read_text())];assert len(cli)==32
modeltable='| 输入 / Top-K | 原实现 ms | 原数值兼容 ms | 最后一层小 M CUDA ms |\n| --- | ---: | ---: | ---: |\n'
rawtable='| 输入 / Top-K | 原实现 ms | 全层 CUDA ms（数值未准入） |\n| --- | ---: | ---: |\n'
for x in perf['model']:
 v={k:p['wall']['median_ms'] for k,p in x['paths'].items()};modeltable+=f"| {x['case']} | {v['original']:.3f} | {v['native_bf16']:.3f} | {v['cuda_split_last_layer']:.3f} |\n";rawtable+=f"| {x['case']} | {v['original']:.3f} | {v['cuda_split']:.3f} |\n"
growth='| Top-K / batch | 原实现 ms | 最后一层小 M CUDA ms |\n| --- | ---: | ---: |\n'
for x in perf['growing_decode']:
 growth+=f"| {x['topk']} / {x['batch']} | {x['paths']['original']['wall']['median_ms']:.3f} | {x['paths']['cuda_split_last_layer']['wall']['median_ms']:.3f} |\n"
mlptable='| local M / counts | 原 MLP Graph μs | CUDA MLP Graph μs | 原 MLP eager μs | CUDA MLP eager μs |\n| --- | ---: | ---: | ---: | ---: |\n'
for x in mlp:mlptable+=f"| {x['m']} / {x['counts']} | {x['graph_us']['original']:.3f} | {x['graph_us']['cuda_split']:.3f} | {x['eager_wall_us']['original']:.3f} | {x['eager_wall_us']['cuda_split']:.3f} |\n"
apitable='| 场景 | 同 kernel 原生调用 Graph μs | 工作区 API Graph μs | 分配 API Graph μs |\n| --- | ---: | ---: | ---: |\n'
for x in api:apitable+=f"| {x['case']} | {x['graph_us']['native_full']:.4f} | {x['graph_us']['api_out']:.4f} | {x['graph_us']['api_functional']:.4f} |\n"
env='''```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04_07_split_20261009
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 BENCH_CPU_BASE=28
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
LIB=results/R16_release_build/my_cuda_grouped_split.so
CKPT=/data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth
BASE=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007
```'''
commands=[
'''```bash
python build_api.py --source source --output results/REPRO_release_build
CUDA_VISIBLE_DEVICES=4 python source/script/verify_api.py --source source --library "$LIB" --output results/REPRO_api
CUDA_VISIBLE_DEVICES=4 python source/script/verify_extra_api.py --source source --library "$LIB" --output results/REPRO_extra_api.json
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 97 python source/script/check_api_sanitizer.py --source source --library "$LIB"
CUDA_VISIBLE_DEVICES=4 python source/script/benchmark_api.py --source source --library "$LIB" --native "$BASE/stage03_weak_20261009/results/R23_release_build/split_bridge.so" --output results/REPRO_api_benchmark
```''',
'''```bash
CUDA_VISIBLE_DEVICES=4 python source/script/replay_split_mlp.py --library "$LIB" --checkpoint "$CKPT" --baseline "$BASE/results/R04_baseline" --extra "$BASE/stage05/results/R01_extra_capture" --output results/REPRO_mlp
CUDA_VISIBLE_DEVICES=4 python source/script/benchmark_mlp.py --library "$LIB" --checkpoint "$CKPT" --output results/REPRO_mlp_benchmark
```''',
'''```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 validate_ep.py --source source --library "$LIB" --checkpoint "$CKPT" --repo /data/user4/tongleiwen/minimind --baseline "$BASE/results/R04_baseline" --output results/REPRO_ep
python -m torch.distributed.run --standalone --nproc_per_node=2 verify_kn_integration.py --source source --library "$LIB" --checkpoint "$CKPT" --kn "$BASE/stage04_weights/results/R01_convert/pretrain_ep_768_moe_kn.pth" --output results/REPRO_kn
python -m torch.distributed.run --standalone --nproc_per_node=2 verify_cli.py --actual --repo /data/user4/tongleiwen/minimind --library "$LIB" --kn-checkpoint "$BASE/stage04_weights/results/R01_convert/pretrain_ep_768_moe_kn.pth" --output results/REPRO_actual_cli
```''',
'''```bash
mkdir results/REPRO_model
python monitor_gpu.py --gpus 4,5 --output results/REPRO_model/gpu_monitor.jsonl --stop results/REPRO_model/monitor.stop &
MONITOR_PID=$!
python -m torch.distributed.run --standalone --nproc_per_node=2 benchmark_model.py --stage6 . --library "$LIB" --checkpoint "$CKPT" --repo /data/user4/tongleiwen/minimind --baseline "$BASE/results/R04_baseline" --output results/REPRO_model
touch results/REPRO_model/monitor.stop
wait "$MONITOR_PID"
python analyze_model.py --input results/REPRO_model --output results/REPRO_model/summary_paired.json
python plot_results.py --results results --model-summary results/REPRO_model/summary_paired.json --output figures/grouped_gemm_split_integration.png
```''']
# Exact arguments used by verify_api are asserted before documents are written.
assert '--library' in (work/'source/script/verify_api.py').read_text()
names=['04_分拆算子封装','05_分拆专家MLP','06_分拆EP接入','07_分拆整体评估']
purposes=['将最终Prefill/Decode算子注册为前向API，减少调用和工作区分配成本，独立保留原BF16数值契约。','连接gate、SiLU、up、乘法、down完整专家MLP，验证真实输入和容量复用，判断单算子收益是否能传递到MLP。','接入MiniMind真正的本地专家扩展点，验证路由、双卡边界、模型传播、KN加载和训练回退。','在相同输入、权重和通信条件下测整体EP与模型，区分数值准入和性能收益。']
plans=[
'注册prefill/decode及各自out接口，GPU counts调度；分配版返回独立输出，工作区版复用缓冲区。支持BF16/FP16，FP32累加，三组固定K/N，E≤16。结构、别名、容量、设备、stream检查保留；counts数值正常路径由调用方保证，debug_counts提供同步检查。先接口数值与opcheck，再Graph/compile及sanitizer，最后对同一kernel测封装成本。',
'一次打包每个专家的三份连续KN权重，保留原BF16 SiLU/乘法舍入边界。每条stream持有独立工作区，最小256行并按2的幂扩容；一次前向只核查一次权重版本。用固定100份真实专家输入对拍FP32投影和完整链，再测真实权重/合成输入下Graph与eager延迟；它们不作为原BF16契约的等价替代。',
'在GPU bincount之后增加可选后端，不改路由、All-to-All、逆排序和combine。默认original；native_bf16保持原专家；cuda_split全部支持层启用新算子；cuda_split_last_layer仅最后一层且local M≤256启用。分别测试所有8层、空Rank/空源/同Rank与跨Rank Top2/偏斜尾块、固定KV与逐步生成，并检查训练输出及输入/参数梯度回退逐位一致。CLI额外覆盖BF16权重和KN直接加载。',
'完整模型10场景、最后一层完整EP10场景、增长KV 4场景；每后端静态60次、5轮轮换，增长KV每样本8步、20样本。计时外barrier，每次先取两Rank较慢wall再统计中位数。模型测量不抓数据、不使用Graph；图上MLP Graph与eager和模型wall分开呈现。首次准备与正式热测量分开。']
results=[
'R02接口100项检查通过（含72项数值、16项动态counts Graph）；R12及最终R16补充Decode/out opcheck、compile自动256/257切换与Decode越界拒绝通过。R10的7个接口memcheck场景ERROR SUMMARY=0。目标M137同kernel原生8.2016μs、工作区API8.2080μs，Graph中封装差约0.08%。\n\n'+apitable,
'最终R14：100真实MLP、300投影FP32对拍/CPU double抽查，完整FP32链bad=0；15项补充含Graph动态counts、容量边界和KN直接打包。对原默认BF16另有3份MLP未通过原误差门槛，未扩大阈值。R11使用真实权重和合成输入，Graph32节点、7轮轮换×10重放；eager每轮64调用。\n\n'+mlptable,
'R08双卡：60完整模型对照、480隔离EP诊断、36边界对照、24生成对照，训练回退输出/输入/参数梯度逐位一致。native_bf16的20个模型对照逐位一致；last-layer的20个对照通过原阈值且next token不变，8步greedy所有样本无token变化。全层CUDA 20个模型对照未过准入，模型rank1有3处next token变化；8步greedy另有rank0的一处变化。独立EP诊断不等于全模型通过。R05的NK/KN各后端logits/KV逐位一致，144个rank Decode步骤通过，FP16 CLI精度12项验证完成。最终R17真实eval init_model：双Rank共32组合，BF16、NK/KN、Top1/2、4后端均可调用；native保持逐位一致，last-layer满足阈值。',
'正式结论使用R15独立复测，106次GPU监控只见两测试Rank。最后一层小M的完整EP局部有收益，但模型静态基本持平（最明显单步退化约0.98%）；8步增长KV未形成稳定加速。全层CUDA部分场景快约3%，但数值未准入，不能宣传为可用的整体模型加速。\n\n'+modeltable+'\n增长KV包含8次真实cache增长和argmax，context从128起：\n\n'+growth+'\n实验全层结果，仅用于诊断：\n\n'+rawtable]
loops=[
'接口设计→验证→测量→调整。保持最新版CUDA源码不变；移除重复CPU权重版本核查、共享可复用输出与metadata、为空Rank跳过无效缓存准备。重建脚本只快照所需dev目录，避免输出位于源码目录时递归复制。最终R16从冻结源码重建、额外API与实际CLI复测通过。',
'R03的测试导入根路径不正确，修正后R03b通过；随后改为每stream容量复用、权重record_stream和一次版本检查，最终R14再次完整回放通过。数值闭环保持原BF16与FP32两份参考；3份原结果差异如实保留。M1 Graph虽变快，eager稍慢，继续以调用粒度决定使用范围。',
'R04曾错误要求native BF16与FP32 oracle相同，改为native逐位对拍原模型，CUDA单独对拍FP32参考，原模型阈值保持不变。全层误差传到下一层Router后产生路由变化，限定last-layer避免后续专家路由传播。R13发现eval在绝对save_dir前加./，修为os.path.join；Top2验证必须修改实际config，复测R13b/R17完成。全层CUDA不会因封装完成而自动成为默认，未启动重新训练。',
'初轮R09监控出现额外GPU进程，正式性能改用独立R15，106次样本无额外进程。工作区复用后完整MLP明显受益，但当前EP排序、counts交换、CPU同步、8层Python调用和通信仍占模型延迟；本轮未声称使用全EP Graph。下一轮优先减少路由元数据同步和Python提交，再考虑通信重叠；全层启用前先解决原BF16兼容性。']
for idx,name in enumerate(names):
 links=f'[说明](../docs/{name}.md) · [指令](../commands/{name}.md) · [结果](../results/{name}.md)'
 text=f'# {name[3:]}\n\n## 目的\n{purposes[idx]}\n\n## 执行方案\n{plans[idx]}\n\n具体可复现指令见[指令篇](../commands/{name}.md)。\n\n## 结果\n{results[idx]}\n\n## 调整闭环\n{loops[idx]}\n\n{links}\n'
 (root/'docs'/f'{name}.md').write_text(text)
 (root/'results'/f'{name}.md').write_text(text.replace(f'# {name[3:]}',f'# {name[3:]}实测结果',1))
 text=f'# {name[3:]}执行指令\n\n## 目的\n{purposes[idx]}\n\n## 执行方案\nServer-Internet，下面变量在同一shell设置；REPRO目录必须使用未存在的新名字。最终构建R16；计时使用R01，其CUDA/C++源码完全相同。\n\n{env}\n\n{commands[idx]}\n\n## 结果\n{results[idx]}\n\n## 调整闭环\n{loops[idx]}\n\n{links}\n'
 (root/'commands'/f'{name}.md').write_text(text)
analysis=f'''# Prefill/Decode封装与MiniMind-EP接入（2026-10-09）

BF16/FP16输入输出、FP32累加的Prefill/Decode已注册并接入真正EP专家扩展点；默认原模型保持不变。单算子和完整专家MLP通过FP32参考，但全层CUDA仍未通过原BF16模型数值准入。最后一层local M≤256的选项通过现有模型/生成检查，整体速度基本持平。

## 实现与使用
独立注册`torch.ops.my_cuda_kernels_split.prefill/decode`及out版本；单次kernel读取GPU counts，local M≤256默认Decode，否则Prefill。一次打包连续KN权重，三GEMM保留BF16激活/乘法舍入边界；工作区按stream隔离、容量复用，版本变化时重建。训练和普通no_grad调用使用原专家。输出工作区视图会被下次调用覆盖。接口与限制见[调用说明](../dev/pytorch/README_grouped_split.md)。

支持E=1～16，(K,N)=(768,2432)/(2432,768)/(1024,1024)。Graph/compile验证针对算子和本地MLP，整个动态EP尚未支持全链路Graph。KN/NK按专家id保持顺序，原模型权重键不因注册后端增加。

## 数值与安全
接口100项通过，补充显式Decode/out opcheck、动态compile 256/257切换通过；memcheck7场景无错误。最终100真实MLP包含300投影全输出FP32检查及CPU double抽查，完整FP32链bad=0。原默认BF16另有3个完整MLP超过原门槛。

双卡60个完整模型对照中，native的20个逐位一致；last-layer的20个通过`abs_error≤0.02+0.02*abs(ref)`且relative RMS≤0.005，下一token不变，8步greedy全部样本无token变化。全层CUDA20个对照数值未准入，rank1的3处下一token变化、8步greedy另有rank0的一处变化，不能默认启用。480隔离EP是诊断对照，36边界对照验证原路由与FP32目标；训练回退梯度与原实现逐位一致。

NK/KN对应后端logits/KV逐位一致（144 rank Decode步骤）；实际eval入口32组合覆盖双Rank、BF16、Top1/2、NK/KN和4后端。CLI的BF16参数加载单独验证；下面整体基线沿用FP32参数+BF16 autocast，二者不混合计时。

## 封装成本：同一kernel，Graph
RTX4090D，BF16，热缓存，64节点，7轮轮换×10重放。这里原生对照是同CUDA kernel的native bridge，检验封装成本，不是额外开源性能基准。CUDA核心与此前CUTLASS对照版本保持相同。

{apitable}
M137工作区接口较原生调用约0.08%差异；eager含Python提交空隙，不能当作纯kernel时间。

## 完整MLP
真实checkpoint权重、合成BF16输入，E=2，hidden768/intermediate2432；Graph32节点、7轮×10重放，eager64调用/轮。原模块沿用默认BF16归约，新CUDA采用FP32累加，数值契约不同。

{mlptable}
M137 Graph延迟降低37.77%，eager降低51.65%。M1的eager反而略慢。不能将这些合成输入MLP数字代替真实全模型结果。

## 模型与EP整体
RTX4090D双卡4/5、8层/4专家、每Rank2专家、native All-to-All、原checkpoint。与基线相同真实输入，每后端60次、5轮轮换，先逐样本取慢Rank wall再统计中位数；不使用Graph。R15的106次GPU监控只见两个测试Rank，频率未锁定。最后一层小M仅在局部完整EP受益；模型基本持平。

{modeltable}
增长KV：每样本8个Decode步骤、20样本，context从128起，含argmax与cache增长：

{growth}
全层CUDA尚未数值准入，以下只作实验诊断：

{rawtable}
![分拆封装与整体性能](../figures/grouped_gemm_split_integration.png)

## 闭环与结论
优化依次处理版本检查、容量复用、独立stream缓冲区和空Rank缓存准备。封装没有明显增加纯kernel时间，MLP性能收益可观；模型整体仍受路由准备、CPU同步、Python提交和EP通信影响。保留原模型默认与原数值兼容选项；最后一层小M通过当前测试，不能推断任意数据均等价。下一步先解决全层BF16契约，再减少路由元数据同步和Python调用。未重新训练，现有算子只实现前向。
'''
(work/'publish/analysis').mkdir(exist_ok=True);(work/'publish/analysis/grouped_gemm_split_integration.md').write_text(analysis)
latest=f'''\n\n## 最新第4～7步：Prefill/Decode封装与接入（2026-10-09）
[04封装](04_分拆算子封装.md) → [05完整MLP](05_分拆专家MLP.md) → [06真实EP接入](06_分拆EP接入.md) → [07整体评估](07_分拆整体评估.md)。已完成注册、工作区/缓存、真实100MLP、双卡模型/边界/训练回退、NK/KN和实际CLI验证。新后端已经接入Code/minimind，默认original；全层CUDA数值未准入，last-layer小M通过当前检查。MLP明显收益尚未稳定传到整体模型，未重新训练。此前kernel9接入数据作为历史，本节是最终Prefill/Decode版本结论。
'''
p=root/'docs/00_总安排.md';s=p.read_text();tag='\n\n## 最新第4～7步';s=s.split(tag)[0];p.write_text(s+latest)
p=root/'测试环境.md';s=p.read_text();tag='\n\n## 最新分拆封装与接入环境';s=s.split(tag)[0];s+=f'''\n\n## 最新分拆封装与接入环境（2026-10-09）
Server-Internet，RTX4090D物理GPU4/5、CPU28/29；TileLang venv Python3.13.5、torch2.9.0+cu128/CUDA12.8，SCVideo GCC12.4，nvcc12.4.99，sm89/O3/C++17，ABI=true，无fast-math。原BF16 reduced-precision=True，FP32参考TF32=False。原checkpoint SHA25637f42844c386e8aca7191c8fbcc6f39427da8f45bfb2a016cbf0ca3cbc913d0c；KN SHA256c2159fbff1813f3c0a60c6272dbc0862e279f1026f111672845c973f7134ccf5。

最终API库R16 SHA256 `{build['sha256']}`；原R01计时库SHA2566bba535283b9058bfb679ca8b3e2985793be0cb94b2e96fd0361c8da18f4dc71，核心CUDA/C++源码相同。模型扩展点SHA256c7a344373dff56099c15cbb45a1880a6990c512adc0f45366a277884451b5a2c，实际本地/服务器/回放快照一致。Prefill/common与第四轮不变，Decode SHA256614ed616e8fc4d3461c82b354f8c4d27833b9e2ebb573b9504fd7f6097054b05。

R02接口100项，R08最终模型与训练回退，R10接口memcheck，R14最终100MLP，R15独立模型测量，R16冻结源码重建与显式Decode额外API，R17实际仓库CLI。R09监控有额外进程，所以正式结论使用R15；R15的106样本只见两个测试Rank。模型参数FP32、BF16 autocast用于基线对照；CLI另外验证BF16参数直接加载及FP16原默认选项。模型wall与MLP Graph/eager计时分开；未锁频、不宣称整机独占、无全EP Graph。未安装新依赖、未训练或改变默认BF16归约。
''';p.write_text(s)
print('REPORTS COMPLETE')
