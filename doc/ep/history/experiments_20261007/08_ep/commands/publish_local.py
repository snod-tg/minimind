"""Publish only validated EP additions, with original-source guards."""
from pathlib import Path
import hashlib
import shutil

workspace=Path('/Users/tongleiwen/Documents/resume/minimind_ep_grouped_gemm')
stage8=workspace/'commands/08_ep'
stage9=workspace/'commands/09_compile'
kernels=Path('/Users/tongleiwen/Code/my_cuda_kernels')
minimind=Path('/Users/tongleiwen/Code/minimind')
guards={'model/model_minimind.py':'c7a344373dff56099c15cbb45a1880a6990c512adc0f45366a277884451b5a2c',
        'trainer/train_pretrain.py':'344473d719dca71524fbf6c81adea7bbcbf7500b3035391c4089b1d223dcb395'}
for relative,expected in guards.items():
    assert hashlib.sha256((minimind/relative).read_bytes()).hexdigest()==expected,relative
assert (kernels/'dev/cuda/common.h').read_bytes()==(stage8/'source/dev/cuda/common.h').read_bytes()
new=[]
for relative in ['dev/cuda/ep_routing.cu','dev/cuda/ep_routing_torch.cpp','dev/pytorch/minimind_ep_routing.py','dev/pytorch/ep_routing_ops.py']:
    new.append((stage8/'source'/relative,kernels/relative))
for path in (stage8/'publication/analysis').glob('*.md'):
    new.append((path,kernels/'analysis'/path.name))
new.append((workspace/'results/08_ep/figures/ep_routing_optimization.png',kernels/'figures/ep_routing_optimization.png'))
new.append((stage8/'minimind/model/ep_routing_backend.py',minimind/'model/ep_routing_backend.py'))
for source,target in new:
    assert not target.exists(),f'refuse to overwrite new file: {target}'
for relative in guards:
    backup=stage8/'before_ep_routing'/relative
    backup.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(minimind/relative,backup)
for source,target in new:
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
shutil.copy2(stage8/'minimind/model/model_minimind.py',minimind/'model/model_minimind.py')
shutil.copy2(stage9/'minimind/trainer/train_pretrain.py',minimind/'trainer/train_pretrain.py')
for source in stage8.glob('*.py'):
    if source.name=='publish_local.py':continue
    target=kernels/'script/ep_routing'/source.name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
for source in stage8.glob('*.sh'):
    shutil.copy2(source,kernels/'script/ep_routing'/source.name)
for name in ['readme.md','guide.md']:
    with (kernels/name).open('a') as output:
        output.write('\n\n## 20261009：原生精度 EP 路由优化\n\n稳定分桶、向量化打包、融合加权合并、推理缓冲复用及可选固定容量 Decode；训练保留原 PyTorch 专家。专家数和 Top-K 可配置，Top-1/2 通过逐元素输出/梯度对照，通用路径三类安全检查通过。分析见 [EP 优化](analysis/ep_routing_optimization.md)，接口见 [EP API](analysis/ep_routing_api.md)。\n')
        if name=='guide.md':
            output.write('''
```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage08_ep_20261009
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
python build_api.py --source source --output results/REPRO_ep_build
CUDA_VISIBLE_DEVICES=1 python validate_configuration.py --library results/REPRO_ep_build/my_cuda_ep.so --output results/REPRO_ep_configuration.json
bash run_configurations.sh results/REPRO_ep_build/my_cuda_ep.so
python run_measured_variants.py --library results/REPRO_ep_build/my_cuda_ep.so --output results/REPRO_ep_variants --gpus 1,3 --rounds 21
python plot_ep.py --input results/REPRO_ep_variants --output results/figures/REPRO_ep_routing.png
```
''')
with (minimind/'readme_ep.md').open('a') as output:
    output.write('''

## 原生精度优化 EP 与 6 选 2 compile 训练

通过 `--num_experts`、`--num_experts_per_tok` 和 torchrun Rank 数配置路由；专家总数必须被 Rank 数整除。`--ep_routing compact` 显式使用优化路由，缺库和不合法训练配置报错。指定 `--ep_routing_library` 和 `--ep_routing_python_dir`；保留原 Router、原 PyTorch 专家及 [N,K] 权重。

EP 允许 `--use_compile 1`，编译专家及模型张量计算；动态 counts、NCCL splits 和专家选择保留明确边界。没有训练失败后的自动回退。推理可通过 `model.ep_routing_backend.attach_ep_routing` 选择 compact/fixed，固定容量可用 `decode_capacity_tokens` 设置。

6 选 2 正式训练沿用 hidden768、8层、batch32/Rank、seq340、BF16、LR5e-4、累积8、裁剪1、2epochs、原预训练数据，从头初始化。独立 tmux 会话 `ep6_top2_compile_20261009`；新权重前缀 `pretrain_ep6_top2_compile_20261009`，不会覆盖原4专家模型。启动入口：

```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage09_compile_20261009
bash launch_tmux.sh
tmux attach -t ep6_top2_compile_20261009
```
''')
print('VALIDATED EP PUBLISHED')
