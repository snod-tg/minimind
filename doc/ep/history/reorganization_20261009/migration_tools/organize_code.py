from pathlib import Path
import ast
import hashlib
import json
import shutil
import textwrap

k=Path('/Users/tongleiwen/Code/my_cuda_kernels')
m=Path('/Users/tongleiwen/Code/minimind')
old=k/'dev/pytorch'
pkg=old/'my_cuda_kernels'
assert not pkg.exists(), 'already organized'
backup_k=k/'doc/history/reorganization_20261009'
backup_m=m/'doc/ep/history/reorganization_20261009'
assert not backup_k.exists() and not backup_m.exists(), 'existing recovery record'
source_names=['grouped_gemm_split.py','grouped_gemm_api.py','grouped_gemm_compat.py',
              'grouped_expert_mlp_split.py','grouped_expert_mlp.py','minimind_ep_split_backend.py',
              'minimind_ep_backend.py','checkpoint_kn.py','minimind_ep_routing.py','ep_routing_ops.py']
assert set(p.name for p in old.glob('*.py'))==set(source_names), 'unexpected files: inspect before moving'
sources={name:(old/name).read_text() for name in source_names}
changes={}
def put(path,text):
    if path.suffix=='.py':ast.parse(text)
    changes[path]=text
def package(name,text):put(pkg/name,text)
def model(name,text):put(m/name,text)

split=sources['grouped_gemm_split.py']
kernel9=sources['grouped_gemm_api.py']
for name,text in [('grouped_gemm.py',split),('kernel9.py',kernel9)]:
    text=text[:text.index('class PackedExpertWeights:')]
    text=text.replace('from pathlib import Path','from ._native import resolve_library')
    text=text.replace('def load_library(path):','def load_library(path=None):')
    text=text.replace('path = Path(path).resolve()','path = resolve_library("grouped_gemm", path)') if name=='grouped_gemm.py' else text.replace('path=Path(path).resolve()','path=resolve_library("kernel9", path)')
    package(name,text)
weights='"""MiniMind expert-module weight caches; tensor GEMMs live in my_cuda_kernels."""\nimport torch\n\n'+split[split.index('class PackedExpertWeights:'):]
weights+='\n\n'+kernel9[kernel9.index('class PackedExpertWeights:'):].replace('class PackedExpertWeights:','class Kernel9PackedExpertWeights:',1)
model('model/backends/weights.py',weights)
package('native_bf16.py',sources['grouped_gemm_compat.py'])
ep=sources['minimind_ep_routing.py']
helpers=ep[ep.index('def combine_inference'):ep.index('class MiniMindEPRouting:')]
package('ep.py','"""Generic EP partition, capacity and native-precision merge helpers."""\nimport torch\nfrom .ep_ops import load_library\n\n'+helpers)
routing='"""MiniMind routing, NCCL orchestration and native expert execution."""\nimport torch\nimport torch.distributed as dist\nfrom my_cuda_kernels.ep import Workspace, combine_inference\n\n'
routing+=ep[ep.index('@torch.compiler.disable'):ep.index('def combine_inference')]
routing+=ep[ep.index('class MiniMindEPRouting:'):].replace('from ep_routing_ops import load_library','from my_cuda_kernels.ep_ops import load_library')
model('model/ep/routing.py',routing)
ops=sources['ep_routing_ops.py'].replace('import torch','import torch\nfrom ._native import resolve_library',1)
ops=ops.replace('_REGISTERED=False','_REGISTERED=False\n_LIBRARY=None',1)
ops=ops.replace('def load_library(path):\n    global _REGISTERED\n    torch.ops.load_library(str(path))\n    if _REGISTERED:return', 'def load_library(path=None):\n    global _REGISTERED, _LIBRARY\n    path=resolve_library("ep", path)\n    if _REGISTERED:\n        if path!=_LIBRARY:raise RuntimeError("another EP library is already loaded")\n        return\n    torch.ops.load_library(str(path))\n    _LIBRARY=path')
package('ep_ops.py',ops)
package('__init__.py','"""CUDA operators used through ordinary Python imports. Native libraries load explicitly."""\n__version__ = "0.1.0"\n')
package('_native.py',textwrap.dedent('''\
    """Resolve native builds without depending on an experiment directory."""
    import os
    from pathlib import Path

    LIBRARIES={"ep":"my_cuda_ep.so", "grouped_gemm":"my_cuda_grouped_split.so", "kernel9":"my_cuda_kernel9.so"}

    def repository_root():
        return Path(__file__).resolve().parents[3]

    def resolve_library(kind, path=None):
        root=Path(os.environ.get("MY_CUDA_KERNELS_BUILD", repository_root()/"build"))
        selected=Path(path).expanduser() if path is not None else root/kind/"current"/LIBRARIES[kind]
        selected=selected.resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"Native library missing: {selected}. Build it with: python script/build.py {kind}")
        return selected
    '''))
for source,target,replacements in [
    ('grouped_expert_mlp_split.py','model/backends/grouped_expert_mlp.py', [('from grouped_gemm_split import GemmWorkspace, PackedExpertWeights, family_for, grouped_gemm','from my_cuda_kernels.grouped_gemm import GemmWorkspace, family_for, grouped_gemm\nfrom .weights import PackedExpertWeights')]),
    ('grouped_expert_mlp.py','model/backends/kernel9_mlp.py',[('from grouped_gemm_api import grouped_gemm,GemmWorkspace,PackedExpertWeights','from my_cuda_kernels.kernel9 import grouped_gemm,GemmWorkspace\nfrom .weights import Kernel9PackedExpertWeights as PackedExpertWeights')]),
    ('minimind_ep_split_backend.py','model/backends/grouped_gemm.py',[('from grouped_expert_mlp_split import GroupedExpertMLP','from .grouped_expert_mlp import GroupedExpertMLP')]),
    ('minimind_ep_backend.py','model/backends/legacy_ep.py',[('from grouped_expert_mlp import GroupedExpertMLP','from .kernel9_mlp import GroupedExpertMLP')]),
    ('checkpoint_kn.py','model/checkpoint_kn.py',[])]:
    text=sources[source]
    for before,after in replacements:text=text.replace(before,after)
    model(target,text)
for name in ['model/ep/__init__.py','model/backends/__init__.py','tests/__init__.py','tests/ep/__init__.py','scripts/__init__.py','scripts/ep/__init__.py']:
    model(name,'')
model('model/ep_routing_backend.py',textwrap.dedent('''\
    """Attach MiniMind EP routing using the installed operator package."""
    def attach_ep_routing(model, *, library=None, mode="compact", decode_capacity_tokens=32):
        from .ep.routing import MiniMindEPRouting
        return MiniMindEPRouting(model, library, mode, decode_capacity_tokens=decode_capacity_tokens)
    '''))
model('model/grouped_gemm_backend.py',textwrap.dedent('''\
    """MiniMind adapters; original PyTorch models need no CUDA extension."""
    def load_kn_model(model, checkpoint):
        from .checkpoint_kn import load_kn_checkpoint
        return load_kn_checkpoint(model, checkpoint, ep_subset=bool(model.config.use_ep))

    def attach_grouped_backend(model, *, backend="native_bf16", library=None):
        if backend in ("cuda_split", "cuda_split_last_layer"):
            from my_cuda_kernels.grouped_gemm import load_library
            load_library(library)
        from .backends.grouped_gemm import MiniMindEPSplitBackend
        return MiniMindEPSplitBackend(model, backend)
    '''))
trainer=(m/'trainer/train_pretrain.py').read_text()
trainer=trainer.replace("    parser.add_argument('--ep_routing_python_dir', default=None, help='EP Python API directory')\n",'')
trainer=trainer.replace('                                                  python_dir=args.ep_routing_python_dir,\n','')
trainer=trainer.replace("    parser.add_argument(\"--save_dir\", type=str, default=\"../out\", help=\"模型保存目录\")","    parser.add_argument(\"--save_dir\", type=str, default=\"../out\", help=\"模型保存目录\")\n    parser.add_argument(\"--checkpoint_dir\", default=\"../checkpoints\", help=\"完整权重及每Rank恢复状态目录\")")
trainer=trainer.replace("save_dir='../checkpoints'",'save_dir=args.checkpoint_dir')
model('trainer/train_pretrain.py',trainer)
evaluation=(m/'eval_llm.py').read_text()
evaluation=evaluation.replace(', python_dir=args.grouped_gemm_python_dir','').replace(',\n                               python_dir=args.grouped_gemm_python_dir','')
evaluation='\n'.join(line for line in evaluation.split('\n') if 'parser.add_argument(\'--grouped_gemm_python_dir\'' not in line)
model('eval_llm.py',evaluation)
put(k/'pyproject.toml','''[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "my-cuda-kernels"
version = "0.1.0"
description = "CUDA operator practice and native-precision EP tensor operations"
requires-python = ">=3.10"
dependencies = ["torch>=2.9"]

[tool.setuptools.packages.find]
where = ["dev/pytorch"]
include = ["my_cuda_kernels*"]
''')
backup_k.mkdir(parents=True)
backup_m.mkdir(parents=True)
for path in changes:
    if path.exists():
        relative=path.relative_to(k) if path.is_relative_to(k) else path.relative_to(m)
        target=(backup_k if path.is_relative_to(k) else backup_m)/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
for name in source_names:
    target=backup_k/'dev/pytorch'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(old/name,target)
record=[]
for path,text in changes.items():
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)
    record.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(backup_k/'changes.json').write_text(json.dumps(record,indent=2)+'\n')
(backup_m/'changes.json').write_text(json.dumps(record,indent=2)+'\n')
print('CODE OWNERSHIP ORGANIZED',len(changes))
