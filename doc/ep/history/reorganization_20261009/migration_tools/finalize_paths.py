from pathlib import Path
import ast
import json
import shutil

K=Path('/Users/tongleiwen/Code/my_cuda_kernels');M=Path('/Users/tongleiwen/Code/minimind')
B=M/'doc/ep/history/reorganization_20261009/final_paths'
assert not B.exists()
changes={}
def put(path,text):
    if path.suffix=='.py':ast.parse(text)
    changes[path]=text

benchmark=(K/'script/benchmark_kernel9.py').read_text()
benchmark=benchmark.replace("p.add_argument('--baselines',default='/data/user4/tongleiwen/moe-open-baselines')", "p.add_argument('--baselines',default=os.environ.get('MOE_OPEN_BASELINES'),help='Directory containing built CUTLASS baselines')")
benchmark=benchmark.replace("p.add_argument('--baseline',default='/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/results/R04_baseline')", "p.add_argument('--baseline',help='Optional recorded input directory for tensor replay')")
benchmark=benchmark.replace("p.add_argument('--checkpoint',default='/data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth')", "p.add_argument('--checkpoint',help='Optional checkpoint for tensor replay; pair with --baseline')")
benchmark=benchmark.replace('    args=p.parse_args()', '''    args=p.parse_args()
    if bool(args.baseline)!=bool(args.checkpoint):p.error('--baseline and --checkpoint must be supplied together')
    if not args.smoke and not args.validate_only and not args.baselines:p.error('Set --baselines or MOE_OPEN_BASELINES for CUTLASS timing')''')
benchmark=benchmark.replace('replays=[] if args.smoke else validate_replays','replays=[] if args.smoke or not args.baseline else validate_replays')
put(K/'script/benchmark_kernel9.py',benchmark)
put(K/'script/kernel9_reference.py','''"""Shared numerical/timing helpers from the canonical kernel9 benchmark."""
from .benchmark_kernel9 import (Ours, bind_ours, cpu_double_check, graph_make,
                               graph_time, reference, tensor_case, validate)
''')
p=K/'script/benchmark_kernel8.py'
put(p,p.read_text().replace("os.environ.get('MOE_OPEN_BASELINES','/data/user4/tongleiwen/moe-open-baselines')","os.environ['MOE_OPEN_BASELINES']"))
p=K/'script/run_kernel8.sh'
text=p.read_text().replace('PYTHON_BIN="${PYTHON_BIN:-/data/user4/tongleiwen/tilelang-3032/.venv/bin/python}"','PYTHON_BIN="${PYTHON_BIN:-python}"\n: "${MOE_OPEN_BASELINES:?Set MOE_OPEN_BASELINES to your built CUTLASS baseline directory}"')
put(p,text)
p=K/'script/run_tilelang_benchmarks.sh'
put(p,p.read_text().replace('# cd /data/user4/tongleiwen/tilelang-3032','# cd /path/to/tilelang'))
p=M/'scripts/ep/train.py'
text=p.read_text().replace('    if not Path(config["data_path"]).is_absolute() and not (ROOT / config["data_path"]).is_file():\n        raise FileNotFoundError(ROOT / config["data_path"])','    if not (ROOT / config["data_path"]).is_file():\n        raise FileNotFoundError(ROOT / config["data_path"])')
text=text.replace('model/tokenizer/tokenizer.json','model/tokenizer.json')
put(p,text)
p=M/'tests/ep/test_local_contracts.py'
text=p.read_text().replace('    def test_gpu_selection_and_original_training_config(self):','''    def test_tokenizer_and_trainer_argument_paths(self):
        self.assertTrue((ROOT/'model/tokenizer.json').is_file())
        source=(ROOT/'scripts/ep/train.py').read_text()
        self.assertIn('ROOT / "model/tokenizer.json"', source)
        trainer=(ROOT/'trainer/train_pretrain.py').read_text()
        import ast
        tree=ast.parse(trainer)
        flags={arg.value for node in ast.walk(tree) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument'
               for arg in node.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str)}
        config=load_config(DEFAULT_CONFIG)
        command=command_for(config,ROOT/'runs/test_only','test_only',None)
        for flag in command[7:]:
            if flag.startswith('--'):self.assertIn(flag,flags)

    def test_gpu_selection_and_original_training_config(self):''')
put(p,text)
for file in [K/'doc/commands/local_workflow.md',M/'doc/ep/commands/local_workflow.md']:
    text=file.read_text()
    if file.is_relative_to(K):
        text=text.replace('EP 的 --output 是 JSON 文件，先创建其父目录；Grouped GEMM 的 --output 是新目录。','EP 的 --output 是 JSON 文件，父目录自动创建；Grouped GEMM 的 --output 是新目录。')
    put(file,text)

# Obsolete AST-patched model adapters no longer occupy the active backend directory.
moves=[(M/'model/backends/legacy_ep.py',M/'doc/ep/history/reorganization_20261009/legacy_model/legacy_ep.py'),
       (M/'model/backends/kernel9_mlp.py',M/'doc/ep/history/reorganization_20261009/legacy_model/kernel9_mlp.py')]
for source,target in moves:assert source.exists() and not target.exists()
p=M/'model/backends/weights.py'
put(p,p.read_text()[:p.read_text().index('class Kernel9PackedExpertWeights:')].rstrip()+'\n')

B.mkdir(parents=True)
for path in changes:
    root=K if path.is_relative_to(K) else M
    target=B/'previous'/root.name/path.relative_to(root)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
for source,target in moves:
    target.parent.mkdir(parents=True,exist_ok=True);shutil.move(source,target)
for path,text in changes.items():path.write_text(text)
(B/'manifest.json').write_text(json.dumps(dict(changes=[str(p) for p in changes],moves=[dict(source=str(s),target=str(t)) for s,t in moves]),indent=2)+'\n')
print('CANONICAL PATHS CHECKED',len(changes),'files;',len(moves),'legacy adapters moved')
