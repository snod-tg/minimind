from pathlib import Path
import ast
import json
import shutil

K = Path('/Users/tongleiwen/Code/my_cuda_kernels')
M = Path('/Users/tongleiwen/Code/minimind')
W = Path('/Users/tongleiwen/Documents/resume/organization_work')
STAGE = W.parent/'minimind_ep_grouped_gemm'
BK = K/'doc/history/reorganization_20261009/entries'
BM = M/'doc/ep/history/reorganization_20261009/entries'
assert not BK.exists() and not BM.exists()
writes = {}
moves = []
def put(path, content):
    if path.suffix == '.py': ast.parse(content)
    writes[path] = content
def move(source, target):
    assert source.exists() and not target.exists(), (source,target)
    moves.append((source,target))

put(K/'script/build.py',(W/'build.py').read_text())
put(K/'script/__init__.py','')
put(K/'tests/__init__.py','')
put(K/'tests/ep/__init__.py','')
put(M/'scripts/ep/train.py',(W/'train.py').read_text())
put(M/'configs/ep/pretrain_6_top2.json',(W/'pretrain_6_top2.json').read_text())

for filename in ['validate_ops.py','validate_functional.py','validate_configuration.py']:
    source=K/'script/ep_routing'/filename
    text=source.read_text()
    text='\n'.join(line for line in text.split('\n') if not line.startswith('sys.path.insert'))
    text=text.replace("p.add_argument('--source',type=Path,default=Path('source'));",'')
    text=text.replace('from minimind_ep_routing import Workspace, combine_inference','from my_cuda_kernels.ep import Workspace, combine_inference')
    text=text.replace('from minimind_ep_routing import Workspace','from my_cuda_kernels.ep import Workspace')
    text=text.replace('from ep_routing_ops import load_library','from my_cuda_kernels.ep_ops import load_library')
    if filename=='validate_ops.py':
        text=text.replace('torch.ops.load_library(a.library);','from my_cuda_kernels.ep_ops import load_library\nload_library(a.library);')
    put(K/'tests/ep'/filename,text)

for filename,target in [('validate_ep.py','tests/ep/validate_model.py'),
                        ('validate_distributed_configuration.py','tests/ep/validate_configurations.py'),
                        ('benchmark_ep_variants.py','scripts/ep/benchmark.py')]:
    source=K/'script/ep_routing'/filename
    text=source.read_text()
    text='\n'.join(line for line in text.split('\n') if not line.startswith('sys.path.insert'))
    if 'spec=importlib.util.spec_from_file_location' in text:
        text='\n'.join('import model.model_minimind as module' if line.startswith('spec=importlib.util.spec_from_file_location') else line for line in text.split('\n'))
        text=text.replace("choices=[4,6],",'')
        text=text.replace("p.add_argument('--experts'","p.add_argument('--checkpoint',type=Path);p.add_argument('--experts'",1)
        text=text.replace('if a.experts==4:', 'if a.checkpoint is not None:',1)
        text=text.replace("checkpoint=Path('/data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth')",'checkpoint=a.checkpoint')
        text=text.replace('if a.experts==6:', 'if a.checkpoint is None:',1)
        text=text.replace('os.sched_setaffinity(0,{int(os.environ.get("BENCH_CPU_BASE",28))+rank});', '')
    else:
        start=text.index("spec = importlib.util.spec_from_file_location")
        end=text.index('spec.loader.exec_module(module)',start)+len('spec.loader.exec_module(module)')
        text=text[:start]+'import model.model_minimind as module'+text[end:]
    text=text.replace('from minimind_ep_routing import MiniMindEPRouting','from model.ep.routing import MiniMindEPRouting')
    put(M/target,text)

driver=(K/'script/ep_routing/run_measured_variants.py').read_text()
driver=driver.replace('import subprocess','import subprocess\nimport sys\nfrom .train import ROOT, validate_gpus')
driver=driver.replace("p.add_argument('--gpus', default='1,3')","p.add_argument('--gpus', required=True)\np.add_argument('--experts', type=int, default=4)\np.add_argument('--checkpoint', type=Path)")
driver=driver.replace('a.output.mkdir(parents=True, exist_ok=False)','validate_gpus(a.gpus)\na.output=a.output.resolve()\na.output.mkdir(parents=True, exist_ok=False)')
driver=driver.replace("'/data/user4/tongleiwen/tilelang-3032/.venv/bin/python'",'sys.executable')
driver=driver.replace("    'benchmark_ep_variants.py',", "    '--module', 'scripts.ep.benchmark',")
driver=driver.replace("str(a.rounds)]","str(a.rounds), '--experts', str(a.experts)]\nif a.checkpoint is not None: command += ['--checkpoint', str(a.checkpoint.resolve())]")
driver=driver.replace('subprocess.Popen(command, env=env','subprocess.Popen(command, cwd=ROOT, env=env')
driver=driver.replace("own='benchmark_ep_variants.py' in cmd","own='scripts.ep.benchmark' in cmd")
driver=driver.replace(", BENCH_CPU_BASE='28'",'')
driver=driver.replace("a.library, '--output'","str(Path(a.library).resolve()), '--output'")
put(M/'scripts/ep/run_benchmark.py',driver)
chunk=(K/'script/ep_routing/check_chunking.py').read_text()
chunk=chunk.replace("p.add_argument('--output'","p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output'",1)
chunk=chunk.replace("'/data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth'",'a.checkpoint')
put(M/'scripts/ep/check_chunking.py',chunk)
put(M/'scripts/ep/plot.py',(K/'script/ep_routing/plot_ep.py').read_text())

for filename in ['benchmark_grouped_gemm_api.py','check_grouped_gemm_api_sanitizer.py']:
    text=(K/'script'/filename).read_text()
    text=text.replace("sys.path.insert(0,str(a.source/'script'));sys.path.insert(0,str(a.source/'dev/pytorch'))",'')
    text=text.replace(";p.add_argument('--source',type=Path,required=True)",'')
    text=text.replace('from grouped_gemm_api import','from my_cuda_kernels.kernel9 import')
    text=text.replace('from kernel9_reference import','from script.kernel9_reference import')
    put(K/'script'/filename,text)

for source,target,mod in [
    (K/'script/verify_grouped_gemm_api.py',K/'tests/verify_kernel9.py','my_cuda_kernels.kernel9'),
    (STAGE/'commands/04_07_split/source/script/verify_api.py',K/'tests/verify_grouped_gemm.py','my_cuda_kernels.grouped_gemm')]:
    text=source.read_text()
    text=text.replace("sys.path.insert(0,str(a.source/'script'));sys.path.insert(0,str(a.source/'dev/pytorch'))",'')
    text=text.replace(";p.add_argument('--source',type=Path,required=True)",'')
    text=text.replace('from grouped_gemm_api import',f'from {mod} import').replace('from grouped_gemm_split import',f'from {mod} import')
    text=text.replace(',PackedExpertWeights','').replace('from kernel9_reference import','from script.kernel9_reference import')
    start=text.index(' # Explicit weight cache lifecycle')
    end=text.index(' # Async forward:',start)
    text=text[:start]+text[end:]
    put(target,text)
text=(STAGE/'commands/04_07_split/source/script/benchmark_api.py').read_text()
text=text.replace("sys.path.insert(0,str(a.source/'script'));sys.path.insert(0,str(a.source/'dev/pytorch'))",'')
text=text.replace(";p.add_argument('--source',type=Path,required=True)",'')
text=text.replace('from grouped_gemm_split import','from my_cuda_kernels.grouped_gemm import')
text=text.replace('from kernel9_reference import','from script.kernel9_reference import')
put(K/'script/benchmark_grouped_split.py',text)

# Named obsolete entry points are retained with their original contents for recovery.
for name in ['ep_routing','split_grouped_gemm','grouped_split','grouped_decode']:
    move(K/'script'/name,BK/'legacy_scripts'/name)
for name in ['build_grouped_gemm_api.py','verify_grouped_gemm_api.py']:
    move(K/'script'/name,BK/'legacy_scripts'/name)
for name in ['minimind_ep','minimind_ep_mlp','checkpoint_kn','measure_baseline.py','model_minimind_snapshot.py']:
    move(K/'script'/name,BM/'legacy_scripts'/name)

for root in [K,M]:
    ignore=(root/'.gitignore').read_text()
    additions=['/build/','*.egg-info/'] if root==K else ['/runs/']
    put(root/'.gitignore',ignore.rstrip()+'\n'+''.join(line+'\n' for line in additions if line not in ignore.splitlines()))

# Preflight all target files and recovery paths before the first mutation.
for path in writes:
    if path.exists():
        backup=(BK if path.is_relative_to(K) else BM)/'previous'/path.relative_to(K if path.is_relative_to(K) else M)
        assert not backup.exists()
BK.mkdir(parents=True);BM.mkdir(parents=True)
for path in writes:
    if path.exists():
        backup=(BK if path.is_relative_to(K) else BM)/'previous'/path.relative_to(K if path.is_relative_to(K) else M)
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup)
for source,target in moves:
    target.parent.mkdir(parents=True,exist_ok=True);shutil.move(source,target)
for path,text in writes.items():
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
record=dict(writes=[str(p) for p in writes],moves=[dict(source=str(s),target=str(t)) for s,t in moves])
(BK/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
(BM/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print('ENTRY POINTS ORGANIZED',len(writes),'writes;',len(moves),'recoverable moves')
