"""Build against installed PyTorch using GCC/nvcc, without Ninja or pip installs."""
import argparse,json,os,hashlib,subprocess,shlex
from pathlib import Path
from torch.utils.cpp_extension import include_paths
import torch
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);out=a.output.resolve();src=a.source.resolve()/'dev/cuda'
import shutil
for subdir in ('dev/cuda', 'dev/pytorch'):
    shutil.copytree(a.source/subdir, out/'source'/subdir, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
cxx=os.environ['CXX'];cuda=Path(os.environ.get('CUDA_HOME','/usr/local/cuda'));torchlib=Path(torch.__file__).parent/'lib'
abi=int(torch._C._GLIBCXX_USE_CXX11_ABI)
commands=[
 [cxx,'-O3','-std=c++17','-fPIC',f'-D_GLIBCXX_USE_CXX11_ABI={abi}',*[f'-I{d}' for d in include_paths()],f'-I{cuda}/include','-c',str(src/'ep_routing_torch.cpp'),'-o',str(out/'binding.o')],
 [str(cuda/'bin/nvcc'),'-ccbin',cxx,'-O3','-std=c++17','-arch=sm_89','--fmad=false','--cudart','shared','-Xcompiler','-fPIC','-c',str(src/'ep_routing.cu'),'-o',str(out/'kernel.o')],
 [cxx,'-shared',str(out/'binding.o'),str(out/'kernel.o'),f'-L{torchlib}',f'-L{cuda}/lib64',f'-Wl,-rpath,{torchlib}',f'-Wl,-rpath,{cuda}/lib64','-ltorch','-ltorch_cpu','-ltorch_cuda','-lc10','-lc10_cuda','-lcudart','-lcublas','-o',str(out/'my_cuda_ep.so')]
]
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
(out/'build.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+'\n'.join(shlex.join(c) for c in commands)+'\n')
for i,cmd in enumerate(commands):
 print(shlex.join(cmd),flush=True)
 with (out/f'compile_{i}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
lib=out/'my_cuda_ep.so';torch.ops.load_library(str(lib))
record=dict(library=str(lib),sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),torch=torch.__version__,torch_cuda=torch.version.cuda,
            python=os.sys.version,abi=bool(abi),cxx=cxx,arch='sm_89',builder='direct GCC/nvcc',ninja_required=False,source_sha256={str(p.relative_to(a.source.resolve())):hashlib.sha256(p.read_bytes()).hexdigest() for subdir in ('dev/cuda','dev/pytorch') for p in (a.source.resolve()/subdir).glob('*') if p.is_file()})
(out/'build.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));print('BUILD PASSED')
