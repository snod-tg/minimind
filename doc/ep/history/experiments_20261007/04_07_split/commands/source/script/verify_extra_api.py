"""Explicit Decode registration and automatic 256/257 dispatch boundary."""
import argparse,sys,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--library',required=True);p.add_argument('--output',required=True);a=p.parse_args()
sys.path.insert(0,str(a.source/'dev/pytorch'));sys.path.insert(0,str(a.source/'script'))
from grouped_gemm_split import load_library,grouped_gemm,GemmWorkspace
from kernel9_reference import reference,validate
load_library(a.library);torch.backends.cuda.matmul.allow_tf32=False
x=torch.randn(137,768,device='cuda',dtype=torch.bfloat16)*.1;b=torch.randn(2,768,2432,device='cuda',dtype=torch.bfloat16)*.1;c=torch.tensor([64,73],device='cuda');ws=GemmWorkspace.allocate(x,b)
rows=[]
for name,args in [('decode',(x,b,c)),('decode_out',(x,b,c,ws.out,ws.metadata,ws.partial))]:
 result=torch.library.opcheck(getattr(torch.ops.my_cuda_kernels_split,name),(args))
 assert all(v=='SUCCESS' for v in result.values()),result
 rows.append(dict(operator=name,opcheck=result))
fn=torch.compile(lambda x,b,c:grouped_gemm(x,b,c),backend='eager',dynamic=True)
for m in [256,257]:
 xx=torch.randn(m,768,device='cuda',dtype=torch.bfloat16)*.1;counts=[m//2,m-m//2];cc=torch.tensor(counts,device='cuda');validate(fn(xx,b,cc),reference(xx,b,counts));rows.append(dict(test='compile_auto_dispatch',m=m))
try:grouped_gemm(xx,b,cc,family='decode')
except (RuntimeError,ValueError):rows.append(dict(test='explicit_decode_257_rejected'))
else:raise AssertionError('Decode must reject M>256')
Path(a.output).write_text(json.dumps(rows,indent=2));print('EXTRA API CHECKS PASSED',flush=True)
