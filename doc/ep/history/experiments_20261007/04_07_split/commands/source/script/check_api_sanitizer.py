"""Focused memory check for dispatcher-owned and caller-owned workspaces."""
import argparse,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--source',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.source/'script'));sys.path.insert(0,str(a.source/'dev/pytorch'))
from grouped_gemm_split import load_library,grouped_gemm,GemmWorkspace
from kernel9_reference import tensor_case,reference,validate
load_library(a.library);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
 for dtype,k,n,counts in [(torch.bfloat16,768,2432,[64,73]),(torch.bfloat16,768,2432,[1480,1432]),(torch.bfloat16,768,2432,[0,1]),(torch.bfloat16,2432,768,[1,15]),
                         (torch.bfloat16,2432,768,[31,33]),(torch.float16,1024,1024,[0,1,0,63]+[0]*12),
                         (torch.float16,1024,1024,[0]*16)]:
  x,b,_=tensor_case(counts,k,n,dtype);ct=torch.tensor(counts,device='cuda');ws=GemmWorkspace.allocate(x,b)
  validate(grouped_gemm(x,b,ct),reference(x,b,counts));validate(ws.run(x,b,ct),reference(x,b,counts))
  torch.cuda.synchronize();print('PASSED',str(dtype),k,n,counts,flush=True)
print('ALL API MEMORY CHECKS PASSED')
