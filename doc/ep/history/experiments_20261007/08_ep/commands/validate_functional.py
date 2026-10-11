import argparse,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(Path('source/dev/pytorch').resolve()))
from ep_routing_ops import load_library
from minimind_ep_routing import Workspace
load_library(a.library);ws=Workspace(reuse=False);rows=[]
for dtype in [torch.float32,torch.bfloat16,torch.float16]:
    n,h=33,768;ids=torch.randint(6,(n*2,),device='cuda')
    with torch.no_grad():perm,inv,counts=ws.sort(ids,3,'send')
    for op in ['pack','gather','combine']:
        torch.manual_seed(53)
        if op=='pack':
            initial=torch.randn(n,h,device='cuda',dtype=dtype)
            def run(x,w,custom):return torch.ops.my_cuda_ep.pack(x,ids,perm,inv,2,3)[0] if custom else x[perm//2]
        elif op=='gather':
            initial=torch.randn(n*2,h,device='cuda',dtype=dtype)
            def run(x,w,custom):return torch.ops.my_cuda_ep.gather(x,perm) if custom else x[perm]
        else:
            initial=torch.randn(n*2,h,device='cuda',dtype=dtype)
            def run(x,w,custom):
                if custom:return torch.ops.my_cuda_ep.combine(x,w,inv,2,(torch.float32,torch.float16,torch.bfloat16).index(dtype))
                return torch.zeros(n,h,device='cuda',dtype=dtype).index_add_(0,torch.arange(n,device='cuda').repeat_interleave(2),(x[inv]*w[:,None]).to(dtype))
        initialw=torch.rand(n*2,device='cuda',dtype=dtype);reference=None
        for custom in [False,True]:
            x=initial.clone().requires_grad_();w=initialw.clone().requires_grad_();y=run(x,w,custom);(y.float().square().sum()).backward()
            values=[y.detach(),x.grad]+([w.grad] if op=='combine' else [])
            if reference is None:reference=values
            else:
                assert all(torch.equal(v,r) for v,r in zip(values,reference)),(dtype,op)
                rows.append(dict(op=op,dtype=str(dtype),forward_and_grad_exact=True))
    x=torch.randn(n,h,device='cuda',dtype=dtype,requires_grad=True)
    checks=torch.library.opcheck(torch.ops.my_cuda_ep.pack.default,(x,ids,perm,inv,2,3))
    rows.append(dict(op='pack_opcheck',dtype=str(dtype),checks=checks))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2)+'\n');print('FUNCTIONAL PASSED',rows,flush=True)
