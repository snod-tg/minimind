"""Test the numerical precondition for naive communication/MLP chunk overlap."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
w=torch.load('/data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth',map_location='cpu',weights_only=True);rows=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for layer in [0,7]:
        prefix=f'model.layers.{layer}.mlp.experts.0.'
        wg=w[prefix+'gate_proj.weight'].float().cuda();wu=w[prefix+'up_proj.weight'].float().cuda();wd=w[prefix+'down_proj.weight'].float().cuda()
        def fn(x):return F.linear(F.silu(F.linear(x,wg))*F.linear(x,wu),wd)
        for n in [137,2720,10880]:
            torch.manual_seed(44+n);x=(torch.randn(n,768,device='cuda')*.1).bfloat16();ref=fn(x)
            y=torch.cat([fn(v) for v in x.split((n+1)//2)])
            delta=(y.float()-ref.float());rows.append(dict(layer=layer,n=n,exact=torch.equal(y,ref),bad=int((y!=ref).sum()),max_abs=float(delta.abs().max()),relative_rms=float(delta.square().sum().sqrt()/ref.float().square().sum().sqrt())))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2)+'\n');print(rows)
