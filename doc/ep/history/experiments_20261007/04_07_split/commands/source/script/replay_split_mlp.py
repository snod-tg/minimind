"""Real expert MLP replay: native BF16 contract and separate CUDA FP32 oracle."""
import argparse,copy,importlib.util,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--baseline',required=True);p.add_argument('--extra',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'dev/pytorch'))
from grouped_gemm_split import load_library
from grouped_expert_mlp_split import GroupedExpertMLP
from kernel9_reference import reference,validate,cpu_double_check,graph_make
from checkpoint_kn import LinearKN
load_library(a.library);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
torch.cuda.set_stream(torch.cuda.Stream())
spec=importlib.util.spec_from_file_location('replay_model',root.parent/'minimind/model/model_minimind.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
config=module.MiniMindConfig(hidden_size=768,moe_intermediate_size=2432,dropout=0.)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);contexts={};rows=[];extra=[]

def ctx(layer,ids):
 key=(layer,tuple(ids))
 if key not in contexts:
  with torch.inference_mode(False):
   experts={i:module.FeedForward(config,2432).cuda().eval() for i in ids}
   for i,e in experts.items():e.load_state_dict({f'{name}.weight':state[f'model.layers.{layer}.mlp.experts.{i}.{name}.weight'].float() for name in ('gate_proj','up_proj','down_proj')})
  contexts[key]=(experts,GroupedExpertMLP(experts,ids))
 return contexts[key]

def native(x,counts,experts,ids):
 parts=[];off=0
 for i,m in zip(ids,counts):
  if m:parts.append(experts[i](x[off:off+m]))
  off+=m
 return torch.cat(parts) if parts else x.clone()

def stats(x,y):
 d=(x.float()-y.float()).abs()
 if not d.numel():return dict(bad=0,relative_rms=0.,exact=True)
 return dict(bad=int((d>.02+.02*y.float().abs()).sum()),relative_rms=float(d.square().mean().sqrt()/y.float().square().mean().sqrt().clamp_min(1e-30)),exact=torch.equal(x,y))

with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for fixtures in (a.baseline,a.extra):
  for path in sorted(Path(fixtures).glob('*/capture_rank*.pt')):
   data=torch.load(path,map_location='cpu',weights_only=True)
   for r in data['replays']:
    x=r['x'].cuda();counts=r['counts'];ids=r['local_expert_ids'];experts,mlp=ctx(r['layer'],ids)
    original=native(x,counts,experts,ids);assert torch.equal(original,r['y'].cuda())
    gpu=torch.tensor(counts,device='cuda',dtype=torch.int64);actual=mlp(x,gpu)
    with torch.autocast('cuda',enabled=False):
     packed=mlp.cache.packed;gate=reference(x,packed['gate_proj'],counts);up=reference(x,packed['up_proj'],counts)
     middle=F.silu(gate)*up;oracle=reference(middle,packed['down_proj'],counts)
     check=stats(actual,oracle);assert check['bad']==0 and check['relative_rms']<=.005,check
     if len(x):
      _,g,u,d,act,h=mlp.prepare(x)
      for name,operand,value in [('gate_proj',x,g.out[:len(x)]),('up_proj',x,u.out[:len(x)]),('down_proj',h[:len(x)],d.out[:len(x)])]:
       validate(value,reference(operand,packed[name],counts));cpu_double_check(operand,packed[name],value,counts,17)
    rows.append(dict(case=path.parent.name,rank=r['rank'],layer=r['layer'],m=len(x),counts=counts,original_exact=True,fp32_chain=check,native=stats(actual,original)))
    mlp.release_workspaces();(out/'replay.json').write_text(json.dumps(rows,indent=2));print('REPLAY',len(rows),path.parent.name,r['layer'],flush=True)
 assert len(rows)==100,len(rows)
 experts,mlp=ctx(7,[0,1]);x=torch.randn((137,768),device='cuda',dtype=torch.bfloat16)*.1;gpu=torch.tensor([64,73],device='cuda');mlp.prepare(x)
 for family in ('prefill','decode'):
  g=graph_make(lambda:mlp(x,gpu,family=family),1)
  for counts in ([64,73],[65,72],[0,137],[137,0]):
   gpu.copy_(torch.tensor(counts,device='cuda'));g.replay();torch.cuda.synchronize()
   state_ws=mlp.prepare(x);actual=state_ws[3].out[:137]
   with torch.autocast('cuda',enabled=False):
    oracle=reference(F.silu(reference(x,mlp.cache.packed['gate_proj'],counts))*reference(x,mlp.cache.packed['up_proj'],counts),mlp.cache.packed['down_proj'],counts)
   check=stats(actual,oracle);assert check['bad']==0 and check['relative_rms']<=.005
   extra.append(dict(test='mlp_graph_dynamic_counts',family=family,counts=counts))
  del g
 mlp.release_workspaces()
 for m in (1,16,137,256,257,511):
  x=torch.randn((m,768),device='cuda',dtype=torch.bfloat16)*.1;counts=[m//2,m-m//2];gpu=torch.tensor(counts,device='cuda');mlp(x,gpu)
  extra.append(dict(test='capacity_reuse',m=m,prepares=mlp.prepare_events,capacity=mlp.prepare(x)[0]))
 assert [r['capacity'] for r in extra if r['test']=='capacity_reuse']==[256]*4+[512]*2
 # Canonical KN expert parameters pack directly, and original NK modules remain exact.
 with torch.inference_mode(False),torch.no_grad():
  kn=copy.deepcopy(experts)
  for e in kn.values():
   for name in ('gate_proj','up_proj','down_proj'):
    layer=getattr(e,name);setattr(e,name,LinearKN(layer,torch.nn.Parameter(layer.weight.T.contiguous())))
 kn_mlp=GroupedExpertMLP(kn,[0,1]);x=x[:137];counts=[64,73];gpu=torch.tensor(counts,device='cuda')
 assert torch.equal(native(x,counts,kn,[0,1]),native(x,counts,experts,[0,1]))
 assert torch.equal(kn_mlp(x,gpu),mlp(x,gpu));extra.append(dict(test='kn_direct_pack',native_bitwise=True,cuda_bitwise=True))
 (out/'extra.json').write_text(json.dumps(extra,indent=2))
 summary=dict(replays=len(rows),fp32_chain_bad=sum(r['fp32_chain']['bad'] for r in rows),native_failed=sum(r['native']['bad']>0 or r['native']['relative_rms']>.005 for r in rows),extra=len(extra),default_bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print('ALL MLP CHECKS PASSED',summary,flush=True)
