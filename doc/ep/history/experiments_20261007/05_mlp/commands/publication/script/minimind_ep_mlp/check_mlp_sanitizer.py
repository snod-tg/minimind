"""Complete-chain memory/lifecycle checks with checkpoint weights and changing counts."""
import argparse,importlib.util,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True);p.add_argument('--library',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.source/'dev/pytorch'));sys.path.insert(0,str(a.source/'script'))
from grouped_gemm_api import load_library
from grouped_expert_mlp import GroupedExpertMLP
from kernel9_reference import reference,graph_make
load_library(a.library);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
spec=importlib.util.spec_from_file_location('mlp_memory_model',a.source/'script/model_minimind_snapshot.py')
model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
config=model.MiniMindConfig(hidden_size=768,moe_intermediate_size=2432,dropout=0.)
experts={i:model.FeedForward(config,intermediate_size=2432).cuda().eval() for i in [0,1]}
for i,e in experts.items():
    e.load_state_dict({f'{n}.weight':state[f'model.layers.7.mlp.experts.{i}.{n}.weight']
                       for n in ['gate_proj','up_proj','down_proj']})
del state
rows=[]
def rejected(fn):
    try:fn()
    except (RuntimeError,ValueError):return
    raise AssertionError('invalid state must raise')

with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
    allocating=GroupedExpertMLP(experts,[0,1],device='cuda:0')
    fast=GroupedExpertMLP(experts,[0,1],device='cuda:0',preallocate=True,cache=allocating.cache)
    rejected(lambda:GroupedExpertMLP({0:experts[1],1:experts[0]},[0,1],device='cuda:0',cache=allocating.cache))
    empty_grad=torch.empty((0,768),device='cuda',dtype=torch.bfloat16,requires_grad=True)
    rejected(lambda:allocating(empty_grad,torch.zeros(2,device='cuda',dtype=torch.int64)))
    def check(x,values):
        ct=torch.tensor(values,device=x.device,dtype=torch.int64)
        y=allocating(x,ct);fast.prepare(x);z=fast(x,ct)
        assert torch.equal(y,z)
        packed=allocating.cache.packed
        hidden=F.silu(reference(x,packed['gate_proj'],values))*reference(x,packed['up_proj'],values)
        oracle=reference(hidden,packed['down_proj'],values)
        torch.testing.assert_close(z,oracle,atol=.02,rtol=.02)
        assert bool(torch.isfinite(z).all())
        if len(x):assert z.data_ptr()==fast.workspaces[2].out.data_ptr()
        else:assert fast.workspaces is None and z.shape==x.shape
        torch.cuda.synchronize();rows.append(dict(counts=values,exact_between_api_modes=True,oracle_passed=True))
        print('PASSED',values,flush=True)
    torch.manual_seed(20261007)
    for values in [[0,0],[1,0],[1,1],[31,33],[0,129]]:
        x=(torch.randn((sum(values),768),device='cuda')*.1).to(torch.bfloat16)
        check(x,values)
    rejected(lambda:fast(x[:1],torch.tensor([1,0],device='cuda',dtype=torch.int64)))
    x=(torch.randn((65,768),device='cuda')*.1).to(torch.bfloat16)
    fast.prepare(x);ct=torch.tensor([0,65],device='cuda',dtype=torch.int64)
    graph=graph_make(lambda:fast(x,ct),1)
    for values in [[0,65],[65,0],[31,34],[1,64]]:
        ct.copy_(torch.tensor(values,device='cuda',dtype=torch.int64));graph.replay();torch.cuda.synchronize()
        packed=fast.cache.packed
        hidden=F.silu(reference(x,packed['gate_proj'],values))*reference(x,packed['up_proj'],values)
        torch.testing.assert_close(fast.workspaces[2].out,reference(hidden,packed['down_proj'],values),atol=.02,rtol=.02)
    del graph
    fast.invalidate('lifecycle test');assert fast.shape is None and fast.workspaces is None
    rejected(lambda:fast.prepare(x))
    rejected(lambda:allocating(x,ct))
    replacement=GroupedExpertMLP(experts,[0,1],device='cuda:0',preallocate=True);replacement.prepare(x)
    experts[0].train();rejected(lambda:replacement(x,ct));experts[0].eval()
    rejected(lambda:replacement(x,ct))
    torch.cuda.synchronize()
result=dict(passed=True,cases=rows,graph_distributions=4,shape_mismatch_rejected=True,
            shared_cache_invalidation_rejected=True,training_and_stale_cache_rejected=True,
            empty_requires_grad_rejected=True,wrong_cache_source_rejected=True)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print('ALL FULL MLP MEMORY/LIFECYCLE CHECKS PASSED',flush=True)
