"""Actual two-rank 6-of-2 optimized-EP training, compile and optimizer smoke test."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
p=argparse.ArgumentParser();p.add_argument('--stage8',type=Path,required=True);p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=4);p.add_argument('--compile',type=int,default=1);p.add_argument('--batch',type=int,default=2);p.add_argument('--length',type=int,default=33);a=p.parse_args()
rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);torch.set_num_threads(1);dist.init_process_group('nccl',device_id=device)
sys.path.insert(0,str(a.stage8/'source/dev/pytorch'));sys.path.insert(0,'/data/user4/tongleiwen/minimind')
spec=importlib.util.spec_from_file_location('test_compile_model',a.stage8/'minimind/model/model_minimind.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
from trainer.trainer_utils import sync_ep_shared_parameters,sync_ep_gradients,clip_ep_grad_norm_
from minimind_ep_routing import MiniMindEPRouting
torch.manual_seed(42+rank);c=m.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True,use_ep=True,ep_size=2,num_experts=6,num_experts_per_tok=2,moe_intermediate_size=2432)
model=m.MiniMindForCausalLM(c).to(device).train();sync_ep_shared_parameters(model)
controller=MiniMindEPRouting(model,a.library,'compact');optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4)
keys=list(model.state_dict());shapes={k:list(v.shape) for k,v in model.state_dict().items()}
if a.compile:
    # Dynamic shapes prevent recompiling GEMMs for each expert's token count.
    for layer in model.model.layers:
        for expert in layer.mlp.experts.values():expert.compile(dynamic=True,fullgraph=True)
    model.compile(dynamic=True,fullgraph=False)
rows=[];a.output.mkdir(parents=True,exist_ok=True)
for step in range(a.steps):
    torch.manual_seed(80+step+rank);x=torch.randint(3,6400,(a.batch,a.length),device=device);labels=torch.randint(3,6400,x.shape,device=device)
    start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=model(x,labels=labels);loss=result.loss+result.aux_loss
    assert torch.isfinite(loss)
    loss.backward();sync_ep_gradients(model);gn=clip_ep_grad_norm_(model,1.)
    assert torch.isfinite(gn)
    optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
    assert list(model.state_dict())==keys and all(list(v.shape)==shapes[k] for k,v in model.state_dict().items())
    rows.append(dict(step=step,loss=float(loss),grad_norm=float(gn),seconds=time.perf_counter()-start,peak_memory=torch.cuda.max_memory_allocated(),routing_calls=controller.calls,routing_fallbacks=controller.fallbacks))
    assert controller.fallbacks==0,rows[-1]
    (a.output/f'rank{rank}.json').write_text(json.dumps(dict(rows=rows,counters={group:dict(counts) for group,counts in torch._dynamo.utils.counters.items()},checkpoint_keys_unchanged=True,experts=6,topk=2),indent=2,default=str)+'\n');print(rows[-1],flush=True)
# Shared Router parameters must remain synchronized after actual optimizer steps.
for layer in model.model.layers:
    w=layer.mlp.gate.weight.detach();other=w.clone();dist.broadcast(other,src=0);assert torch.equal(w,other)
model.eval();dist.barrier();dist.destroy_process_group();print('COMPILE TRAINING PASSED',rank,flush=True)
