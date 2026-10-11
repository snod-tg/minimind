"""Two-rank native-expert validation and paired end-to-end measurements."""
import argparse,importlib.util,json,os,sys,types,time,statistics
from pathlib import Path
import torch
import torch.distributed as dist
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--experts',type=int,default=4);p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--variants',default='original,compact,fixed,dedup');p.add_argument('--mode',choices=['validate','benchmark'],default='benchmark');p.add_argument('--routing-mode',default='compact');p.add_argument('--rounds',type=int,default=9);a=p.parse_args()
rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);torch.set_num_threads(1)
dist.init_process_group('nccl',device_id=device)
import model.model_minimind as module
config=module.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True,use_ep=True,ep_size=2,num_experts=a.experts,num_experts_per_tok=1,moe_intermediate_size=2432,dropout=0.,ep_comm_dtype='native',max_position_embeddings=4096)
model=module.MiniMindForCausalLM(config)
if a.checkpoint is not None:
    checkpoint=a.checkpoint;weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict({k:weights[k] for k in model.state_dict()},strict=True);del weights
model.to(device).eval()
if a.checkpoint is None:
    for name,param in model.named_parameters():
        if '.mlp.experts.' not in name:dist.broadcast(param.data,src=0)
from model.ep.routing import MiniMindEPRouting
controller=MiniMindEPRouting(model,a.library,a.routing_mode);rows=[];a.output.mkdir(parents=True,exist_ok=True)
def write():
    (a.output/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,mode=a.routing_mode,checks=rows,calls=controller.calls,fallbacks=controller.fallbacks,workspace_growth=sum(w.growth for w in controller.workspaces.values()),python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name()),indent=2)+'\n')
def check(y,ref,label,**info):
    exact=torch.equal(y,ref);bad=int((y!=ref).sum());maximum=float((y.float()-ref.float()).abs().max()) if y.numel() else 0
    row=dict(test=label,exact=exact,bad=bad,max_abs=maximum,**info);rows.append(row);write();print(row,flush=True)
    assert exact,row
workloads=[('decode1',1,1,128),('decode16',16,1,128),('prefill128',1,128,0),('prefill2720',8,340,0),('prefill10880',32,340,0)]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for topk in [1,2]:
        config.num_experts_per_tok=topk
        for name,b,l,context in workloads:
            torch.manual_seed(100+rank);ids=torch.randint(3,config.vocab_size,(b,context or l),device=device)
            controller.fixed_decode=bool(context);controller.enabled=False;kv=None
            if context:
                out=model(ids,use_cache=True,logits_to_keep=1);kv=out.past_key_values;ids=out.logits[:,-1].argmax(-1,keepdim=True)
            fn=lambda:model(ids,past_key_values=kv,use_cache=True,logits_to_keep=1)
            if a.mode=='validate':
                controller.enabled=False;expected=fn();ref_logits=expected.logits.clone();ref_hidden=expected.hidden_states.clone()
                controller.enabled=True;actual=fn();check(actual.logits,ref_logits,'model_logits',case=name,topk=topk);check(actual.hidden_states,ref_hidden,'model_hidden',case=name,topk=topk)
                del expected,actual,ref_logits,ref_hidden
            else:
                variants=a.variants.split(',');samples={name:[] for name in variants}
                def choose(name):
                    controller.enabled=name!='original';controller.mode='compact' if name=='original' else name
                for variant in variants:
                    choose(variant)
                    for _ in range(5):fn()
                for r in range(a.rounds):
                    order=variants[r%len(variants):]+variants[:r%len(variants)]
                    for variant in order:
                        choose(variant);dist.barrier();torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize()
                        samples[variant].append((time.perf_counter()-start)*1e6)
                rows.append(dict(test='model_timing',case=name,topk=topk,medians={k:statistics.median(v) for k,v in samples.items()},samples=samples));write();print(name,topk,rows[-1]['medians'],flush=True)
            del ids,kv;torch.cuda.empty_cache();dist.barrier()
    if a.mode=='validate':
        owner=model.model.layers[-1].mlp;gate=owner.gate.forward
        patterns=[('all0',1,17,23),('all2',1,17,23),('pair01',2,17,23),('pair12',2,31,33),('skew',1,65,65),('random',2,257,513),('empty_source',1,0,17),('both_empty',2,0,0)]
        for pattern,topk,left,right in patterns:
            controller.fixed_decode=pattern in ('all0','all2','pair01','empty_source','both_empty');config.num_experts_per_tok=topk;n=left if rank==0 else right;torch.manual_seed(22+rank);x=torch.randn(1,n,768,device=device,dtype=torch.bfloat16)*.1
            def forced(mod,value):
                logits=torch.full((len(value),a.experts),-4.,device=device,dtype=torch.bfloat16)
                if pattern=='all2':logits[:,2]=8
                elif pattern=='pair01':logits[:,0]=8;logits[:,1]=7
                elif pattern=='pair12':
                    first=a.experts//2-1;logits[:,first]=8;logits[:,first+1]=7
                elif pattern=='random':logits=torch.randn_like(logits)
                else:
                    logits[:,0]=8
                    if pattern=='skew' and len(value):logits[-1,0]=-4;logits[-1,3]=8
                return logits
            # Random route must be identical for both invocations.
            owner.gate.forward=types.MethodType(forced,owner)
            torch.manual_seed(41+rank);controller.enabled=False;ref=owner(x)
            torch.manual_seed(41+rank);controller.enabled=True;y=owner(x)
            check(y,ref,'boundary',pattern=pattern,topk=topk,tokens=n)
        owner.gate.forward=gate
        # Same workspace receives different assignments on successive calls.
        controller.fixed_decode=False;config.num_experts_per_tok=2
        for step in range(6):
            torch.manual_seed(88+step+rank);x=torch.randn(1,33,768,device=device,dtype=torch.bfloat16)*.1
            controller.enabled=False;ref=owner(x);controller.enabled=True;y=owner(x);check(y,ref,'dynamic_routing',step=step)
if a.mode=='validate':
    # No inference scratch may enter autograd. Original train path remains exact.
    config.num_experts_per_tok=2;model.train();owner=model.model.layers[-1].mlp
    torch.manual_seed(55+rank);initial=torch.randn(1,17,768,device=device,dtype=torch.bfloat16)
    reference=None
    for enabled in [False,True]:
        controller.enabled=enabled;model.zero_grad(set_to_none=True);x=initial.clone().requires_grad_(True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y=owner(x);loss=y.float().square().mean()+owner.aux_loss
        loss.backward();data=dict(input=x.grad.clone(),output=y.detach().clone(),**{n:p.grad.clone() for n,p in owner.named_parameters() if p.grad is not None})
        if reference is None:reference=data
        else:
            for key,value in data.items():check(value,reference[key],'training_fused',tensor=key)
    model.eval()
write();dist.barrier();dist.destroy_process_group();print('EP PASSED',rank,a.mode,flush=True)
