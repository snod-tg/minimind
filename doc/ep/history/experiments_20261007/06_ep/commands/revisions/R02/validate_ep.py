"""Two-rank isolated EP, model propagation, dynamic-routing boundaries and training fallback."""
import argparse,copy,json,sys,types
from pathlib import Path
import torch
import torch.distributed as dist
from ep_common import setup,inputs,metrics,route_metrics,write,WORKLOADS

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--library',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--repo',type=Path,required=True)
p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--workloads',default=','.join(WORKLOADS));p.add_argument('--topks',default='1,2')
a=p.parse_args();rank,device,module,model,config=setup(a.source,a.checkpoint,a.output)
from grouped_gemm_api import load_library
from minimind_ep_backend import MiniMindEPBackend
load_library(a.library);controller=MiniMindEPBackend(model);controller.initialize()
full=[];isolated=[];boundaries=[];generations=[]

def captured(fn):
    controller.reset_capture();controller.capture=True
    ep_inputs={};ep_outputs={};blocks={};hooks=[]
    for i,layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_pre_hook(lambda m,v,i=i:ep_inputs.__setitem__(i,v[0].detach().cpu())))
        hooks.append(layer.mlp.register_forward_hook(lambda m,v,y,i=i:ep_outputs.__setitem__(i,y.detach().cpu())))
        hooks.append(layer.register_forward_hook(lambda m,v,y,i=i:blocks.__setitem__(i,y[0].detach().cpu())))
    result=fn()
    for hook in hooks:hook.remove()
    data=dict(logits=result.logits.detach().cpu(),hidden=result.hidden_states.detach().cpu(),
              ep_inputs=ep_inputs,ep_outputs=ep_outputs,blocks=blocks,
              local=controller.rows,routes=controller.routes,paths=controller.calls,
              last_tokens=result.logits[:,-1].argmax(-1).cpu())
    controller.capture=False
    return data

with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for topk in map(int,a.topks.split(',')):
        config.num_experts_per_tok=topk
        for name in a.workloads.split(','):
            case=f'{name}_top{topk}';controller.backend='original'
            x,cache,info=inputs(a.repo,rank,name,topk,model,a.baseline)
            cache_before=[(k.clone(),v.clone()) for k,v in cache] if cache else None
            fn=lambda:model(x,past_key_values=cache,use_cache=True)
            plain=fn();expected_logits=plain.logits.cpu();expected_hidden=plain.hidden_states.cpu();del plain
            reference=captured(fn)
            assert torch.equal(reference['logits'],expected_logits) and torch.equal(reference['hidden'],expected_hidden)
            saved=torch.load(a.baseline/case/f'capture_rank{rank}.pt',map_location='cpu',weights_only=True)
            assert all(reference['local'][r['layer']]['counts']==r['counts'] for r in saved['counts'])
            for backend in ['kernel9','hybrid']:
                controller.backend=backend;candidate=captured(fn)
                layer_rows=[]
                for i in range(8):
                    layer_rows.append(dict(layer=i,path=candidate['paths'][i],m=candidate['local'][i]['m'],
                                           counts=candidate['local'][i]['counts'],baseline_counts=reference['local'][i]['counts'],
                                           routing=route_metrics(candidate['routes'][i],reference['routes'][i]),
                                           ep=metrics(candidate['ep_outputs'][i],reference['ep_outputs'][i]),
                                           block=metrics(candidate['blocks'][i],reference['blocks'][i])))
                row=dict(case=case,rank=rank,backend=backend,layers=layer_rows,
                         logits=metrics(candidate['logits'],reference['logits']),hidden=metrics(candidate['hidden'],reference['hidden']),
                         next_tokens_changed=int((candidate['last_tokens']!=reference['last_tokens']).sum()),batch=len(x))
                full.append(row);write(a.output/f'full_model_rank{rank}.json',full)
                print('MODEL',case,rank,backend,row['logits'],row['next_tokens_changed'],flush=True)
                del candidate
            # Same EP input on both ranks isolates local expert math from prior-layer propagation.
            for i,layer in enumerate(model.model.layers):
                ep_x=reference['ep_inputs'][i].to(device);controller.backend='original';controller.reset_capture();controller.capture=True
                expected=layer.mlp(ep_x);assert torch.equal(expected.cpu(),reference['ep_outputs'][i])
                routes=controller.routes[i];counts=controller.rows[i]['counts']
                for backend in ['kernel9','hybrid']:
                    controller.backend=backend;controller.reset_capture();actual=layer.mlp(ep_x)
                    assert controller.rows[i]['counts']==counts
                    routing=route_metrics(controller.routes[i],routes);assert routing['changed_tokens']==0
                    isolated.append(dict(case=case,rank=rank,layer=i,backend=backend,m=controller.rows[i]['m'],counts=counts,
                                         path=controller.calls[i],routing=routing,error=metrics(actual,expected)))
                controller.capture=False;del expected,actual,ep_x
            write(a.output/f'isolated_ep_rank{rank}.json',isolated)
            if cache:
                assert all(torch.equal(k,bk) and torch.equal(v,bv) for (k,v),(bk,bv) in zip(cache,cache_before))
            del reference,expected_logits,expected_hidden,cache,cache_before,x,saved
            controller.release_workspaces();torch.cuda.empty_cache();dist.barrier()

    # Artificial routers target zero receivers, zero source rank, expert skew and tail tiles.
    mlp=model.model.layers[0].mlp;gate_original=mlp.gate.forward
    cases=[('rank1_zero',1,17,23,'all0'),('rank0_zero',1,17,23,'all2'),
           ('top2_same_rank',2,17,23,'pair01'),('top2_cross_rank',2,31,33,'pair12'),
           ('skew_tail',1,65,65,'skew'),('zero_source',1,0,17,'all0')]
    for label,topk,left,right,pattern in cases:
        config.num_experts_per_tok=topk;tokens=left if rank==0 else right
        torch.manual_seed(20261007+rank);ep_x=(torch.randn((1,tokens,768),device=device)*.1).to(torch.bfloat16)
        def gate(_,value):
            logits=torch.full((len(value),4),-4.,device=value.device,dtype=torch.bfloat16)
            if pattern in ['all0','all2']:logits[:,0 if pattern=='all0' else 2]=8
            elif pattern in ['pair01','pair12']:
                first=0 if pattern=='pair01' else 1;logits[:,first]=8;logits[:,first+1]=7
            else:
                logits[:,0]=8
                if len(value):logits[-1,0]=-4;logits[-1,3]=8
            return logits
        mlp.gate.forward=types.MethodType(gate,mlp)
        controller.backend='original';controller.capture=True;controller.reset_capture()
        expected=mlp(ep_x);counts=controller.rows[0]['counts'];routes=controller.routes[0]
        local_sum=torch.tensor([sum(counts)],device=device);dist.all_reduce(local_sum)
        assert int(local_sum)==(left+right)*topk
        if label=='rank1_zero' or label=='top2_same_rank':assert sum(counts)==0 if rank==1 else sum(counts)>0
        if label=='rank0_zero':assert sum(counts)==0 if rank==0 else sum(counts)>0
        for backend in ['kernel9','hybrid']:
            controller.backend=backend;controller.reset_capture();actual=mlp(ep_x)
            assert controller.rows[0]['counts']==counts and route_metrics(controller.routes[0],routes)['changed_tokens']==0
            error=metrics(actual,expected);assert error['strict_pass'],(label,backend,error)
            boundaries.append(dict(case=label,rank=rank,backend=backend,source_tokens=tokens,counts=counts,error=error))
        controller.capture=False;mlp.gate.forward=gate_original
    write(a.output/f'boundaries_rank{rank}.json',boundaries)

    # Free-running greedy decode. Each backend builds and advances its own KV cache.
    from measure_baseline import generate_inputs
    for topk in [1,2]:
        config.num_experts_per_tok=topk
        for batch in [1,4]:
            prompt,_=generate_inputs(a.repo,rank,batch,128,topk);prompt=prompt.to(device)
            outputs={}
            for backend in ['original','kernel9','hybrid']:
                controller.backend=backend;prefill=model(prompt,use_cache=True,logits_to_keep=1)
                kv=prefill.past_key_values;token=prefill.logits[:,-1].argmax(-1,keepdim=True);seq=[]
                for step in range(8):
                    seq.append(token.cpu());result=model(token,past_key_values=kv,use_cache=True,logits_to_keep=1)
                    kv=result.past_key_values;token=result.logits[:,-1].argmax(-1,keepdim=True)
                outputs[backend]=torch.cat(seq,dim=1);del prefill,kv,result
            for backend in ['kernel9','hybrid']:
                changed=outputs[backend]!=outputs['original']
                generations.append(dict(rank=rank,topk=topk,batch=batch,backend=backend,steps=8,
                                        changed_tokens=int(changed.sum()),elements=changed.numel(),
                                        original=outputs['original'].tolist(),candidate=outputs[backend].tolist()))
            controller.release_workspaces();torch.cuda.empty_cache();dist.barrier()
    write(a.output/f'greedy_rank{rank}.json',generations)

# A real differentiable two-rank forward/backward must bypass the inference backend.
config.num_experts_per_tok=2;mlp=model.model.layers[0].mlp;mlp.train();controller.capture=False
torch.manual_seed(20261007+rank);base=(torch.randn((1,2,768),device=device)*.1).to(torch.bfloat16)
gradients=[]
for backend in ['original','kernel9']:
    controller.backend=backend;mlp.zero_grad(set_to_none=True);x=base.clone().requires_grad_()
    with torch.autocast('cuda',dtype=torch.bfloat16):y=mlp(x);loss=y.float().square().mean()+.01*mlp.aux_loss
    loss.backward();gradients.append(dict(output=y.detach().clone(),x=x.grad.clone(),
                                         parameters={k:v.grad.detach().clone() for k,v in mlp.named_parameters() if v.grad is not None}))
assert torch.equal(gradients[0]['output'],gradients[1]['output']) and torch.equal(gradients[0]['x'],gradients[1]['x'])
assert gradients[0]['parameters'].keys()==gradients[1]['parameters'].keys()
assert all(torch.equal(gradients[0]['parameters'][k],gradients[1]['parameters'][k]) for k in gradients[0]['parameters'])
assert 0 not in controller.states;mlp.eval();mlp.zero_grad(set_to_none=True)
write(a.output/f'training_rank{rank}.json',dict(passed=True,output_exact=True,input_gradient_exact=True,
                                            parameter_gradients_exact=True,parameter_gradients=len(gradients[0]['parameters']),cache_invalidated=True))
write(a.output/f'summary_rank{rank}.json',dict(completed=True,rank=rank,model_comparisons=len(full),isolated_comparisons=len(isolated),
                                           boundary_comparisons=len(boundaries),greedy_comparisons=len(generations),training_fallback=True,
                                           strict_model_passes=sum(r['logits']['strict_pass'] for r in full),
                                           next_tokens_changed=sum(r['next_tokens_changed'] for r in full)))
dist.barrier();dist.destroy_process_group();print('EP VALIDATION COMPLETE; numerical differences retained',rank,flush=True)
