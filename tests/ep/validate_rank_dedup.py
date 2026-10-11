"""Distributed EP boundary checks against the previous compact implementation."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.ep_routing_backend import attach_ep_routing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(1)
    dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    assert dist.get_world_size()==2
    records=[]
    for experts in (6,8):
        torch.manual_seed(61)
        cfg=MiniMindConfig(hidden_size=64,num_hidden_layers=1,num_attention_heads=4,
            num_key_value_heads=2,intermediate_size=128,moe_intermediate_size=128,
            use_moe=True,use_ep=True,ep_size=2,num_experts=experts,num_experts_per_tok=2,
            ep_comm_dtype='native',router_aux_loss_coef=0.)
        model=MiniMindForCausalLM(cfg).cuda().train();owner=model.model.layers[0].mlp
        for case in ('same_rank','cross_rank','random','empty_source','empty_both'):
            n=0 if case=='empty_both' or case=='empty_source' and rank==1 else 17+rank
            torch.manual_seed(80+rank)
            x0=torch.randn(n,64,device='cuda')
            weights0=torch.rand(n,2,device='cuda');upstream=torch.randn(n,64,device='cuda')
            if case=='same_rank':ids=torch.tensor([0,1],device='cuda').expand(n,2).contiguous()
            elif case=='cross_rank':ids=torch.tensor([0,experts//2],device='cuda').expand(n,2).contiguous()
            else:ids=torch.rand(n,experts,device='cuda').topk(2).indices
            reference=None
            for mode in ('compact_legacy','compact','dedup_train'):
                model.zero_grad(set_to_none=True)
                controller=attach_ep_routing(model,library=a.library,mode=mode)
                x=x0.clone().requires_grad_();weights=weights0.clone().requires_grad_()
                with torch.autocast('cuda',dtype=torch.bfloat16):out=controller(owner,x,ids,weights)
                out.backward(upstream)
                gradients={key:p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                    for key,p in owner.named_parameters()}
                values=(out.detach().clone(),x.grad.detach().clone(),weights.grad.detach().clone(),gradients)
                if reference is None:reference=values
                else:
                    torch.testing.assert_close(values[0],reference[0],atol=0,rtol=0)
                    torch.testing.assert_close(values[1],reference[1],atol=2e-5,rtol=5e-4)
                    torch.testing.assert_close(values[2],reference[2],atol=0,rtol=0)
                    for key,value in gradients.items():torch.testing.assert_close(value,reference[3][key],atol=2e-5,rtol=5e-4)
                assert controller.fallbacks==0
                controller.uninstall()
            records.append(dict(experts=experts,topk=2,case=case,rank=rank,tokens=n,forward_exact=True,router_weight_gradient_exact=True,input_and_expert_gradients_close=True))
        del model
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/f'rank{rank}.json').write_text(json.dumps(dict(passed=True,cases=records),indent=2)+'\n')
    print('DISTRIBUTED DEDUP PASSED',rank,len(records),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
