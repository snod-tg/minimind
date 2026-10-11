"""Actual EP extension point with KN checkpoint and FP16 inference dtype."""
import argparse,json,sys
from pathlib import Path
import torch
import torch.distributed as dist
from ep_common import setup,metrics,write
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--source',type=Path,required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--kn',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();rank,device,module,nk,config=setup(a.source,a.checkpoint,a.output)
from grouped_gemm_split import load_library
from minimind_ep_split_backend import MiniMindEPSplitBackend
from checkpoint_kn import load_kn_checkpoint
load_library(a.library)
kn=load_kn_checkpoint(module.MiniMindForCausalLM(config),a.kn,ep_subset=True).to(device).eval()
left=MiniMindEPSplitBackend(nk);right=MiniMindEPSplitBackend(kn)
keys_before=set(nk.state_dict());rows=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for topk in (1,2):
  config.num_experts_per_tok=topk
  for batch,length in ((1,17),(1,128),(8,340)):
   torch.manual_seed(20261009+rank);ids=torch.randint(3,6400,(batch,length),device=device)
   left.backend=right.backend='original';ref=nk(ids,use_cache=True,logits_to_keep=1)
   for backend in ('native_bf16','cuda_split','cuda_split_last_layer'):
    left.backend=right.backend=backend
    x=nk(ids,use_cache=True,logits_to_keep=1);y=kn(ids,use_cache=True,logits_to_keep=1)
    assert torch.equal(x.logits,y.logits),(topk,batch,length,backend)
    assert all(torch.equal(u,v) for xx,yy in zip(x.past_key_values,y.past_key_values) for u,v in zip(xx,yy))
    if backend=='native_bf16':assert torch.equal(y.logits,ref.logits)
    for step in range(4):
     token=x.logits[:,-1].argmax(-1,keepdim=True)
     x=nk(token,past_key_values=x.past_key_values,use_cache=True,logits_to_keep=1)
     y=kn(token,past_key_values=y.past_key_values,use_cache=True,logits_to_keep=1)
     assert torch.equal(x.logits,y.logits),(backend,step)
    rows.append(dict(rank=rank,topk=topk,batch=batch,length=length,backend=backend,kn_vs_nk_bitwise=True,decode_steps=4))
    write(a.output/f'kn_rank{rank}.json',rows)
   left.release_workspaces();right.release_workspaces();torch.cuda.empty_cache();dist.barrier()
# FP16 matches the existing eval CLI's default dtype; no autocast needed.
nk.half();rows16=[]
with torch.inference_mode():
 for topk in (1,2):
  config.num_experts_per_tok=topk
  ids=torch.randint(3,6400,(1,128),device=device);left.backend='original'
  ref=nk(ids,use_cache=True,logits_to_keep=1)
  for backend in ('native_bf16','cuda_split','cuda_split_last_layer'):
   left.backend=backend;actual=nk(ids,use_cache=True,logits_to_keep=1);error=metrics(actual.logits,ref.logits)
   if backend=='native_bf16':assert error['exact']
   if backend=='cuda_split_last_layer':assert error['strict_pass']
   rows16.append(dict(rank=rank,topk=topk,backend=backend,error=error))
 write(a.output/f'fp16_rank{rank}.json',rows16)
assert set(nk.state_dict())==keys_before
left.uninstall();right.uninstall()
write(a.output/f'summary_rank{rank}.json',dict(kn_cases=len(rows),decode_steps=4*len(rows),fp16_cases=len(rows16),state_dict_keys_unchanged=True,uninstall=True,bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
dist.barrier();dist.destroy_process_group();print('ALL KN INTEGRATION CHECKS PASSED',rank,flush=True)
