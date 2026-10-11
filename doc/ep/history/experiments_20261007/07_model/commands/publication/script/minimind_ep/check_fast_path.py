"""Audit-disabled production call must match the annotated/captured math exactly."""
import argparse,hashlib,sys
from pathlib import Path
import torch
import torch.distributed as dist
from ep_common import setup,inputs,write,WORKLOADS
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--library',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();rank,device,module,model,config=setup(a.source,a.checkpoint,a.output)
from grouped_gemm_api import load_library
from minimind_ep_backend import MiniMindEPBackend
load_library(a.library);controller=MiniMindEPBackend(model);controller.initialize();rows=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for topk in [1,2]:
  config.num_experts_per_tok=topk
  for name in WORKLOADS:
   controller.backend='original';x,cache,_=inputs(a.repo,rank,name,topk,model,a.baseline)
   for backend in ['kernel9','hybrid','last_layer']:
    controller.backend=backend;controller.profile=True;audited=model(x,past_key_values=cache,use_cache=True)
    controller.profile=False;fast=model(x,past_key_values=cache,use_cache=True)
    assert torch.equal(audited.logits,fast.logits) and torch.equal(audited.hidden_states,fast.hidden_states),(name,topk,backend)
    rows.append(dict(case=f'{name}_top{topk}',rank=rank,backend=backend,logits_exact=True,hidden_exact=True))
    del audited,fast
   write(a.output/f'fast_paths_rank{rank}.json',rows);del x,cache
   controller.release_workspaces();torch.cuda.empty_cache();dist.barrier()
write(a.output/f'summary_rank{rank}.json',dict(passed=True,comparisons=len(rows),adapter_sha256=hashlib.sha256((a.source/'dev/pytorch/minimind_ep_backend.py').read_bytes()).hexdigest()))
dist.barrier();dist.destroy_process_group();print('ALL FAST/AUDIT PATHS EXACT',rank,flush=True)
