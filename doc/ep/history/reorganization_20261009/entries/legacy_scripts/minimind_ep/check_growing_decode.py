"""Match stage7's B16, eight-step, original-KV-start decode numerical scope."""
import argparse,sys
from pathlib import Path
import torch
import torch.distributed as dist
from ep_common import setup,metrics,write
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--library',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();rank,device,module,model,config=setup(a.source,a.checkpoint,a.output)
from grouped_gemm_api import load_library
from minimind_ep_backend import MiniMindEPBackend
from measure_baseline import generate_inputs
load_library(a.library);controller=MiniMindEPBackend(model);controller.initialize();rows=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for topk in [1,2]:
  config.num_experts_per_tok=topk
  prompt,_=generate_inputs(a.repo,rank,16,128,topk);prompt=prompt.to(device)
  controller.backend='original';initial=model(prompt,use_cache=True,logits_to_keep=1)
  kv0=initial.past_key_values;token0=initial.logits[:,-1].argmax(-1,keepdim=True);reference=[];sequences={}
  for backend in ['original','kernel9','hybrid','last_layer']:
   controller.backend=backend;kv=kv0;token=token0;seq=[];errors=[]
   for step in range(8):
    seq.append(token.cpu());out=model(token,past_key_values=kv,use_cache=True,logits_to_keep=1)
    if backend=='original':reference.append(out.logits.cpu())
    else:errors.append(metrics(out.logits,reference[step]))
    kv=out.past_key_values;token=out.logits[:,-1].argmax(-1,keepdim=True)
   sequences[backend]=torch.cat(seq,dim=1)
   if backend!='original':
    rows.append(dict(rank=rank,topk=topk,batch=16,steps=8,backend=backend,logits=errors,
                     changed_tokens=int((sequences[backend]!=sequences['original']).sum()),elements=128))
  controller.release_workspaces();torch.cuda.empty_cache();dist.barrier()
write(a.output/f'growing_rank{rank}.json',rows)
write(a.output/f'summary_rank{rank}.json',dict(completed=True,rank=rank,comparisons=len(rows),
     last_layer_strict_pass=all(e['strict_pass'] for r in rows if r['backend']=='last_layer' for e in r['logits']),
     last_layer_tokens_changed=sum(r['changed_tokens'] for r in rows if r['backend']=='last_layer')))
dist.barrier();dist.destroy_process_group();print('GROWING B16 AUDIT COMPLETE',rank,flush=True)
