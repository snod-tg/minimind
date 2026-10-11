"""Two-rank EP numerical regression of canonical KN loading."""
import argparse,json,sys,os
from pathlib import Path
import torch,torch.distributed as dist
p=argparse.ArgumentParser();p.add_argument('--stage6',required=True);p.add_argument('--source',required=True);p.add_argument('--original',required=True);p.add_argument('--converted',required=True);p.add_argument('--output',required=True);a=p.parse_args()
sys.path.insert(0,a.stage6);sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dev/pytorch'))
from checkpoint_kn import load_kn_checkpoint,prepare_native_caches,save_kn_checkpoint,decode_state
from ep_common import setup,metrics
out=Path(a.output);rank,device,module,original,config=setup(Path(a.source),a.original,out)
kn=load_kn_checkpoint(module.MiniMindForCausalLM(config),a.converted,ep_subset=True).to(device).eval()
assert kn.lm_head.weight is kn.model.embed_tokens.weight
prepare_native_caches(kn);torch.manual_seed(20261007+rank);rows=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for topk in [1,2]:
  config.num_experts_per_tok=topk
  for batch,length in [(1,1),(2,17),(1,128),(8,340),(1,513)]:
   ids=torch.randint(3,6400,(batch,length),device=device)
   ref=original(ids,use_cache=True);actual=kn(ids,use_cache=True)
   assert torch.equal(ref.logits,actual.logits),(rank,topk,batch,length,metrics(actual.logits,ref.logits))
   assert all(torch.equal(x,y) for r,s in zip(ref.past_key_values,actual.past_key_values) for x,y in zip(r,s))
   for _ in range(4):
    token=ref.logits[:,-1].argmax(-1,keepdim=True)
    ref=original(token,past_key_values=ref.past_key_values,use_cache=True);actual=kn(token,past_key_values=actual.past_key_values,use_cache=True)
    assert torch.equal(ref.logits,actual.logits),(rank,topk,batch,length,'decode')
   rows.append(dict(rank=rank,topk=topk,batch=batch,length=length,bitwise_native=True,bitwise_kv=True,decode_steps=4))
   (out/f'cases_rank{rank}.json').write_text(json.dumps(rows,indent=2));print('EP',rows[-1],flush=True)
 save_kn_checkpoint(kn,out/'ep_resaved_kn.pth')
 if rank==0:
  saved=torch.load(out/'ep_resaved_kn.pth',map_location='cpu',weights_only=True)
  recovered=decode_state(saved);source=torch.load(a.original,map_location='cpu',weights_only=True)
  assert recovered.keys()==source.keys() and all(torch.equal(source[key],recovered[key]) for key in source)
 load_kn_checkpoint(kn,out/'ep_resaved_kn.pth',ep_subset=True)
 assert kn.lm_head.weight is kn.model.embed_tokens.weight
 ref=original(ids,use_cache=False);actual=kn(ids,use_cache=False);assert torch.equal(ref.logits,actual.logits)
 (out/f'summary_rank{rank}.json').write_text(json.dumps(dict(ep_full_save=True,ep_reload=True,cases=len(rows),decode_steps=40,bitwise_native=True,aliases=True,allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),indent=2))
dist.barrier();dist.destroy_process_group();print('ALL CHECKS PASSED',rank)
