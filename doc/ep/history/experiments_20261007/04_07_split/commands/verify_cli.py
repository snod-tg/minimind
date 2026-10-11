"""Exercise the real eval init_model entry point on staged MiniMind sources."""
import argparse,sys,json,os,gc
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--library',required=True);p.add_argument('--kn-checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--actual',action='store_true')
a=p.parse_args();root=Path(__file__).resolve().parent;sys.path.insert(0,a.repo);
if not a.actual:sys.path.insert(0,str(root/'minimind'))
from eval_llm import init_ep,init_model
from ep_common import metrics
args=SimpleNamespace(use_ep=True,load_from=str(Path(a.repo)/'model'),device='cuda',use_moe=1,save_dir=str(Path(a.repo)/'out'),weight='pretrain_ep',hidden_size=768,num_hidden_layers=8,ep_comm_dtype='native',num_experts=4,num_experts_per_tok=1,inference_rope_scaling=False,lora_weight='None',kn_checkpoint=None,grouped_gemm_python_dir=str(root/'source/dev/pytorch'),grouped_gemm_library=a.library,model_dtype='bf16',expert_backend='original')
rank,_=init_ep(args);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
x=torch.tensor([[1,45,70,83]],device=args.device);rows=[];reference={};out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
with torch.inference_mode():
 for layout,checkpoint in [('nk',None),('kn',a.kn_checkpoint)]:
  for backend in ('original','native_bf16','cuda_split_last_layer','cuda_split'):
   args.kn_checkpoint=checkpoint;args.expert_backend=backend
   with torch.inference_mode(False):model,tokenizer=init_model(args)
   assert next(model.parameters()).dtype==torch.bfloat16
   controller=getattr(model.model.layers[0].mlp,'_grouped_expert_backend',None)
   if controller:controller.capture=True
   for topk in (1,2):
    model.config.num_experts_per_tok=topk
    y=model(x,use_cache=True);decode=model(y.logits[:,-1].argmax(-1,keepdim=True),past_key_values=y.past_key_values,use_cache=True)
    pair=(y.logits.detach().cpu(),decode.logits.detach().cpu());key=(layout,topk)
    if backend=='original':reference[key]=pair
    equal=[torch.equal(v,w) for v,w in zip(pair,reference[key])]
    errors=[metrics(v,w) for v,w in zip(pair,reference[key])]
    if backend=='cuda_split_last_layer':assert all(e['strict_pass'] for e in errors),errors
    if backend=='native_bf16':assert all(equal),equal
    if layout=='kn':assert all(torch.equal(v,w) for v,w in zip(pair,reference[('nk',topk,backend)]))
    reference[(layout,topk,backend)]=pair
    rows.append(dict(layout=layout,backend=backend,topk=topk,dtype=str(next(model.parameters()).dtype),original_bitwise=equal,error=errors,paths=controller.calls if controller else {},next_tokens=[v[:,-1].argmax(-1).tolist() for v in pair]))
   if controller:controller.uninstall()
   del model,tokenizer,y,decode;gc.collect();torch.cuda.empty_cache();dist.barrier()
 (out/f'cli_rank{rank}.json').write_text(json.dumps(rows,indent=2));print('CLI INITIALIZATION PASSED',rank,len(rows),flush=True)
dist.destroy_process_group()
