"""Verify complete expert export, shard resumes, and optimized EP reload."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

p=argparse.ArgumentParser()
p.add_argument('--library',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local);torch.set_num_threads(1)
dist.init_process_group('nccl',device_id=torch.device('cuda',local))
stage=Path('/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage09_compile_20261009')
stage8=stage.parent/'stage08_ep_20261009'
sys.path.insert(0,str(stage/'minimind'))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.ep_routing_backend import attach_ep_routing
full=torch.load(stage/'minimind/out/ep_compile_smoke_768_moe.pth',map_location='cpu',weights_only=True)
resume=torch.load(stage/f'minimind/checkpoints/ep_compile_smoke_768_moe_ep_rank{rank}_resume.pth',map_location='cpu',weights_only=False)
assert resume['step']==2 and resume['world_size']==2 and resume['optimizer']['state']
assert all('_orig_mod' not in key for key in full)
for layer in range(8):
    for expert in range(6):
        for projection,shape in [('gate_proj',(2432,768)),('up_proj',(2432,768)),('down_proj',(768,2432))]:
            key=f'model.layers.{layer}.mlp.experts.{expert}.{projection}.weight'
            assert tuple(full[key].shape)==shape and full[key].dtype==torch.float16,key
assert all(torch.equal(full[key],value) for key,value in resume['model'].items())
config=MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True,use_ep=True,ep_size=2,num_experts=6,num_experts_per_tok=2)
model=MiniMindForCausalLM(config)
model.load_state_dict({key:full[key] for key in model.state_dict()},strict=True)
model.cuda().eval()
del full,resume
controller=attach_ep_routing(model,library=a.library,python_dir=stage8/'source/dev/pytorch')
rows=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for length in [1,17,33]:
        torch.manual_seed(501+rank)
        ids=torch.randint(3,6400,(1,length),device='cuda')
        controller.enabled=False;ref=model(ids)
        controller.enabled=True;y=model(ids)
        assert torch.equal(ref.logits,y.logits) and torch.equal(ref.hidden_states,y.hidden_states)
        rows.append(dict(tokens=length,logits_exact=True,hidden_exact=True))
a.output.mkdir(parents=True,exist_ok=True)
(a.output/f'rank{rank}.json').write_text(json.dumps(dict(export_complete=True,resume_optimizer_present=True,parameter_layout='NK',experts=6,topk=2,checks=rows),indent=2)+'\n')
dist.barrier();dist.destroy_process_group()
print('TRAINED CHECKPOINT RELOAD PASSED',rank,flush=True)
