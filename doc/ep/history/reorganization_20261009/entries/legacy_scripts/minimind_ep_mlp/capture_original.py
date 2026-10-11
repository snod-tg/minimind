"""Capture missing large-prefill/single-decode MLP inputs using the unchanged EP model."""
import argparse,sys,json,os,importlib.util,hashlib
from pathlib import Path
import torch
import torch.distributed as dist
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.source/'script'))
from measure_baseline import Probe,instrument_ep_forward,generate_inputs
rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
dist.init_process_group('nccl',device_id=device);assert dist.get_world_size()==2
source=a.source/'script/model_minimind_snapshot.py'
spec=importlib.util.spec_from_file_location('mlp_capture_model',source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
config=module.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True,use_ep=True,ep_size=2,num_experts=4,num_experts_per_tok=1,moe_intermediate_size=2432,dropout=0.,ep_comm_dtype='native',max_position_embeddings=4096)
model=module.MiniMindForCausalLM(config);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
model.load_state_dict({key:state[key] for key in model.state_dict()},strict=True);del state
model=model.to(device).eval();original,annotated,_=instrument_ep_forward(module)
class AllLayerProbe(Probe):
 def collect(self,layer,x,counts,y,mlp):
  if not self.capture:return
  super().collect(layer,x,counts,y,mlp)
  if layer not in (0,3,7):self.replays.append(dict(layer=layer,rank=self.rank,counts=counts,local_expert_ids=list(mlp.local_expert_ids),x=x.detach().cpu(),y=y.detach().cpu()))
probe=AllLayerProbe(a.output,rank);annotated.__globals__['probe']=probe
for i,layer in enumerate(model.model.layers):layer.mlp._baseline_layer=i
checks=[];a.output.mkdir(parents=True,exist_ok=True)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for topk in [1,2]:
  config.num_experts_per_tok=topk
  for name,batch,length,context in [('prefill_large',32,340,0),('decode_single',1,1,128)]:
   case=f'{name}_top{topk}';out=a.output/case;out.mkdir(exist_ok=True)
   module.ExpertParallelMOEFeedForward.forward=original
   cpu,input_info=generate_inputs(a.repo,rank,batch,context or length,topk);inputs=cpu.to(device);cache=None
   if context:
    initial=model(inputs,use_cache=True);cache=initial.past_key_values;inputs=initial.logits[:,-1].argmax(-1,keepdim=True);del initial
   baseline=torch.load(a.baseline/case/f'capture_rank{rank}.pt',map_location='cpu',weights_only=True)
   assert torch.equal(inputs.cpu(),baseline['input_ids'])
   reference=model(inputs,past_key_values=cache,use_cache=True)
   logits=reference.logits.detach().clone();hidden=reference.hidden_states.detach().clone();del reference
   module.ExpertParallelMOEFeedForward.forward=annotated
   probe.capture=True;probe.save_replay=True;probe.rows=[];probe.replays=[]
   captured=model(inputs,past_key_values=cache,use_cache=True)
   assert torch.equal(logits,captured.logits) and torch.equal(hidden,captured.hidden_states)
   old={r['layer']:r['counts'] for r in baseline['counts']}
   assert all(r['counts']==old[r['layer']] for r in probe.rows)
   assert len(probe.replays)==8
   torch.save(dict(input_ids=inputs.cpu(),replays=probe.replays,counts=probe.rows,checkpoint=a.checkpoint),out/f'capture_rank{rank}.pt')
   check=dict(case=case,rank=rank,original_output_exact=True,baseline_inputs_exact=True,baseline_counts_exact=True,replays=8,m_max=max(sum(r['counts']) for r in probe.replays),input=input_info)
   checks.append(check);print(json.dumps(check),flush=True)
   probe.capture=False;probe.replays=[];probe.rows=[];del logits,hidden,captured,cache,baseline
   torch.cuda.empty_cache();dist.barrier()
record=dict(passed=True,rank=rank,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),checks=checks,torch=torch.__version__,python=sys.version,physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'))
(a.output/f'summary_rank{rank}.json').write_text(json.dumps(record,indent=2)+'\n');dist.barrier();dist.destroy_process_group();print('ALL CAPTURE CHECKS PASSED',flush=True)
