"""Read-only acceptance of an exported EP model and per-rank optimizer states."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def finite_tree(value):
    if isinstance(value, torch.Tensor):
        assert not value.is_floating_point() or bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor'
        return 1
    if isinstance(value, dict):
        return sum(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(finite_tree(v) for v in value)
    return 0

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--training-config',type=Path,required=True)
    p.add_argument('--resume-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--routing',choices=['original','compact'],default='original')
    p.add_argument('--library')
    p.add_argument('--expected-step',type=int,default=19848)
    p.add_argument('--expected-epoch',type=int,default=1)
    a=p.parse_args()
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(1)
    device=torch.device('cuda',local)
    dist.init_process_group('nccl',device_id=device)
    record=json.loads(a.training_config.read_text());values=dict(record['model'])
    values.update(use_ep=True,ep_size=dist.get_world_size())
    config=MiniMindConfig(**values)
    assert (config.num_experts,config.num_experts_per_tok)==(6,2)
    model=MiniMindForCausalLM(config)
    full=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    assert all('_orig_mod' not in key for key in full)
    tensors=finite_tree(full)
    for layer in range(config.num_hidden_layers):
        for expert in range(config.num_experts):
            for name,shape in [('gate_proj',(2432,768)),('up_proj',(2432,768)),('down_proj',(768,2432))]:
                key=f'model.layers.{layer}.mlp.experts.{expert}.{name}.weight'
                assert tuple(full[key].shape)==shape,(key,full[key].shape)
    local_state={key:full[key] for key in model.state_dict()}
    model.load_state_dict(local_state,strict=True)
    resume_path=a.resume_dir/f'{a.checkpoint.stem}_ep_rank{rank}_resume.pth'
    resume=torch.load(resume_path,map_location='cpu',weights_only=False,mmap=True)
    assert resume['world_size']==2 and resume['epoch']==a.expected_epoch and resume['step']==a.expected_step
    assert resume['model'].keys()==local_state.keys()
    assert all(torch.equal(local_state[key],value) for key,value in resume['model'].items())
    resume_tensors=finite_tree(resume)
    optimizer=torch.optim.AdamW(model.parameters(),lr=record['arguments']['learning_rate'])
    optimizer.load_state_dict(resume['optimizer'])
    assert len(optimizer.state)>0
    optimizer_steps=[]
    for parameter,state in optimizer.state.items():
        assert state['exp_avg'].shape==parameter.shape and state['exp_avg_sq'].shape==parameter.shape
        assert float(state['step'])>0
        optimizer_steps.append(float(state['step']))
    assert model.model.embed_tokens.weight is model.lm_head.weight
    checkpoint_sha=[sha256(a.checkpoint) if rank==0 else None]
    dist.broadcast_object_list(checkpoint_sha,src=0)
    summary=dict(rank=rank,experts=6,topk=2,checkpoint_sha256=checkpoint_sha[0],
        checkpoint_tensors_finite=tensors,resume_tensors_finite=resume_tensors,
        resume_epoch=resume['epoch'],resume_step=resume['step'],optimizer_restored=True,
        optimizer_parameters=len(optimizer.state),optimizer_step_min=min(optimizer_steps),
        optimizer_step_max=max(optimizer_steps),full_vs_resume_bitwise=True,tied_parameter_preserved=True,
        checkpoint_dtypes=sorted({str(v.dtype) for v in full.values()}),parameter_layout='NK',
        model_parameters_dtype=str(next(model.parameters()).dtype),routing=a.routing,
        model_source_sha256=sha256(Path(__import__('model.model_minimind',fromlist=['x']).__file__)),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'])
    del optimizer,resume,full,local_state;gc.collect()
    model.to(device).eval()
    controller=None
    if a.routing=='compact':
        from model.ep_routing_backend import attach_ep_routing
        controller=attach_ep_routing(model,library=a.library,mode='compact')
    tokenizer=AutoTokenizer.from_pretrained(Path(__import__('model.model_minimind',fromlist=['x']).__file__).parent,local_files_only=True)
    expected=torch.load(a.reference/f'outputs_rank{rank}.pt',map_location='cpu',weights_only=True) if a.reference else None
    captured={};checks=[];generations=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for length in [1,17,128,340]:
            generator=torch.Generator(device=device).manual_seed(501+rank+length)
            ids=torch.randint(3,config.vocab_size,(1,length),device=device,generator=generator)
            out=model(ids,use_cache=True,logits_to_keep=1)
            assert bool(torch.isfinite(out.logits).all()) and bool(torch.isfinite(out.hidden_states).all())
            key=f'prefill_{length}'
            value=dict(logits=out.logits.cpu(),hidden=out.hidden_states.cpu())
            next_ids=out.logits[:,-1].argmax(-1,keepdim=True)
            decoded=model(next_ids,past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1)
            assert bool(torch.isfinite(decoded.logits).all())
            value['decode_logits']=decoded.logits.cpu();captured[key]=value
            if expected is not None:
                assert all(torch.equal(value[name],expected[key][name]) for name in value),key
            checks.append(dict(case=key,finite=True,reference_exact=expected is not None))
        for i,prompt in enumerate(['人工智能的主要应用包括','哈尔滨位于','实现专家并行时，需要将输入数据']):
            ids=tokenizer(prompt,return_tensors='pt')['input_ids'].to(device);past=None;generated=[]
            for step in range(16):
                out=model(ids,past_key_values=past,use_cache=True,logits_to_keep=1)
                assert bool(torch.isfinite(out.logits).all())
                ids=out.logits[:,-1].argmax(-1,keepdim=True);dist.broadcast(ids,src=0)
                generated.append(int(ids[0,0]));past=out.past_key_values
            captured[f'generation_{i}']=torch.tensor(generated)
            if expected is not None:assert torch.equal(captured[f'generation_{i}'],expected[f'generation_{i}'])
            generations.append(dict(prompt=prompt,completion=tokenizer.decode(generated),tokens=generated))
    summary.update(checks=checks,generations=generations,passed=True,
        optimized_calls=controller.calls if controller else 0,fallbacks=controller.fallbacks if controller else 0,
        quality_claim='Numerical/loading acceptance only; no held-out quality evaluation')
    assert summary['fallbacks']==0
    a.output.mkdir(parents=True,exist_ok=True)
    assert not (a.output/f'rank{rank}.json').exists(),'Choose a new output directory'
    torch.save(captured,a.output/f'outputs_rank{rank}.pt')
    (a.output/f'rank{rank}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
