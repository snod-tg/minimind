"""Paired original/compact EP inference on one trained six-expert Top-2 model."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
CASES=[('decode_b1',1,1,128),('decode_b16',16,1,128),
       ('prefill_m128',1,128,0),('prefill_m2720',8,340,0),('prefill_m10880',32,340,0)]

def median(values):return statistics.median(values)

def gpu_processes(selected):
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    rows=[]
    for line in query.splitlines():
        uuid,pid,memory=map(str.strip,line.split(','))
        if uuid not in selected:continue
        try:cmd=(Path('/proc')/pid/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except OSError:cmd=''
        rows.append(dict(uuid=uuid,pid=int(pid),memory=memory,own='scripts.ep.benchmark_trained_ep' in cmd))
    return rows

def launch(a):
    from .train import validate_gpus
    ids=validate_gpus(a.gpus)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    selected={}
    for line in inventory.splitlines():
        index,uuid,memory,util=map(str.strip,line.split(','))
        if int(index) in ids:
            assert int(memory)<100 and int(util)==0,('GPU is occupied',line)
            selected[uuid]=int(index)
    assert len(selected)==2
    assert not gpu_processes(selected)
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--module','scripts.ep.benchmark_trained_ep',
             '--checkpoint',str(a.checkpoint.resolve()),'--training-config',str(a.training_config.resolve()),
             '--output',str(a.output),'--rounds',str(a.rounds),'--warmup',str(a.warmup)]
    if a.library:command+=['--library',str(Path(a.library).resolve())]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    (a.output/'command.json').write_text(json.dumps(dict(command=command,cwd=str(ROOT),gpus=a.gpus,inventory=inventory),indent=2)+'\n')
    observations=[]
    with (a.output/'console.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while True:
            observations.append(dict(time=time.time(),processes=gpu_processes(selected)))
            if process.poll() is not None:break
            time.sleep(1)
    foreign=[p for row in observations for p in row['processes'] if not p['own']]
    (a.output/'gpu_monitor.json').write_text(json.dumps(dict(selected=selected,foreign=foreign,observations=observations,exit_code=process.returncode),indent=2)+'\n')
    (a.output/'exit_code.txt').write_text(str(process.returncode)+'\n')
    if process.returncode:
        print((a.output/'console.log').read_text(errors='replace')[-6000:]);raise SystemExit(process.returncode)
    assert not foreign,('Selected GPUs had foreign work; do not accept these timings',foreign)
    summary=json.loads((a.output/'summary.json').read_text())
    summary.update(occupancy_clean=True,occupancy_samples=len(observations),physical_gpus=a.gpus)
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def worker(a):
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
    from model.ep_routing_backend import attach_ep_routing
    from tests.ep.validate_trained_checkpoint import sha256
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(1)
    device=torch.device('cuda',local);dist.init_process_group('nccl',device_id=device)
    config=MiniMindConfig(**json.loads(a.training_config.read_text())['model'])
    assert config.num_experts==6 and config.num_experts_per_tok==2 and dist.get_world_size()==2
    model=MiniMindForCausalLM(config)
    full=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict({k:full[k] for k in model.state_dict()},strict=True)
    del full;gc.collect();model.to(device).eval()
    controller=attach_ep_routing(model,library=a.library,mode='compact')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    corpus='人工智能通过数据学习规律，并应用于语言理解、图像识别和科学研究。哈尔滨位于中国东北，冬季寒冷，冰雪文化闻名。专家并行把不同专家分配到多张显卡，路由器根据每个输入选择专家，通信负责发送输入并合并输出。矩阵乘法的性能取决于计算效率、访存和调度开销。'
    tokens=torch.tensor(tokenizer(corpus,add_special_tokens=False)['input_ids'],device=device)
    def inputs(batch,length):
        offset=torch.arange(batch,device=device)[:,None]*37+rank*53
        return tokens[(torch.arange(length,device=device)[None,:]+offset)%len(tokens)]
    records=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for name,batch,length,context in CASES:
            controller.workspaces.clear();past=None
            ids=inputs(batch,context or length)
            if context:
                controller.enabled=False
                prefill=model(ids,use_cache=True,logits_to_keep=1)
                past=prefill.past_key_values;ids=prefill.logits[:,-1].argmax(-1,keepdim=True);del prefill
            fn=lambda:model(ids,past_key_values=past,use_cache=True,logits_to_keep=1)
            controller.enabled=False;ref=fn()
            ref_logits=ref.logits.clone();ref_hidden=ref.hidden_states.clone();del ref
            controller.enabled=True;actual=fn()
            assert torch.equal(actual.logits,ref_logits) and torch.equal(actual.hidden_states,ref_hidden),name
            assert bool(torch.isfinite(actual.logits).all()),name
            del actual,ref_logits,ref_hidden
            memory={}
            for variant in ['original','compact']:
                controller.enabled=variant=='compact';controller.workspaces.clear()
                for _ in range(a.warmup):fn()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                before=torch.cuda.memory_allocated();value=fn();torch.cuda.synchronize()
                memory[variant]=dict(baseline_bytes=before,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    workspace_bytes=sum(buffer.numel()*buffer.element_size() for ws in controller.workspaces.values() for buffer in ws.buffers.values()))
                del value
            samples={'original':[],'compact':[]}
            for iteration in range(a.rounds):
                order=['original','compact'] if iteration%2==0 else ['compact','original']
                for variant in order:
                    controller.enabled=variant=='compact'
                    dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
                    fn();torch.cuda.synchronize()
                    samples[variant].append((time.perf_counter()-start)*1000)
            row=dict(case=name,batch_per_rank=batch,tokens_per_rank=length,kv_context=context,
                logits_exact=True,hidden_exact=True,samples_ms=samples,memory=memory)
            records.append(row);print(name,rank,{k:median(v) for k,v in samples.items()},flush=True)
            del ids,past;controller.workspaces.clear();torch.cuda.empty_cache();dist.barrier()
    assert controller.fallbacks==0
    payload=dict(rank=rank,cases=records,optimized_calls=controller.calls,fallbacks=controller.fallbacks)
    (a.output/f'rank{rank}.json').write_text(json.dumps(payload,indent=2)+'\n')
    gathered=[None]*2;dist.all_gather_object(gathered,payload)
    if rank==0:
        summary=[]
        for index,(name,batch,length,context) in enumerate(CASES):
            rows=[x['cases'][index] for x in gathered]
            samples={variant:[max(pair) for pair in zip(*[row['samples_ms'][variant] for row in rows])] for variant in ['original','compact']}
            medians={k:median(v) for k,v in samples.items()}
            count=2*batch*length
            summary.append(dict(case=name,original_ms=medians['original'],compact_ms=medians['compact'],
                latency_reduction_pct=100*(1-medians['compact']/medians['original']),speedup=medians['original']/medians['compact'],
                original_tokens_per_second=count*1000/medians['original'],compact_tokens_per_second=count*1000/medians['compact'],
                samples_ms=samples,memory={variant:max(row['memory'][variant]['peak_allocated_bytes'] for row in rows) for variant in ['original','compact']},
                exact=True,batch_per_rank=batch,input_tokens_per_rank=length,kv_context=context))
        result=dict(experts=6,topk=2,ranks=2,checkpoint_sha256=sha256(a.checkpoint),torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),rounds=a.rounds,warmup=a.warmup,model_compiled=False,
            precision='FP32 model parameters loaded from FP16 export; BF16 autocast, original PyTorch experts',
            timing='Alternating paired wall-clock samples; CUDA synchronized; median of per-pair max across ranks; pre-barrier excluded',
            input='Deterministic sequences constructed from one Chinese text corpus, distinct offsets across rank/batch',
            scope='Full-model inference with last-position logits and KV cache; decode is one next-token step at fixed context 128',
            cases=summary,rank_fallbacks=[r['fallbacks'] for r in gathered],passed=True)
        (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--training-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='5,7')
    p.add_argument('--library')
    p.add_argument('--rounds',type=int,default=31)
    p.add_argument('--warmup',type=int,default=8)
    a=p.parse_args()
    assert a.rounds>=9 and a.warmup>=3
    worker(a) if 'RANK' in os.environ else launch(a)

if __name__=='__main__':main()
