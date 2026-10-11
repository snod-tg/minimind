"""Paired two-rank wall timings: real EP/model paths, dynamic prepare and decode growth."""
import argparse,hashlib,json,os,statistics,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
p=argparse.ArgumentParser();p.add_argument('--stage6',type=Path,required=True);p.add_argument('--library',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--repo',type=Path,required=True)
p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--profile-only',action='store_true');p.add_argument('--steps',type=int,default=60)
p.add_argument('--workloads',default='prefill_small,prefill_medium,prefill_large,decode_single,decode_batch')
p.add_argument('--no-ep',action='store_true');p.add_argument('--no-growing',action='store_true')
a=p.parse_args();sys.path.insert(0,str(a.stage6));from ep_common import setup,inputs,write,WORKLOADS
rank,device,module,model,config=setup(a.stage6/'source',a.checkpoint,a.output)
from grouped_gemm_split import load_library
from minimind_ep_split_backend import MiniMindEPSplitBackend
load_library(a.library);controller=MiniMindEPSplitBackend(model)
torch.cuda.synchronize();clock=time.perf_counter();controller.initialize();torch.cuda.synchronize()
initialization_ms=(time.perf_counter()-clock)*1000
records=[];ep_records=[];decode_records=[];cold=[]
backends=['original','cuda_split','native_bf16','cuda_split_last_layer'];rounds=5;assert a.steps%rounds==0

def one(fn):
    dist.barrier();torch.cuda.synchronize(device)
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    clock=time.perf_counter();start.record();result=fn();end.record();end.synchronize()
    wall=(time.perf_counter()-clock)*1000;event=start.elapsed_time(end)
    del result
    return wall,event

def paired_samples(fn):
    rows={b:dict(wall=[],event=[],prepare_events=0) for b in backends}
    for b in backends:
        controller.backend=b
        for _ in range(20):fn()
        torch.cuda.synchronize()
    for turn in range(rounds):
        for j in range(len(backends)):
            b=backends[(turn+j)%len(backends)];controller.backend=b
            # Backend switching can change later-layer M. Warm each block before timing.
            for _ in range(3):fn()
            torch.cuda.synchronize();before=controller.prepare_events
            for _ in range(a.steps//rounds):
                wall,event=one(fn);rows[b]['wall'].append(wall);rows[b]['event'].append(event)
            rows[b]['prepare_events']+=controller.prepare_events-before
    return rows

def capture_ep_inputs(fn):
    values={};hooks=[]
    for i,layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_pre_hook(lambda m,v,i=i:values.__setitem__(i,v[0].detach().clone())))
    controller.backend='original';fn()
    for hook in hooks:hook.remove()
    return values

with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for topk in [1,2]:
        config.num_experts_per_tok=topk
        for name in a.workloads.split(','):
            case=f'{name}_top{topk}'
            if a.profile_only and case not in ['decode_batch_top1','prefill_medium_top1','prefill_small_top1']:continue
            controller.backend='original';x,cache,info=inputs(a.repo,rank,name,topk,model,a.baseline)
            fn=lambda:model(x,past_key_values=cache,use_cache=True)
            if a.profile_only:
                for backend in backends:
                    controller.backend=backend
                    for _ in range(10):fn()
                    torch.cuda.synchronize();dist.barrier();controller.profile=True
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(3):fn()
                        torch.cuda.synchronize()
                    controller.profile=False
                    prof.export_chrome_trace(str(a.output/f'{case}_{backend}_rank{rank}.json'))
                del x,cache;controller.release_workspaces();torch.cuda.empty_cache();continue
            # Warm GPU/runtime, then clear workspaces to measure a first prepared-cache forward.
            controller.backend='original'
            for _ in range(3):fn()
            for backend in backends:
                controller.release_workspaces();controller.backend=backend;before=controller.prepare_events
                wall,event=one(fn)
                cold.append(dict(case=case,rank=rank,backend=backend,wall_ms=wall,event_ms=event,
                                 prepare_events=controller.prepare_events-before,
                                 note='one first workspace-empty call; packed weights preinitialized; not a distribution'))
            rows=paired_samples(fn)
            records.append(dict(case=case,rank=rank,batch=WORKLOADS[name][0],seq_len=WORKLOADS[name][1],context=WORKLOADS[name][2],topk=topk,paths=rows))
            write(a.output/f'model_rank{rank}.json',records);write(a.output/f'cold_rank{rank}.json',cold)
            print('MODEL_TIMING',case,rank,{b:round(statistics.median(r['wall']),3) for b,r in rows.items()},flush=True)
            # Complete EP replay on the same original layer-7 input; not a sum of layer medians.
            if not a.no_ep:
                ep_x=capture_ep_inputs(fn)[7];mlp=model.model.layers[7].mlp
                ep_rows=paired_samples(lambda:mlp(ep_x))
                ep_records.append(dict(case=case,rank=rank,layer=7,shape=list(ep_x.shape),paths=ep_rows))
                write(a.output/f'ep_rank{rank}.json',ep_records);del ep_x
            del x,cache;controller.release_workspaces();torch.cuda.empty_cache();dist.barrier()
    if not a.profile_only and not a.no_growing:
        # Growing-KV greedy block timing includes 8 actual model calls + argmax/cache progression.
        from measure_baseline import generate_inputs
        for topk in [1,2]:
            config.num_experts_per_tok=topk
            for batch in [1,16]:
                prompt,_=generate_inputs(a.repo,rank,batch,128,topk);prompt=prompt.to(device)
                def fn():
                    controller.backend='original';initial=model(prompt,use_cache=True,logits_to_keep=1)
                    return initial
                original=fn();kv0=original.past_key_values;token0=original.logits[:,-1].argmax(-1,keepdim=True);del original
                def decode_block():
                    kv=kv0;token=token0
                    for _ in range(8):
                        out=model(token,past_key_values=kv,use_cache=True,logits_to_keep=1)
                        kv=out.past_key_values;token=out.logits[:,-1].argmax(-1,keepdim=True)
                    return token
                rows={b:dict(wall=[],event=[],prepare_events=0) for b in backends}
                for turn in range(5):
                    for j in range(len(backends)):
                        b=backends[(turn+j)%len(backends)];controller.backend=b;decode_block();torch.cuda.synchronize()
                        before=controller.prepare_events
                        for _ in range(4):
                            wall,event=one(decode_block);rows[b]['wall'].append(wall);rows[b]['event'].append(event)
                        rows[b]['prepare_events']+=controller.prepare_events-before
                decode_records.append(dict(rank=rank,topk=topk,batch=batch,steps=8,context_start=128,paths=rows))
                write(a.output/f'growing_decode_rank{rank}.json',decode_records)
                print('GROWING_DECODE',topk,batch,rank,{b:round(statistics.median(r['wall']),3) for b,r in rows.items()},flush=True)
                del prompt,kv0,token0;controller.release_workspaces();torch.cuda.empty_cache();dist.barrier()
write(a.output/f'summary_rank{rank}.json',dict(completed=True,rank=rank,profile_only=a.profile_only,model_cases=len(records),ep_cases=len(ep_records),
                                           growing_decode_cases=len(decode_records),cache_initialization_ms=initialization_ms,
                                           packed_bytes=sum(s.cache.nbytes for s in controller.states.values()),
                                           adapter_sha256=hashlib.sha256((a.stage6/'source/dev/pytorch/minimind_ep_split_backend.py').read_bytes()).hexdigest(),
                                           timing='60 calls/backend, 5 rotated blocks; barrier outside timed interval; paired slower-rank wall times',
                                           scope='model includes routing/count exchange/dispatch/experts/prepare/combine/attention/all logits/cache; excludes weights initialization and barrier',
                                           clocks='unlocked',cpu_affinity=sorted(os.sched_getaffinity(0))))
dist.barrier();dist.destroy_process_group();print('MODEL MEASUREMENT COMPLETE',rank,flush=True)
