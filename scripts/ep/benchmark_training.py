"""Matched DDP / original EP / compact EP short-training baseline.

All paths load the same FP32 random initialization and use the original
non-EP auxiliary objective. No trained model or optimizer is overwritten.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('ddp', 'ep_original', 'ep_legacy', 'ep_compact', 'ep_dedup')


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(a):
    import torch
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import setup_seed
    a.output.mkdir(parents=True, exist_ok=False)
    configuration = dict(hidden_size=768, num_hidden_layers=8, use_moe=True,
        num_experts=6, num_experts_per_tok=2, router_aux_loss_mode='local_topk_sum')
    for name, changes in [('initial', {}), ('correctness_initial', dict(hidden_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=128, moe_intermediate_size=128, max_position_embeddings=128))]:
        setup_seed(42)
        config = MiniMindConfig(**(configuration | changes))
        model = MiniMindForCausalLM(config)
        torch.save(model.state_dict(), a.output / f'{name}.pt')
        save_json(a.output / f'{name}_config.json', config.to_dict())
        del model
        gc.collect()
    save_json(a.output / 'protocol.json', dict(model=configuration, batch_per_rank=32,
        seq=340, accumulation=8, learning_rate=5e-4, grad_clip=1., ranks=2,
        precision='FP32 parameters and AdamW state; BF16 autocast; native expert GEMM',
        profiled=a.profile, compiled=False, warmup_updates=a.warmup_updates, measured_updates=a.updates,
        repeats=a.repeats, auxiliary_contract='original non-EP local Top-K sum',
        initial_sha256=digest(a.output / 'initial.pt'),
        timing='Synchronized full training window: data loading, forward, backward, gradient synchronization, clipping and AdamW. Excludes initialization, warmup and checkpoints.',
        tokens='Valid shifted-label tokens summed across both ranks; padded tokens reported separately',
        data_path=str(a.data.resolve()), data_stat=dict(size=a.data.stat().st_size,
            mtime_ns=a.data.stat().st_mtime_ns),
        source_hashes={p: digest(ROOT / p) for p in ('model/model_minimind.py',
            'model/moe_aux_loss.py', 'model/ep/routing.py','trainer/trainer_utils.py',
            'scripts/ep/benchmark_training.py')}))
    print('Prepared common FP32 initialization and protocol', flush=True)


def worker(a):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoTokenizer
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import setup_seed, sync_ep_gradients, clip_ep_grad_norm_
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device)
    assert dist.get_world_size() == 2
    setup_seed(42)
    is_ep = a.variant != 'ddp'
    name = 'correctness_initial' if a.stage == 'correctness' else 'initial'
    cfg_data = json.loads((a.output / f'{name}_config.json').read_text())
    cfg_data.update(use_ep=is_ep, ep_size=2 if is_ep else None, ep_comm_dtype='native')
    config = MiniMindConfig(**cfg_data)
    raw = MiniMindForCausalLM(config)
    full = torch.load(a.output / f'{name}.pt', map_location='cpu', mmap=True, weights_only=True)
    raw.load_state_dict({k: full[k] for k in raw.state_dict()}, strict=True)
    del full
    gc.collect()
    raw.to(device).train()
    controller = None
    if a.variant in ('ep_compact','ep_legacy','ep_dedup'):
        from model.ep_routing_backend import attach_ep_routing
        controller = attach_ep_routing(raw, library=a.library, mode={'ep_legacy':'compact_legacy','ep_compact':'compact','ep_dedup':'dedup_train'}[a.variant])
    model = raw if is_ep else DDP(raw, device_ids=[local])
    optimizer = torch.optim.AdamW(raw.parameters(), lr=5e-4)
    trial = a.output / f'{a.stage}_{a.repeat}_{a.variant}'
    trial.mkdir(exist_ok=True)
    if a.stage == 'correctness':
        # Nontrivial global batch with distinct samples and dynamic expert loads.
        generator = torch.Generator().manual_seed(120 + rank)
        inputs = torch.randint(3, config.vocab_size, (4, 2, 17), generator=generator).to(device)
        first_gradients = None
        losses, outputs = [], []
        for update in range(2):
            for micro in range(2):
                ids = inputs[update * 2 + micro]
                sync_context = model.no_sync() if not is_ep and micro < 1 else nullcontext()
                with sync_context:
                    result = model(ids, labels=ids)
                    loss = result.loss + result.aux_loss
                    (loss / 2).backward()
                losses.append([result.loss.item(), result.aux_loss.item()])
                if update == 0 and micro == 0:
                    outputs = [result.hidden_states.detach().cpu(), result.logits.detach().cpu()]
                del result, loss
            if is_ep:
                sync_ep_gradients(raw)
            if update == 0:
                first_gradients = {k: p.grad.detach().cpu() for k, p in raw.named_parameters() if p.grad is not None}
            norm = clip_ep_grad_norm_(raw, 1.) if is_ep else torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.)
            assert bool(torch.isfinite(norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.save(dict(outputs=outputs, losses=losses, gradients=first_gradients,
            parameters={k: p.detach().cpu() for k, p in raw.named_parameters()}), trial / f'rank{rank}.pt')
        save_json(trial / f'rank{rank}.json', dict(rank=rank, variant=a.variant,
            fallbacks=controller.fallbacks if controller else 0, passed=True))
    else:
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
        dataset = PretrainDataset(str(a.data), tokenizer, max_length=340)
        sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True, seed=42)
        sampler.set_epoch(a.repeat)
        batches = 8 * (a.warmup_updates + a.updates)
        indices = list(iter(sampler))[:32*batches]
        assert len(indices) == 32*batches
        loader = DataLoader(dataset, batch_size=32, sampler=indices, num_workers=8,
            pin_memory=True, drop_last=True, persistent_workers=True)
        iterator = iter(loader)
        records, losses, norms = [], [], []
        input_hash = hashlib.sha256()
        valid_tokens = 0
        allocated_init = torch.cuda.memory_allocated()
        job_start = time.perf_counter()
        for update in range(a.warmup_updates + a.updates):
            measured = update >= a.warmup_updates
            if update == a.warmup_updates:
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.reset_peak_memory_stats()
                window_start = time.perf_counter()
                if a.profile:
                    profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True)
                    profiler.start()
            update_start = time.perf_counter()
            count = 0
            for micro in range(8):
                cpu_ids, cpu_labels = next(iterator)
                if measured:
                    input_hash.update(cpu_ids.numpy().tobytes())
                    input_hash.update(cpu_labels.numpy().tobytes())
                    count += int((cpu_labels[:,1:] != -100).sum())
                ids, labels = cpu_ids.to(device), cpu_labels.to(device)
                sync_context = model.no_sync() if not is_ep and micro < 7 else nullcontext()
                with sync_context:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model(ids, labels=labels)
                        loss = result.loss + result.aux_loss
                    (loss / 8).backward()
                losses.append(torch.stack([result.loss.detach(), result.aux_loss.detach()]))
                del ids, labels, result, loss
            if is_ep:
                sync_ep_gradients(raw)
            norm = clip_ep_grad_norm_(raw,1.) if is_ep else torch.nn.utils.clip_grad_norm_(raw.parameters(),1.)
            norms.append(norm.detach())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Same synchronized update boundary in all three paths.
            torch.cuda.synchronize()
            seconds = time.perf_counter() - update_start
            row = dict(update=update + 1, measured=measured, seconds=seconds,
                valid_tokens=count, allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved())
            records.append(row)
            if measured:
                valid_tokens += count
            print(a.variant, a.repeat, rank, 'update',update+1,'seconds',round(seconds,3),flush=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - window_start
        job_seconds = time.perf_counter() - job_start
        if a.profile:
            profiler.stop()
            profiler.export_chrome_trace(str(trial/f'profile_rank{rank}.json'))
            save_json(trial/f'profile_summary_rank{rank}.json',[dict(key=e.key,count=e.count,cpu_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,device_us=e.device_time_total,self_device_us=e.self_device_time_total) for e in profiler.key_averages()])
        finite_losses = bool(torch.isfinite(torch.stack(losses)).all())
        finite_norms = bool(torch.isfinite(torch.stack(norms)).all())
        assert finite_losses and finite_norms
        assert controller is None or controller.fallbacks == 0
        payload = dict(rank=rank, variant=a.variant, repeat=a.repeat, seconds=seconds,
            job_seconds=job_seconds, valid_tokens=valid_tokens, padded_tokens=a.updates*8*32*340,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            allocated_initial=allocated_init, updates=records, input_sha256=input_hash.hexdigest(),
            finite_losses=finite_losses, finite_grad_norms=finite_norms,
            losses=torch.stack(losses).cpu().tolist(), gradient_norms=torch.stack(norms).cpu().tolist(),
            optimized_calls=controller.calls if controller else 0,
            fallbacks=controller.fallbacks if controller else 0,
            transport_rows=controller.transport_rows if controller else 0,original_transport_rows=controller.original_transport_rows if controller else 0,optimizer_parameters=len(optimizer.state), local_parameters=sum(p.numel() for p in raw.parameters()),
            precision='BF16 autocast', compiled=False)
        save_json(trial / f'rank{rank}.json',payload)
        gathered=[None]*2
        dist.all_gather_object(gathered,payload)
        if rank==0:
            elapsed=max(p['seconds'] for p in gathered)
            save_json(trial / 'summary.json',dict(variant=a.variant,repeat=a.repeat,
                seconds=elapsed,valid_tokens=sum(p['valid_tokens'] for p in gathered),
                valid_tokens_per_second=sum(p['valid_tokens'] for p in gathered)/elapsed,
                padded_tokens_per_second=sum(p['padded_tokens'] for p in gathered)/elapsed,
                optimizer_update_seconds=elapsed/a.updates,
                peak_allocated=max(p['peak_allocated'] for p in gathered),
                peak_reserved=max(p['peak_reserved'] for p in gathered),ranks=gathered))
    dist.barrier()
    dist.destroy_process_group()


def processes(uuids, expected_output=None, known_own=None):
    rows=[]
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader'],text=True)
    for line in query.splitlines():
        uuid,pid,memory=map(str.strip,line.split(','))
        if uuid not in uuids:
            continue
        try:command=(Path('/proc')/pid/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except OSError:command=''
        current='scripts.ep.benchmark_training' in command and (expected_output is None or str(expected_output) in command)
        exited=not command and known_own is not None and (uuid,int(pid)) in known_own
        rows.append(dict(uuid=uuid,pid=int(pid),memory=memory,own=current or exited,
            identity_source='command' if current else 'previously_observed' if exited else 'foreign'))
    return rows


def run_trial(a,stage,variant,repeat,selected):
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        '--module','scripts.ep.benchmark_training','--worker','--stage',stage,
        '--variant',variant,'--repeat',str(repeat),'--output',str(a.output),
        '--data',str(a.data),'--warmup-updates',str(a.warmup_updates),'--updates',str(a.updates)]
    if a.library:command+=['--library',str(a.library)]
    if a.profile:command+=['--profile']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',MINIMIND_EP_PROFILE='1' if a.profile else '0')
    print('START',stage,variant,repeat,flush=True)
    save_json(a.output/f'{stage}_{repeat}_{variant}_command.json',dict(command=command,cwd=str(ROOT)))
    observations=[];known_own=set()
    with (a.output/f'{stage}_{repeat}_{variant}.log').open('w') as log:
        start=time.perf_counter()
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while True:
            occupants=processes(selected,a.output,known_own)
            known_own.update((p['uuid'],p['pid']) for p in occupants if p['own'])
            observations.append(dict(time=time.time(),processes=occupants,
                inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,clocks.sm',
                    '--format=csv,noheader,nounits'],text=True)))
            if process.poll() is not None:break
            time.sleep(2)
    foreign=[p for row in observations for p in row['processes'] if not p['own']]
    save_json(a.output/f'{stage}_{repeat}_{variant}_monitor.json',dict(exit_code=process.returncode,
        foreign=foreign,observations=observations,process_seconds=time.perf_counter()-start))
    if process.returncode:
        print((a.output/f'{stage}_{repeat}_{variant}.log').read_text(errors='replace')[-7000:])
        raise RuntimeError(f'{stage} {variant} failed: {process.returncode}')
    assert not foreign, ('Foreign GPU work invalidates baseline',foreign)
    print('DONE',stage,variant,repeat,flush=True)


def compare_correctness(a):
    import torch
    rows=[]
    for rank in range(2):
        ref=torch.load(a.output/f'correctness_0_ddp/rank{rank}.pt',weights_only=True)
        for variant in a.variants[1:]:
            value=torch.load(a.output/f'correctness_0_{variant}/rank{rank}.pt',weights_only=True)
            metrics={}
            for kind in ('gradients','parameters'):
                errors=[]
                for name,actual in value[kind].items():
                    expected=ref[kind][name]
                    errors.append(float((actual-expected).abs().max()))
                    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=5e-4)
                metrics[kind+'_max_abs']=max(errors)
            for actual,expected in zip(value['outputs'],ref['outputs']):
                torch.testing.assert_close(actual,expected,atol=2e-5,rtol=5e-4)
            torch.testing.assert_close(torch.tensor(value['losses']),torch.tensor(ref['losses']),atol=2e-5,rtol=5e-4)
            rows.append(dict(rank=rank,variant=variant,**metrics,passed=True))
    save_json(a.output/'correctness_summary.json',dict(rows=rows,passed=True,
        scope='FP32 small two-layer model, six experts Top-2, distinct per-rank input; first synchronized gradients and parameters after two AdamW updates',
        atol=2e-5,rtol=5e-4))
    print('CORRECTNESS PASSED',flush=True)


def summarize(a):
    trials={variant:[json.loads((a.output/f'performance_{i}_{variant}/summary.json').read_text())
        for i in range(a.repeats)] for variant in a.variants}
    for repeat in range(a.repeats):
        for rank in range(2):
            hashes={trials[v][repeat]['ranks'][rank]['input_sha256'] for v in a.variants}
            assert len(hashes)==1,'Data differs across paths'
    results=[]
    for variant,values in trials.items():
        results.append(dict(variant=variant,
            valid_tokens_per_second=statistics.median(p['valid_tokens_per_second'] for p in values),
            padded_tokens_per_second=statistics.median(p['padded_tokens_per_second'] for p in values),
            optimizer_update_seconds=statistics.median(p['optimizer_update_seconds'] for p in values),
            window_seconds=statistics.median(p['seconds'] for p in values),
            peak_allocated=max(p['peak_allocated'] for p in values),
            peak_reserved=max(p['peak_reserved'] for p in values),
            trials=[{k:v for k,v in p.items() if k!='ranks'} for p in values]))
    base=results[0]
    for row in results:
        row['speedup_vs_ddp']=row['valid_tokens_per_second']/base['valid_tokens_per_second']
        row['allocated_reduction_pct']=100*(1-row['peak_allocated']/base['peak_allocated'])
    save_json(a.output/'summary.json',dict(results=results,passed=True,occupancy_clean=True,
        physical_gpus=a.gpus,repeats=a.repeats,measured_updates=a.updates,warmup_updates=a.warmup_updates,
        profiled=a.profile, compiled=False,initial_sha256=digest(a.output/'initial.pt'),
        auxiliary_contract='Original non-EP local Top-K sum',
        limitation='Short training-window timings; excludes initialization/warmup/checkpoints and does not measure a complete two-epoch run or validate model quality'))
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


def launch(a):
    from scripts.ep.train import validate_gpus
    ids=validate_gpus(a.gpus)
    assert not a.output.exists(), 'Use a new output directory'
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    selected={}
    for line in inventory.splitlines():
        index,uuid,memory,util=map(str.strip,line.split(','))
        if int(index) in ids:
            assert int(memory)<100 and int(util)==0, ('GPU occupied',line)
            selected[uuid]=int(index)
    assert len(selected)==2 and not processes(selected)
    prepare(a)
    save_json(a.output/'gpu_selection.json',dict(selected=selected,inventory=inventory))
    try:
        for variant in a.variants:
            run_trial(a,'correctness',variant,0,selected)
        compare_correctness(a)
        for repeat in range(a.repeats):
            order=tuple(a.variants[repeat%len(a.variants):]+a.variants[:repeat%len(a.variants)])
            for variant in order:
                assert not processes(selected),'Selected GPUs became occupied'
                run_trial(a,'performance',variant,repeat,selected)
        summarize(a)
    except Exception as exc:
        save_json(a.output/'failure.json',dict(error=str(exc)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'dataset/pretrain_t2t_mini.jsonl')
    p.add_argument('--gpus',default='5,7')
    p.add_argument('--library')
    p.add_argument('--warmup-updates',type=int,default=4)
    p.add_argument('--updates',type=int,default=8)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['ddp','ep_original','ep_compact'])
    p.add_argument('--profile',action='store_true')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--stage',choices=['correctness','performance'])
    p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--repeat',type=int,default=0)
    a=p.parse_args()
    a.output=a.output.resolve();a.data=a.data.resolve()
    assert a.variants[0]=='ddp'
    assert a.warmup_updates>=2 and a.updates>=4 and a.repeats>=3
    worker(a) if a.worker else launch(a)


if __name__=='__main__':
    main()
