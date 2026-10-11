"""Shared model/input construction, metrics and checkpoint identity for stages 6/7."""
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import torch
import torch.distributed as dist

WORKLOADS=dict(prefill_small=(1,128,0),prefill_medium=(8,340,0),prefill_large=(32,340,0),
               decode_single=(1,1,128),decode_batch=(16,1,128))

def setup(source,checkpoint,output):
    rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    if os.environ.get('BENCH_CPU_BASE'):os.sched_setaffinity(0,{int(os.environ['BENCH_CPU_BASE'])+rank})
    torch.cuda.set_device(device);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',device_id=device);assert dist.get_world_size()==2
    sys.path.insert(0,str(source/'dev/pytorch'));sys.path.insert(0,str(source/'script'))
    spec=importlib.util.spec_from_file_location('ep_candidate_model',source/'script/model_minimind_snapshot.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    config=module.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True,use_ep=True,ep_size=2,
                                num_experts=4,num_experts_per_tok=1,moe_intermediate_size=2432,dropout=0.,
                                ep_comm_dtype='native',max_position_embeddings=4096)
    model=module.MiniMindForCausalLM(config)
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict({k:weights[k] for k in model.state_dict()},strict=True);del weights
    model=model.to(device).eval();output.mkdir(parents=True,exist_ok=True)
    env=dict(rank=rank,physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(device),
             python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),tf32=False,
             pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),
             checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
             model_source_sha256=hashlib.sha256((source/'script/model_minimind_snapshot.py').read_bytes()).hexdigest())
    (output/f'environment_rank{rank}.json').write_text(json.dumps(env,indent=2)+'\n')
    return rank,device,module,model,config

def inputs(repo,rank,name,topk,model,baseline):
    from measure_baseline import generate_inputs
    batch,length,context=WORKLOADS[name]
    cpu,info=generate_inputs(repo,rank,batch,context or length,topk);x=cpu.cuda();cache=None
    if context:
        initial=model(x,use_cache=True);cache=initial.past_key_values;x=initial.logits[:,-1].argmax(-1,keepdim=True)
    saved=torch.load(baseline/f'{name}_top{topk}'/f'capture_rank{rank}.pt',map_location='cpu',weights_only=True)
    assert torch.equal(x.cpu(),saved['input_ids']),('changed baseline inputs',name,topk,rank)
    return x,cache,info

def metrics(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype,(actual.shape,expected.shape,actual.dtype,expected.dtype)
    # Chunk full-logit comparisons to bound temporary storage on large Prefill.
    x=actual.reshape(-1);y=expected.to(actual.device).reshape(-1)
    failures=0;squared=0.;norm=0.;maximum=0.;finite=True;exact=True
    for start in range(0,len(x),2**20):
        xa=x[start:start+2**20].float();ya=y[start:start+2**20].float();d=xa-ya
        failures+=int((d.abs()>.02+.02*ya.abs()).sum());squared+=float(d.square().double().sum());norm+=float(ya.square().double().sum())
        maximum=max(maximum,float(d.abs().max()) if len(d) else 0.);finite=finite and bool(torch.isfinite(xa).all());exact=exact and torch.equal(xa,ya)
    relative=(squared/norm)**.5 if norm else squared**.5
    assert finite,'nonfinite output'
    return dict(elements=len(x),element_failures=failures,relative_rms=relative,relative_rms_pass=relative<=.005,
                strict_pass=failures==0 and relative<=.005,max_abs=maximum,exact=exact,finite=finite)

def route_metrics(actual,expected):
    # sorted=False topk order is not a semantic expert change.
    aa=actual['ids'].sort(dim=-1).values;bb=expected['ids'].sort(dim=-1).values
    return dict(tokens=len(aa),changed_tokens=int((aa!=bb).any(dim=-1).sum()),ordered_exact=torch.equal(actual['ids'],expected['ids']))

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
