"""True checkpoint/input MLP replay; isolate cache/layout/grouping and interface cost."""
import argparse,sys,json,time,statistics,importlib.util,copy,hashlib,os
from pathlib import Path
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--library',required=True)
p.add_argument('--baseline',type=Path,required=True);p.add_argument('--extra',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--validate-only',action='store_true');p.add_argument('--benchmark-only',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.source/'dev/pytorch'));sys.path.insert(0,str(a.source/'script'))
from grouped_gemm_api import load_library
from grouped_expert_mlp import GroupedExpertMLP
from kernel9_reference import reference,validate,cpu_double_check,graph_make,graph_time
load_library(a.library);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
stream=torch.cuda.Stream();torch.cuda.set_stream(stream);device=torch.device('cuda:0')
model_source=a.source/'script/model_minimind_snapshot.py'
spec=importlib.util.spec_from_file_location('mlp_replay_model',model_source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
config=module.MiniMindConfig(hidden_size=768,moe_intermediate_size=2432,dropout=0.)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
fixtures=[]
for root in [a.baseline,a.extra]:
 for path in sorted(root.glob('*/capture_rank*.pt')):
  data=torch.load(path,map_location='cpu',weights_only=True)
  for row in data['replays']:fixtures.append(dict(case=path.parent.name,**row))
assert len(fixtures)==100,len(fixtures)
contexts={};packing=[]

def context(layer,ids):
 key=(layer,tuple(ids))
 if key in contexts:return contexts[key]
 with torch.inference_mode(False):
  experts={i:module.FeedForward(config,intermediate_size=2432).to(device).eval() for i in ids}
  for i,e in experts.items():e.load_state_dict({f'{name}.weight':state[f'model.layers.{layer}.mlp.experts.{i}.{name}.weight'].float() for name in ['gate_proj','up_proj','down_proj']},strict=True)
 cached={i:copy.deepcopy(e).to(dtype=torch.bfloat16).eval() for i,e in experts.items()}
 torch.cuda.synchronize();clock=time.perf_counter();alloc=GroupedExpertMLP(experts,ids,device=device);torch.cuda.synchronize()
 pack_us=(time.perf_counter()-clock)*1e6
 fast=GroupedExpertMLP(experts,ids,device=device,preallocate=True,cache=alloc.cache)
 for name in ['gate_proj','up_proj','down_proj']:
  for j,i in enumerate(ids):assert torch.equal(alloc.cache.get(name)[j].T,getattr(cached[i],name).weight)
 value=dict(experts=experts,cached=cached,alloc=alloc,fast=fast,ids=ids)
 contexts[key]=value;packing.append(dict(layer=layer,expert_ids=ids,pack_us=pack_us,packed_bytes=alloc.cache.nbytes));return value

def original(x,counts,ctx,cached=False):
 parts=[];offset=0;experts=ctx['cached'] if cached else ctx['experts']
 for i,m in zip(ctx['ids'],counts):
  if m:parts.append(experts[i](x[offset:offset+m]))
  offset+=m
 return torch.cat(parts) if parts else x.clone()

def metrics(y,expected):
 torch.testing.assert_close(y,expected,atol=.02,rtol=.02)
 diff=y.float()-expected.float();rms=float(diff.square().mean().sqrt()) if diff.numel() else 0.
 norm=float(expected.float().square().mean().sqrt()) if expected.numel() else 0.
 rr=rms/norm if norm else rms;assert rr<=.005,(rr,.005)
 return dict(exact=torch.equal(y,expected),max_abs=float(diff.abs().max()) if diff.numel() else 0.,relative_rms=rr)

validation=[];benchmarks=[]
selected={('decode_single_top1',0,7),('decode_single_top2',0,0),('decode_batch_top1',0,0),('decode_batch_top2',0,0),
          ('prefill_small_top1',0,0),('prefill_medium_top1',0,0),('prefill_medium_top2',0,0),
          ('prefill_large_top1',0,0),('prefill_large_top2',0,5)}
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 if not a.benchmark_only:
  for r in fixtures:
   x=r['x'].to(device);expected=r['y'].to(device);counts=r['counts'];ct=torch.tensor(counts,device=device,dtype=torch.int64);ctx=context(r['layer'],r['local_expert_ids'])
   y=original(x,counts,ctx);assert torch.equal(y,expected),(r['case'],r['rank'],r['layer'])
   cached=original(x,counts,ctx,True);cached_error=metrics(cached,expected)
   func=ctx['alloc'](x,ct);func_error=metrics(func,expected)
   ctx['fast'].prepare(x);reuse=ctx['fast'](x,ct);reuse_error=metrics(reuse,expected)
   assert torch.equal(func,reuse)
   gate,up,down=ctx['fast'].workspaces;hidden=F.silu(gate.out)*up.out
   projections=[]
   with torch.autocast('cuda',enabled=False):
    for name,operand,out in [('gate_proj',x,gate.out),('up_proj',x,up.out),('down_proj',hidden,down.out)]:
     b=ctx['alloc'].cache.get(name);ref=reference(operand,b,counts)
     projections.append(dict(name=name,**validate(out,ref),cpu_samples=cpu_double_check(operand,b,out,counts,17)))
   validation.append(dict(case=r['case'],rank=r['rank'],layer=r['layer'],m=len(x),counts=counts,original_exact=True,
                          cached_error=cached_error,functional_error=func_error,preallocated_error=reuse_error,projections=projections))
   print('VALIDATED',r['case'],r['rank'],r['layer'],len(x),reuse_error,flush=True)
   (a.output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
   ctx['fast'].workspaces=None;ctx['fast'].shape=None
   del x,expected,y,cached,func,reuse,hidden,gate,up,down,ref
 if not a.validate_only:
  for r in fixtures:
   key=(r['case'],r['rank'],r['layer'])
   if key not in selected:continue
   x=r['x'].to(device);expected=r['y'].to(device);counts=r['counts'];ct=torch.tensor(counts,device=device,dtype=torch.int64);ctx=context(r['layer'],r['local_expert_ids']);ctx['fast'].prepare(x)
   functions=[('original_fp32_autocast',lambda:original(x,counts,ctx)),('per_expert_bf16_cached',lambda:original(x,counts,ctx,True)),
              ('grouped_allocating',lambda:ctx['alloc'](x,ct)),('grouped_preallocated',lambda:ctx['fast'](x,ct))]
   for _,fn in functions:metrics(fn(),expected)
   graphs=[(label,graph_make(fn,16)) for label,fn in functions];graph_samples={label:[] for label,_ in functions}
   eager_samples={label:[] for label,_ in functions};event_samples={label:[] for label,_ in functions}
   for turn in range(7):
    for j in range(4):
     label,g=graphs[(turn+j)%4];graph_samples[label].append(graph_time(g,16,5))
   for turn in range(7):
    for j in range(4):
     label,fn=functions[(turn+j)%4]
     for _ in range(10):fn()
     torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     clock=time.perf_counter();start.record()
     for _ in range(32):fn()
     end.record();end.synchronize();eager_samples[label].append((time.perf_counter()-clock)*1e6/32);event_samples[label].append(start.elapsed_time(end)*1000/32)
   row=dict(case=r['case'],rank=r['rank'],layer=r['layer'],m=len(x),counts=counts,
            graph_us={k:statistics.median(v) for k,v in graph_samples.items()},eager_wall_us={k:statistics.median(v) for k,v in eager_samples.items()},
            eager_event_us={k:statistics.median(v) for k,v in event_samples.items()},graph_samples=graph_samples,eager_samples=eager_samples,event_samples=event_samples)
   benchmarks.append(row);(a.output/'benchmark.json').write_text(json.dumps(benchmarks,indent=2)+'\n');print('RESULT',json.dumps(row),flush=True)
   # Profile only two small-M cases, after timing; five complete invocations each.
   if r['case'] in ['decode_single_top1','decode_batch_top1']:
    diagnostic={}
    for label,fn in functions:
     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
      for _ in range(5):fn()
     prof.export_chrome_trace(str(a.output/f'{r["case"]}_{label}.json'))
     diagnostic[label]=[dict(name=e.key,count=e.count,cpu_us=e.cpu_time_total) for e in prof.key_averages()]
    (a.output/f'{r["case"]}_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
   graphs.clear();ctx['fast'].workspaces=None;ctx['fast'].shape=None
   del x,expected,ct
  assert len(benchmarks)==9
(a.output/'packing.json').write_text(json.dumps(packing,indent=2)+'\n')
env=dict(passed=True,torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(),tf32=False,
         checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),model_source_sha256=hashlib.sha256(model_source.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
         mlp_source_sha256=hashlib.sha256((a.source/'dev/pytorch/grouped_expert_mlp.py').read_bytes()).hexdigest(),
         timing='Graph 16 calls, 7 rotated rounds, 5 replay; eager 32 calls/batch, 7 rotated rounds; hot cache, unlocked clocks; one autocast context across calls',
         scope='standalone local expert MLP only; excludes router, sorting, communication, counts creation, H2D, cache initialization, workspace prepare')
(a.output/'environment.json').write_text(json.dumps(env,indent=2)+'\n');print('ALL MLP CHECKS PASSED',flush=True)
