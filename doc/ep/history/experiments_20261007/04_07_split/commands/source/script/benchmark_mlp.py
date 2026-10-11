"""Paired original/module and latest complete grouped MLP, Graph and eager separated."""
import argparse,importlib.util,json,statistics,sys,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'dev/pytorch'))
from grouped_gemm_split import load_library
from grouped_expert_mlp_split import GroupedExpertMLP
from kernel9_reference import graph_make,graph_time
load_library(a.library);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_stream(torch.cuda.Stream())
spec=importlib.util.spec_from_file_location('mlp_model',root.parent/'minimind/model/model_minimind.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
config=module.MiniMindConfig(hidden_size=768,moe_intermediate_size=2432,dropout=0.)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
experts={i:module.FeedForward(config,2432).cuda().eval() for i in [0,1]}
for i,e in experts.items():e.load_state_dict({f'{name}.weight':state[f'model.layers.7.mlp.experts.{i}.{name}.weight'].float() for name in ('gate_proj','up_proj','down_proj')})
mlp=GroupedExpertMLP(experts,[0,1]);rows=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for counts in ([0,1],[8,8],[64,73],[1480,1432]):
  torch.manual_seed(20261009);x=torch.randn((sum(counts),768),device='cuda',dtype=torch.bfloat16)*.1;gpu=torch.tensor(counts,device='cuda');mlp.prepare(x)
  def original():
   values=[];off=0
   for i,m in enumerate(counts):
    if m:values.append(experts[i](x[off:off+m]))
    off+=m
   return torch.cat(values)
  functions=[('original',original),('cuda_split',lambda:mlp(x,gpu))]
  graphs=[(name,graph_make(fn,32)) for name,fn in functions];samples={name:[] for name,_ in functions};wall={name:[] for name,_ in functions}
  for turn in range(7):
   for j in range(2):
    name,g=graphs[(turn+j)%2];samples[name].append(graph_time(g,32,10))
   for j in range(2):
    name,fn=functions[(turn+j)%2];torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(64):fn()
    torch.cuda.synchronize();wall[name].append((time.perf_counter()-start)*1e6/64)
  rows.append(dict(counts=counts,m=sum(counts),graph_us={k:statistics.median(v) for k,v in samples.items()},eager_wall_us={k:statistics.median(v) for k,v in wall.items()},graph_samples=samples,wall_samples=wall))
  graphs.clear();mlp.release_workspaces();(out/'benchmark.json').write_text(json.dumps(rows,indent=2));print('MLP',rows[-1]['m'],rows[-1]['graph_us'],rows[-1]['eager_wall_us'],flush=True)
print('ALL MLP BENCHMARKS COMPLETE',flush=True)
