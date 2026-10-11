"""Validate variable expert counts, then compare with tuned CUTLASS on identical inputs."""
import ctypes,json,statistics,time,os,random
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('KERNEL8_RESULT',ROOT/'result/20261006/moe_grouped_kernel8'));OUT.mkdir(parents=True,exist_ok=True)
OPEN=Path(os.environ.get('MOE_OPEN_BASELINES','/data/user4/tongleiwen/moe-open-baselines'))
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
SM=torch.cuda.get_device_properties(0).multi_processor_count
own=ctypes.CDLL(str(OUT/'kernel8_bridge.so'))
own.k8_create.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int]+[ctypes.c_void_p]*3;own.k8_create.restype=ctypes.c_void_p
own.k8_run.argtypes=[ctypes.c_void_p]*2;own.k8_run.restype=ctypes.c_int
own.k8_destroy.argtypes=[ctypes.c_void_p];own.k8_error.restype=ctypes.c_char_p
libs={name:ctypes.CDLL(str(OPEN/file)) for name,file in [('device','bridge.so'),('host','bridge_host.so')]}
for lib in libs.values():
 lib.gmm_create.argtypes=[ctypes.c_int]*3+[ctypes.POINTER(ctypes.c_int)]+[ctypes.c_int]*3+[ctypes.c_void_p]*3;lib.gmm_create.restype=ctypes.c_void_p
 lib.gmm_run.argtypes=[ctypes.c_void_p]*2;lib.gmm_run.restype=ctypes.c_int
 lib.gmm_destroy.argtypes=[ctypes.c_void_p];lib.gmm_error.restype=ctypes.c_char_p
configs=json.loads((OPEN/'cutlass_configs.json').read_text())
class Ours:
 def __init__(self,version,counts,a,b,c):
  ct=(ctypes.c_int*16)(*counts)
  start=time.perf_counter();self.ptr=own.k8_create(version,ct,16,a.data_ptr(),b.data_ptr(),c.data_ptr());self.setup_us=(time.perf_counter()-start)*1e6
  if not self.ptr:raise RuntimeError(own.k8_error().decode())
 def __call__(self):
  if own.k8_run(self.ptr,stream.cuda_stream):raise RuntimeError(own.k8_error().decode())
 def close(self):
  if self.ptr:own.k8_destroy(self.ptr);self.ptr=None
class Cutlass:
 def __init__(self,mode,cfg,blocks,counts,a,b,c):
  self.lib=libs[mode];ct=(ctypes.c_int*16)(*counts)
  self.ptr=self.lib.gmm_create(0,cfg,blocks,ct,16,1024,1024,a.data_ptr(),b.data_ptr(),c.data_ptr())
  if not self.ptr:raise RuntimeError(self.lib.gmm_error().decode())
 def __call__(self):
  if self.lib.gmm_run(self.ptr,stream.cuda_stream):raise RuntimeError(self.lib.gmm_error().decode())
 def close(self):
  if self.ptr:self.lib.gmm_destroy(self.ptr);self.ptr=None

def reference(a,b,counts,out):
 offset=0
 for e,m in enumerate(counts):
  if m:torch.mm(a[offset:offset+m],b[e],out=out[offset:offset+m])
  offset+=m

def valid(c,ref):
 delta=(c.float()-ref.float()).abs();bad=(~torch.isfinite(c))|(delta>2e-4+1e-3*ref.float().abs())
 if bad.any():raise AssertionError(f'{bad.sum().item()} mismatches; max abs {delta.max().item()}')
 return delta.max().item()

def cpu_check(a,b,c,counts):
 ha=a.cpu().double();hb=b.cpu().double();offset=0
 for e,m in enumerate(counts):
  if m:
   idx=torch.linspace(0,m*1024-1,257,dtype=torch.int64);rows=idx//1024;cols=idx%1024
   ref=(ha[offset+rows]*hb[e,:,cols].T).sum(1).half().cuda()
   valid(c[offset+rows.cuda(),cols.cuda()],ref)
  offset+=m

def graph_make(fn,nodes):
 for _ in range(3):fn()
 torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=stream):
  for _ in range(nodes):fn()
 return graph

def graph_time(graph,nodes,rounds=3,replays=5):
 for _ in range(3):graph.replay()
 start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);samples=[]
 for _ in range(rounds):
  start.record()
  for _ in range(replays):graph.replay()
  end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000/(nodes*replays))
 return statistics.median(samples)

tails=[0,1,2,7,8,15,16,17,31,32,33,63,64,65,129,0];tails[-1]=2048-sum(tails)
rng=random.Random(43);cuts=sorted([0,2048]+[rng.randrange(2049) for _ in range(15)]);random_counts=[cuts[i+1]-cuts[i] for i in range(16)]
scenarios=[('original',[64,96,160,192]*4),('balanced',[128]*16),('odd_rows',[127]*15+[143]),('tails',tails),('skewed',[1536]+[35]*2+[34]*13),('single',[2048]+[0]*15),('random',random_counts)]
validation=[];results=[];tuning=[]
for name,counts in scenarios:
 assert len(counts)==16 and sum(counts)==2048
 torch.manual_seed(0);a=torch.empty((2048,1024),device='cuda',dtype=torch.float16);b=torch.empty((16,1024,1024),device='cuda',dtype=torch.float16);c=torch.empty_like(a);ref=torch.empty_like(c)
 torch.cuda.synchronize();runner=Ours(8,counts,a,b,c)
 for pattern in ('random0','random17','random43','zero','expert_constant','single_k'):
  if pattern.startswith('random'):
   torch.manual_seed(int(pattern[6:]));a.copy_((torch.rand_like(a,dtype=torch.float32)-.5).half());b.copy_((torch.rand_like(b,dtype=torch.float32)-.5).half())
  elif pattern=='zero':a.zero_();b.fill_(1)
  elif pattern=='expert_constant':
   a.fill_(1)
   for e in range(16):b[e].fill_((e+1)/1024)
  else:
   a.zero_();b.zero_();a[:,37]=1
   for e in range(16):b[e,37,:]=torch.arange(1024,device='cuda',dtype=torch.float32)/1024+(e+1)/16
  c.fill_(float('nan'));reference(a,b,counts,ref);runner();torch.cuda.synchronize();err=valid(c,ref);cpu_check(a,b,c,counts)
  validation.append(dict(scenario=name,pattern=pattern,counts=counts,max_abs_error=err))
 print('VALIDATED',name,flush=True)
 # Identical random inputs for every candidate; all allocations and metadata precede timing.
 torch.manual_seed(20261006);a.copy_((torch.rand_like(a,dtype=torch.float32)-.5).half());b.copy_((torch.rand_like(b,dtype=torch.float32)-.5).half());reference(a,b,counts,ref);torch.cuda.synchronize()
 candidates=[]
 for mode in libs:
  for cfg in configs:
   for blocks in (0,SM,2*SM):
    candidate=None;graph=None
    try:
     candidate=Cutlass(mode,cfg['id'],blocks,counts,a,b,c);c.fill_(float('nan'));candidate();torch.cuda.synchronize();valid(c,ref)
     graph=graph_make(candidate,16);us=graph_time(graph,16)
     row=dict(scenario=name,mode=mode,cfg=cfg,blocks=blocks,us=us);candidates.append(row);tuning.append(row)
    except Exception as ex:raise RuntimeError(f'CUTLASS {name} {mode} {cfg} {blocks}: {ex}') from ex
    finally:
     if graph is not None:del graph
     if candidate is not None:candidate.close()
 best=None
 for row in sorted(candidates,key=lambda r:r['us'])[:3]:
  candidate=Cutlass(row['mode'],row['cfg']['id'],row['blocks'],counts,a,b,c);graph=graph_make(candidate,64);us=graph_time(graph,64,rounds=3,replays=5);del graph;candidate.close()
  if best is None or us<best[0]:best=(us,row)
 selected=best[1];cutlass=Cutlass(selected['mode'],selected['cfg']['id'],selected['blocks'],counts,a,b,c)
 ops=[('kernel8',runner),('CUTLASS',cutlass)];old=None
 if name=='original':old=Ours(7,counts,a,b,c);ops.append(('kernel7',old))
 graphs=[]
 for label,op in ops:
  c.fill_(float('nan'));op();torch.cuda.synchronize();valid(c,ref);cpu_check(a,b,c,counts);graphs.append((label,graph_make(op,128)))
 times={label:[] for label,_ in graphs}
 for round in range(7):
  for j in range(len(graphs)):
   label,graph=graphs[(j+round)%len(graphs)];times[label].append(graph_time(graph,128,rounds=1,replays=10))
 medians={label:statistics.median(t) for label,t in times.items()}
 row=dict(scenario=name,counts=counts,tile_count=sum((m+31)//32*8 for m in counts),medians_us=medians,rounds_us=times,relative_throughput_pct=medians['CUTLASS']/medians['kernel8']*100,cutlass=selected,metadata_setup_us=runner.setup_us)
 results.append(row);print('RESULT',json.dumps(row),flush=True)
 del graphs,graph
 cutlass.close();runner.close()
 if old is not None:old.close()
 del a,b,c,ref
 (OUT/'validation.json').write_text(json.dumps(validation,indent=2));(OUT/'results.json').write_text(json.dumps(results,indent=2));(OUT/'tuning.json').write_text(json.dumps(tuning,indent=2))
 print('COMPLETE_SCENARIO',name,flush=True)
(OUT/'environment.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,cuda=torch.version.cuda,SM=SM,cutlass_configs=len(configs),cutlass_blocks=[0,SM,2*SM],graph_nodes=128,rounds=7,replays=10),indent=2))
print('ALL VALIDATION AND BENCHMARKS PASSED',flush=True)
