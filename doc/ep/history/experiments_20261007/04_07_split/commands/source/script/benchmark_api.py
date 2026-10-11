"""Compare dispatcher/workspace cost with the same kernel's native benchmark bridge."""
import argparse,sys,json,time,statistics,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--native',required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.source/'script'));sys.path.insert(0,str(a.source/'dev/pytorch'))
from grouped_gemm_split import load_library,grouped_gemm,GemmWorkspace
from kernel9_reference import tensor_case,reference,validate,bind_ours,Ours,graph_make,graph_time
import ctypes
lib=ctypes.CDLL(str(Path(a.native).resolve()))
lib.s9_create.argtypes=[ctypes.c_int]*6+[ctypes.c_void_p]*4+[ctypes.c_int]*5;lib.s9_create.restype=ctypes.c_void_p
lib.s9_run.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int];lib.s9_run.restype=ctypes.c_int;lib.s9_destroy.argtypes=[ctypes.c_void_p]
class Native:
 def __init__(self,counts,x,b,out):
  self.ptr=lib.s9_create(int(len(x)<=256),int(x.dtype==torch.bfloat16),len(counts),len(x),b.shape[2],x.shape[1],counts.data_ptr(),x.data_ptr(),b.data_ptr(),out.data_ptr(),-1,-1,-1,0,0)
  assert self.ptr
 def run(self):assert lib.s9_run(self.ptr,torch.cuda.current_stream().cuda_stream,0)==0
 def close(self):lib.s9_destroy(self.ptr)
load_library(a.library);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
stream=torch.cuda.Stream();torch.cuda.set_stream(stream);rows=[]
with torch.inference_mode():
 for name,counts,k,n,dtype in [('decode_single_down',[0,1],2432,768,torch.bfloat16),
   ('decode_batch_gate_up',[8,8],768,2432,torch.bfloat16),('decode_137_gate_up',[64,73],768,2432,torch.bfloat16),('prefill_medium_down',[1480,1432],2432,768,torch.bfloat16),
   ('prefill_large_gate_up',[10986,11477],768,2432,torch.bfloat16),('regression_2048',[64,96,160,192]*4,1024,1024,torch.float16)]:
  x,b,out=tensor_case(counts,k,n,dtype);ct=torch.tensor(counts,device='cuda');ws=GemmWorkspace.allocate(x,b)
  native=Native(ct,x,b,out);expected=reference(x,b,counts)
  for y in [grouped_gemm(x,b,ct),ws.run(x,b,ct)]:validate(y,expected)
  native.run();torch.cuda.synchronize();validate(out,expected)
  functions=[('native_full',native.run),('api_out',lambda:ws.run(x,b,ct)),('api_functional',lambda:grouped_gemm(x,b,ct))]
  graphs=[(label,graph_make(fn,64)) for label,fn in functions];samples={label:[] for label,_ in graphs}
  for turn in range(7):
   for j in range(3):
    label,g=graphs[(turn+j)%3];samples[label].append(graph_time(g,64,10))
  eager={label:[] for label,_ in functions};host={label:[] for label,_ in functions}
  for turn in range(7):
   for j in range(3):
    label,fn=functions[(turn+j)%3]
    for _ in range(10):fn()
    torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    clock=time.perf_counter();start.record()
    for _ in range(64):fn()
    end.record();end.synchronize()
    host[label].append((time.perf_counter()-clock)*1e6/64)
    eager[label].append(start.elapsed_time(end)*1000/64)
  rows.append(dict(case=name,counts=counts,m=sum(counts),k=k,n=n,dtype=str(dtype),
                   graph_us={label:statistics.median(v) for label,v in samples.items()},
                   eager_event_us={label:statistics.median(v) for label,v in eager.items()},
                   eager_wall_us={label:statistics.median(v) for label,v in host.items()},
                   graph_samples=samples,eager_event_samples=eager,eager_wall_samples=host,
                   metadata_bytes=ws.metadata.numel(),partial_bytes=ws.partial.numel()*4))
  graphs.clear();native.close();print('RESULT',json.dumps(rows[-1]),flush=True)
  (a.output/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
record=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),seed=20261007,tf32=False,library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
            protocol='7 rotated rounds; graph 64 nodes and 10 replay; eager 64 calls per batch; preallocated hot cache; clocks unlocked',
            eager_caveat='CUDA event and wall include CPU submission gaps; event is not pure kernel time',
            no_model_measurement=True)
(a.output/'environment.json').write_text(json.dumps(record,indent=2)+'\n');print('ALL API BENCHMARK CHECKS PASSED')
