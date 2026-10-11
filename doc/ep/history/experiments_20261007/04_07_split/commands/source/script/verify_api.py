"""Exercise registration, numerics, stream/device, graph, checks and explicit cache."""
import argparse,json,sys,importlib.util,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.source/'script'));sys.path.insert(0,str(a.source/'dev/pytorch'))
from grouped_gemm_split import load_library,grouped_gemm,GemmWorkspace,PackedExpertWeights
from kernel9_reference import tensor_case,reference,validate,cpu_double_check
load_library(a.library);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
records=[];errors=[];stream=torch.cuda.Stream();torch.cuda.set_stream(stream)

def record(name,**details):records.append(dict(test=name,**details));print('PASSED',name,flush=True)
def rejected(name,fn):
 try:fn()
 except (RuntimeError,ValueError,NotImplementedError) as e:errors.append(dict(test=name,error=str(e)));print('REJECTED',name,flush=True)
 else:raise AssertionError(f'{name} unexpectedly accepted')

with torch.inference_mode():
 patterns=[[0],[1],[0,1],[31,33],[0,129],[1480,1432],[0,32769],[32768,32768],
           [0]*16,[1]*16,[64,96,160,192]*4,[0,1,31,33,129,256,0,511]+[0]*7+[3136]]
 for dtype in [torch.float16,torch.bfloat16]:
  for k,n in [(768,2432),(2432,768),(1024,1024)]:
   for counts in patterns:
    x,b,_=tensor_case(counts,k,n,dtype);ct=torch.tensor(counts,device=x.device,dtype=torch.int64)
    expected=reference(x,b,counts);ws=GemmWorkspace.allocate(x,b)
    y=grouped_gemm(x,b,ct,debug_counts=True);stats=validate(y,expected)
    z=ws.run(x,b,ct);torch.cuda.synchronize();validate(z,expected)
    assert not y.numel() or (y.data_ptr()!=x.data_ptr() and y.data_ptr()!=z.data_ptr())
    assert y.shape==(sum(counts),n) and y.dtype==dtype
    sampled=cpu_double_check(x,b,z,counts) if sum(counts) in [1,64,129] else 0
    record('numerical',m=sum(counts),counts=counts,k=k,n=n,dtype=str(dtype),**stats,cpu_samples=sampled)
    del x,b,y,z,expected,ws
 # Live counts, functional/out Graph capture, small split and regular.
 for m,k,n in [(16,2432,768),(65,768,2432)]:
  x,b,_=tensor_case([m,0],k,n,torch.bfloat16);ct=torch.tensor([m,0],device='cuda');ws=GemmWorkspace.allocate(x,b)
  for functional in [False,True]:
   fn=lambda:grouped_gemm(x,b,ct) if functional else ws.run(x,b,ct)
   for _ in range(3):fn()
   torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream):y=fn()
   for counts in [[m,0],[0,m],[1,m-1],[m//2,m-m//2]]:
    ct.copy_(torch.tensor(counts,device='cuda'));graph.replay();validate(y,reference(x,b,counts))
    record('graph_live_counts',m=m,k=k,n=n,functional=functional,counts=counts)
   del graph
 # Device guard: input on device0 while ambient current device1; guard must restore device1.
 if torch.cuda.device_count()>1:
  x,b,_=tensor_case([1,0],2432,768,torch.bfloat16);ct=torch.tensor([1,0],device='cuda')
  with torch.cuda.device(1):
   y=grouped_gemm(x,b,ct);assert torch.cuda.current_device()==1
  validate(y,reference(x,b,[1,0]));record('device_guard_restored',input_device=0,ambient_device=1)
  x1,b1,_=tensor_case([1],768,2432,torch.bfloat16)
  rejected('mixed_device',lambda:grouped_gemm(x,b1.to('cuda:1'),ct))
 # Cross-stream dependency explicitly established; allocator stays current-stream owned.
 with torch.cuda.stream(torch.cuda.Stream()):
  other=torch.cuda.current_stream();x,b,_=tensor_case([31,33],2432,768,torch.bfloat16);ct=torch.tensor([31,33],device='cuda')
 stream.wait_stream(other);y=grouped_gemm(x,b,ct);validate(y,reference(x,b,[31,33]));record('nondefault_stream_dependency')
 # Schema/fake/autograd registration + dynamic shape compile through eager backend.
 x,b,_=tensor_case([31,33],2432,768,torch.bfloat16);ct=torch.tensor([31,33],device='cuda')
 check=torch.library.opcheck(torch.ops.my_cuda_kernels_split.prefill.default,(x,b,ct))
 record('opcheck',results=check)
 out_ws=GemmWorkspace.allocate(x,b)
 out_check=torch.library.opcheck(torch.ops.my_cuda_kernels_split.prefill_out.default,(x,b,ct,out_ws.out,out_ws.metadata,out_ws.partial),test_utils=('test_schema','test_faketensor'))
 record('out_opcheck',results=out_check)
 compiled=torch.compile(lambda a,b,c:torch.ops.my_cuda_kernels_split.prefill(a,b,c),backend='eager',fullgraph=True,dynamic=True)
 for m in [1,65,129]:
  xx,bb,_=tensor_case([m,0],2432,768,torch.bfloat16);cc=torch.tensor([m,0],device='cuda')
  validate(compiled(xx,bb,cc),reference(xx,bb,[m,0]));record('compile_eager_dynamic',m=m)
 # Host-known invalid properties and explicit debug counts checks.
 x,b,_=tensor_case([31,33],1024,1024,torch.bfloat16);ct=torch.tensor([31,33],device='cuda');ws=GemmWorkspace.allocate(x,b)
 rejected('cpu_input',lambda:grouped_gemm(x.cpu(),b,ct))
 rejected('fp32',lambda:grouped_gemm(x.float(),b.float(),ct))
 rejected('dtype_mismatch',lambda:grouped_gemm(x,b.half(),ct))
 rejected('counts_dtype',lambda:grouped_gemm(x,b,ct.int()))
 rejected('counts_length',lambda:grouped_gemm(x,b,ct[:1]))
 rejected('shape_rank',lambda:grouped_gemm(x.unsqueeze(0),b,ct))
 rejected('unsupported_nk',lambda:grouped_gemm(x[:,:512].contiguous(),b[:,:512,:512].contiguous(),ct))
 rejected('noncontiguous',lambda:grouped_gemm(x.T,b,ct))
 raw=torch.empty(x.numel()+1,device='cuda',dtype=x.dtype);unaligned=raw[1:].view_as(x)
 rejected('unaligned_x',lambda:grouped_gemm(unaligned,b,ct))
 rejected('negative_counts_debug',lambda:grouped_gemm(x,b,torch.tensor([-1,65],device='cuda'),debug_counts=True))
 rejected('counts_sum_debug',lambda:grouped_gemm(x,b,torch.tensor([1,1],device='cuda'),debug_counts=True))
 rejected('counts_overflow_debug',lambda:grouped_gemm(x,b,torch.tensor([2**63-1,65],device='cuda'),debug_counts=True))
 out=ws.out;ws.out=x;rejected('out_alias_input',lambda:ws.run(x,b,ct));ws.out=out
 meta=ws.metadata;ws.metadata=meta[:8];rejected('metadata_small',lambda:ws.run(x,b,ct));ws.metadata=meta
 rawmeta=torch.empty(meta.numel()+1,device='cuda',dtype=torch.uint8);ws.metadata=rawmeta[1:]
 rejected('metadata_alignment',lambda:ws.run(x,b,ct));ws.metadata=meta
 ws.out=out[:1];rejected('out_shape',lambda:ws.run(x,b,ct));ws.out=out
 xs,bs,_=tensor_case([0,1],2432,768,torch.bfloat16);cts=torch.tensor([0,1],device='cuda');wss=GemmWorkspace.allocate(xs,bs)
 wss.partial=wss.partial[:1];rejected('partial_small',lambda:wss.run(xs,bs,cts))
 # Invalid values in the fast path are guarded by the kernel but are not synchronous Python errors.
 out.fill_(float('nan'));bad=torch.tensor([-1,65],device='cuda');ws.run(x,b,bad);torch.cuda.synchronize()
 assert torch.isnan(out).all();record('invalid_gpu_counts_no_write')
 # No scalar read or host/device memcpy in the preallocated invocation.
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  ws.run(x,b,ct)
 keys=[e.key for e in prof.key_averages()]
 assert not any('_local_scalar_dense' in k or k=='aten::item' or 'Memcpy' in k for k in keys),keys
 prof.export_chrome_trace(str(a.output/'out_trace.json'));record('out_no_scalar_read_or_copy',events=keys)
 # Explicit weight cache lifecycle and expert ordering; FP32 source unchanged.
 class Expert(torch.nn.Module):
  def __init__(self):
   super().__init__();self.gate_proj=torch.nn.Linear(768,2432,bias=False);self.up_proj=torch.nn.Linear(768,2432,bias=False);self.down_proj=torch.nn.Linear(2432,768,bias=False)
 experts={i:Expert().eval() for i in [0,1]};snap=[e.gate_proj.weight.clone() for e in experts.values()]
 cache=PackedExpertWeights(experts,[1,0],device='cuda');assert cache.nbytes==22413312
 for j,i in enumerate([1,0]):assert torch.equal(cache.get('gate_proj')[j].T,experts[i].gate_proj.weight.to(device='cuda',dtype=torch.bfloat16))
 assert all(torch.equal(e.gate_proj.weight,s) for e,s in zip(experts.values(),snap));record('cache_order_size',nbytes=cache.nbytes,expert_ids=[1,0])
 cache.invalidate('checkpoint reload');rejected('cache_invalidated',lambda:cache.get('gate_proj'))
 experts[0].gate_proj.weight.fill_(0);cache=PackedExpertWeights(experts,[1,0],device='cuda')
 assert not cache.get('gate_proj')[1].count_nonzero();record('cache_rebuilt_after_update')
 experts[0].train();rejected('cache_training',lambda:cache.get('gate_proj'));experts[0].eval()
 rejected('cache_duplicate_ids',lambda:PackedExpertWeights(experts,[0,0],device='cuda'))
 cache=PackedExpertWeights(experts,[1,0],device='cuda');experts[0].gate_proj= torch.nn.Linear(768,2432,bias=False).eval()
 rejected('cache_parameter_replaced',lambda:cache.get('gate_proj'))
 # Async forward: delay the stream, call preallocated op and query completion before synchronize.
 x,b,_=tensor_case([0,1],2432,768,torch.bfloat16);ct=torch.tensor([0,1],device='cuda');ws=GemmWorkspace.allocate(x,b)
 torch.cuda.synchronize();torch.cuda._sleep(100_000_000);start=time.perf_counter();ws.run(x,b,ct);elapsed=time.perf_counter()-start
 done=torch.cuda.Event();done.record();pending=not done.query();done.synchronize()
 assert pending and elapsed<0.03,(pending,elapsed);record('async_no_host_wait',host_seconds=elapsed,pending=pending)
# Reject gradient request outside inference_mode.
x=torch.randn(1,768,device='cuda',dtype=torch.bfloat16,requires_grad=True);b=torch.randn(1,768,2432,device='cuda',dtype=torch.bfloat16);ct=torch.ones(1,device='cuda',dtype=torch.int64)
rejected('forward_gradients',lambda:grouped_gemm(x,b,ct))
(a.output/'validation.json').write_text(json.dumps(dict(passed=True,checks=records,rejections=errors),indent=2)+'\n')
print('ALL API CHECKS PASSED',flush=True)
