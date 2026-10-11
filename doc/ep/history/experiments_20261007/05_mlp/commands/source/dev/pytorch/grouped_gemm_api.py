"""Forward-only kernel9 API, explicit workspaces and expert-weight cache."""
from dataclasses import dataclass
from pathlib import Path
import torch

_loaded=None

def load_library(path):
    global _loaded
    path=Path(path).resolve()
    if _loaded is not None:
        if path!=_loaded:raise RuntimeError('another kernel9 library is already loaded')
        return
    torch.ops.load_library(str(path))
    # Shape-only functional registration; does not read data or launch CUDA.
    @torch.library.register_fake('my_cuda_kernels::grouped_gemm')
    def fake(x,b,counts):
        torch._check(x.is_cuda and b.is_cuda and counts.is_cuda)
        torch._check(x.device==b.device==counts.device)
        torch._check(x.ndim==2 and b.ndim==3 and counts.ndim==1)
        torch._check(x.is_contiguous() and b.is_contiguous() and counts.is_contiguous())
        torch._check(x.dtype in (torch.float16,torch.bfloat16) and b.dtype==x.dtype and counts.dtype==torch.int64)
        torch._check((b.shape[0]>=1)&(b.shape[0]<=16)&(counts.shape[0]==b.shape[0]))
        torch._check(x.shape[1]==b.shape[1])
        torch._check((x.shape[1]==768 and b.shape[2]==2432) or (x.shape[1]==2432 and b.shape[2]==768) or (x.shape[1]==1024 and b.shape[2]==1024))
        torch._check(not x.requires_grad and not b.requires_grad)
        return x.new_empty((x.shape[0],b.shape[2]))
    @torch.library.register_fake('my_cuda_kernels::grouped_gemm_out')
    def fake_out(x,b,counts,out,metadata,partial):
        return None
    def backward(ctx,grad):
        raise RuntimeError('grouped_gemm has no backward implementation')
    torch.library.register_autograd('my_cuda_kernels::grouped_gemm',backward)
    _loaded=path

def check_counts(x,b,counts):
    """Synchronous debug validation, deliberately excluded from normal calls."""
    if counts.ndim!=1 or counts.dtype!=torch.int64 or counts.device!=x.device:
        raise ValueError('counts must be same-device int64 [E]')
    values=counts.cpu().tolist()
    if len(values)!=b.shape[0] or any(v<0 or v>x.shape[0] for v in values) or sum(values)!=x.shape[0]:
        raise ValueError('counts must be nonnegative and sum to M')

def grouped_gemm(x,packed_b,counts,*,validate_counts=False):
    if validate_counts:check_counts(x,packed_b,counts)
    return torch.ops.my_cuda_kernels.grouped_gemm(x,packed_b,counts)

@dataclass
class GemmWorkspace:
    """One invocation at a time; use a separate workspace per concurrent stream."""
    out:torch.Tensor
    metadata:torch.Tensor
    partial:torch.Tensor

    @classmethod
    def allocate(cls,x,b):
        m,k=x.shape;n=b.shape[2]
        size=torch.ops.my_cuda_kernels.grouped_gemm_metadata_bytes()
        parts=torch.ops.my_cuda_kernels.grouped_gemm_parts(m,n,k)
        return cls(torch.empty((m,n),device=x.device,dtype=x.dtype),
                   torch.empty(size,device=x.device,dtype=torch.uint8),
                   torch.empty(parts*m*n,device=x.device,dtype=torch.float32))

    def run(self,x,b,counts,*,validate_counts=False):
        if validate_counts:check_counts(x,b,counts)
        torch.ops.my_cuda_kernels.grouped_gemm_out(x,b,counts,self.out,self.metadata,self.partial)
        return self.out

class PackedExpertWeights:
    """Pack once after load/eval; explicitly invalidate on any weight update."""
    def __init__(self,experts,expert_ids,*,device,dtype=torch.bfloat16):
        if dtype not in (torch.float16,torch.bfloat16):raise ValueError('FP16/BF16 cache required')
        self.expert_ids=tuple(expert_ids)
        if not 1<=len(self.expert_ids)<=16 or len(set(self.expert_ids))!=len(self.expert_ids):
            raise ValueError('one to sixteen unique local expert ids required')
        self.sources=tuple(experts[i] for i in self.expert_ids)
        if any(e.training for e in self.sources):raise RuntimeError('build weights after eval()')
        self.valid=True;self.reason='';self.dtype=dtype
        self.fingerprints=self._fingerprints()
        with torch.inference_mode():
            self.packed={name:torch.stack([getattr(e,name).weight.detach() for e in self.sources]).to(device=device,dtype=dtype).transpose(1,2).contiguous()
                         for name in ('gate_proj','up_proj','down_proj')}
        self.device=self.packed['gate_proj'].device

    def _fingerprints(self):
        return tuple((id(w),w.data_ptr(),w.device,w.dtype) for e in self.sources
                     for w in (e.gate_proj.weight,e.up_proj.weight,e.down_proj.weight))

    def invalidate(self,reason='weights or model state changed'):
        self.valid=False;self.reason=reason;self.packed.clear()

    def get(self,name):
        if any(e.training for e in self.sources):self.invalidate('model entered training')
        elif self.valid and self._fingerprints()!=self.fingerprints:self.invalidate('parameter identity/device/dtype changed')
        if not self.valid:raise RuntimeError(f'packed cache invalid: {self.reason}; rebuild it')
        return self.packed[name]

    @property
    def nbytes(self):return sum(t.numel()*t.element_size() for t in self.packed.values())
