"""Three kernel9 projections with the original BF16 SiLU/multiply boundaries."""
import torch
import torch.nn.functional as F
from my_cuda_kernels.kernel9 import grouped_gemm,GemmWorkspace
from .weights import Kernel9PackedExpertWeights as PackedExpertWeights

class GroupedExpertMLP:
    def __init__(self,experts,expert_ids,*,device,dtype=torch.bfloat16,preallocate=False,cache=None):
        self.cache=cache if cache is not None else PackedExpertWeights(experts,expert_ids,device=device,dtype=dtype)
        if self.cache.expert_ids!=tuple(expert_ids) or self.cache.dtype!=dtype or self.cache.sources!=tuple(experts[i] for i in expert_ids):
            raise ValueError('cache source modules/expert order/dtype must match the requested MLP')
        self.preallocate=preallocate;self.shape=None;self.workspaces=None;self.activated=None;self.hidden=None

    def prepare(self,x):
        """Prepare outside forward/capture; re-prepare when M/device/dtype changes."""
        gate_b=self.cache.get('gate_proj')
        if x.device!=self.cache.device or x.dtype!=self.cache.dtype or x.ndim!=2 or x.shape[1]!=gate_b.shape[1]:
            raise ValueError('x must match the prepared expert cache')
        self.shape=(tuple(x.shape),x.device,x.dtype)
        if self.preallocate and x.shape[0]:
            gate=GemmWorkspace.allocate(x,gate_b);up=GemmWorkspace.allocate(x,self.cache.packed['up_proj'])
            self.activated=torch.empty_like(gate.out);self.hidden=torch.empty_like(gate.out)
            down=GemmWorkspace.allocate(self.hidden,self.cache.packed['down_proj'])
            self.workspaces=(gate,up,down)
        elif self.preallocate:self.workspaces=None;self.activated=None;self.hidden=None
        return self

    def __call__(self,x,counts):
        # Validate the whole source cache once; parameters stay fixed during this forward.
        gate_b=self.cache.get('gate_proj');up_b=self.cache.packed['up_proj'];down_b=self.cache.packed['down_proj']
        if self.preallocate and self.shape!=(tuple(x.shape),x.device,x.dtype):
            raise RuntimeError('prepare workspace for this input shape before forward')
        if x.ndim==2 and x.shape[0]==0:
            if x.device!=self.cache.device or x.dtype!=self.cache.dtype or x.shape[1]!=gate_b.shape[1] or not x.is_contiguous() or x.requires_grad:
                raise ValueError('empty x must match the prepared expert cache')
            if counts.device!=x.device or counts.dtype!=torch.int64 or counts.shape!=(len(self.cache.expert_ids),) or not counts.is_contiguous():
                raise ValueError('counts must be same-device contiguous int64 [E]')
            return x.clone()
        if self.preallocate:
            gate,up,down=self.workspaces
            gate.run(x,gate_b,counts)
            torch.ops.aten.silu.out(gate.out,out=self.activated)
            up.run(x,up_b,counts)
            torch.mul(self.activated,up.out,out=self.hidden)
            return down.run(self.hidden,down_b,counts)
        gate=grouped_gemm(x,gate_b,counts);activated=F.silu(gate)
        up=grouped_gemm(x,up_b,counts)
        return grouped_gemm(activated*up,down_b,counts)

    def invalidate(self,reason='weights changed'):
        self.cache.invalidate(reason);self.workspaces=None;self.activated=None;self.hidden=None;self.shape=None
