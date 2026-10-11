"""Three kernel9 projections with the original BF16 SiLU/multiply boundaries."""
import torch
import torch.nn.functional as F
from grouped_gemm_api import grouped_gemm,GemmWorkspace,PackedExpertWeights

class GroupedExpertMLP:
    def __init__(self,experts,expert_ids,*,device,dtype=torch.bfloat16,preallocate=False,cache=None):
        self.cache=cache if cache is not None else PackedExpertWeights(experts,expert_ids,device=device,dtype=dtype)
        self.preallocate=preallocate;self.shape=None;self.workspaces=None

    def prepare(self,x):
        """Prepare outside timed forward/capture; re-prepare when M/device/dtype changes."""
        gate_b=self.cache.get('gate_proj');up_b=self.cache.get('up_proj');down_b=self.cache.get('down_proj')
        if x.device!=self.cache.device or x.dtype!=self.cache.dtype or x.ndim!=2 or x.shape[1]!=gate_b.shape[1]:
            raise ValueError('x must match the prepared expert cache')
        self.shape=(tuple(x.shape),x.device,x.dtype)
        if self.preallocate:
            gate=GemmWorkspace.allocate(x,gate_b);up=GemmWorkspace.allocate(x,up_b)
            middle=torch.empty_like(gate.out);down=GemmWorkspace.allocate(middle,down_b)
            self.workspaces=(gate,up,down)
        return self

    def __call__(self,x,counts):
        # First implementation keeps explicit cache validation at each projection.
        if self.preallocate and self.shape!=(tuple(x.shape),x.device,x.dtype):
            raise RuntimeError('prepare workspace for this input shape before forward')
        gate_b=self.cache.get('gate_proj')
        gate=self.workspaces[0].run(x,gate_b,counts) if self.preallocate else grouped_gemm(x,gate_b,counts)
        activated=F.silu(gate)
        up_b=self.cache.get('up_proj')
        up=self.workspaces[1].run(x,up_b,counts) if self.preallocate else grouped_gemm(x,up_b,counts)
        middle=activated*up
        down_b=self.cache.get('down_proj')
        return self.workspaces[2].run(middle,down_b,counts) if self.preallocate else grouped_gemm(middle,down_b,counts)

    def invalidate(self,reason='weights changed'):
        self.cache.invalidate(reason);self.workspaces=None;self.shape=None
