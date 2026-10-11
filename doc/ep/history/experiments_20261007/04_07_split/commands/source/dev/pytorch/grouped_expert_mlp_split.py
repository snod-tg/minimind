"""Latest GEMMs with original BF16 SiLU/multiply rounding boundaries."""
import torch
from grouped_gemm_split import GemmWorkspace, PackedExpertWeights, family_for, grouped_gemm

class GroupedExpertMLP:
    def __init__(self, experts, expert_ids, *, dtype=torch.bfloat16):
        self.cache = PackedExpertWeights(experts,expert_ids,dtype=dtype)
        self.streams = {}
        self.prepare_events = 0

    def prepare(self, x):
        return self._prepare(x,self.cache.get("gate_proj"))

    def _prepare(self, x, gate_b):
        if x.ndim != 2 or x.device != self.cache.device or x.dtype != self.cache.dtype or x.shape[1] != gate_b.shape[1]:
            raise ValueError("input must match expert cache [M,K]/device/dtype")
        key = torch.cuda.current_stream(x.device).cuda_stream
        state = self.streams.get(key)
        if state is not None and state[0] >= len(x):
            return state
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("reserve workspace before Graph capture")
        for weight in self.cache.packed.values():
            weight.record_stream(torch.cuda.current_stream(x.device))
        capacity = max(256,1 << max(0,len(x)-1).bit_length())
        gate = GemmWorkspace.allocate(x,gate_b,capacity=capacity)
        up = GemmWorkspace.allocate(x,self.cache.packed["up_proj"],capacity=capacity)
        activated = torch.empty_like(gate.out)
        hidden = torch.empty_like(gate.out)
        down = GemmWorkspace.allocate(hidden,self.cache.packed["down_proj"],capacity=capacity)
        state = (capacity,gate,up,down,activated,hidden)
        self.streams[key] = state
        self.prepare_events += 1
        return state

    def __call__(self, x, counts, *, family="auto"):
        return self._run(x,counts,self.cache.get("gate_proj"),family)

    def _run(self, x, counts, gate_b, family="auto"):
        # Internal path: caller has validated the source cache once for this forward.
        if not len(x):
            grouped_gemm(x,gate_b,counts,family=family)  # validate empty input schema
            return x.new_empty((0,self.cache.packed["down_proj"].shape[2]))
        _,gate,up,down,activated,hidden = self._prepare(x,gate_b)
        selected = family_for(len(x),family)
        gate_y = gate.run(x,gate_b,counts,family=selected)
        act = activated[:len(x)];middle = hidden[:len(x)]
        torch.ops.aten.silu.out(gate_y,out=act)
        up_y = up.run(x,self.cache.packed["up_proj"],counts,family=selected)
        torch.mul(act,up_y,out=middle)
        return down.run(middle,self.cache.packed["down_proj"],counts,family=selected)

    def invalidate(self):
        self.cache.invalidate()
        self.streams.clear()

    def release_workspaces(self):
        self.streams.clear()
