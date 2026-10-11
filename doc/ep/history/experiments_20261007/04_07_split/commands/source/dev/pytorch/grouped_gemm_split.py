"""Forward-only registered prefill/decode GEMMs; counts remain on the GPU."""
from dataclasses import dataclass
from pathlib import Path
import torch

_loaded = None

def family_for(m, family="auto"):
    if family == "auto":
        return "decode" if m <= 256 else "prefill"
    if family not in ("prefill", "decode"):
        raise ValueError("family must be auto/prefill/decode")
    return family

def load_library(path):
    global _loaded
    path = Path(path).resolve()
    if _loaded is not None:
        if path != _loaded:
            raise RuntimeError("another split library is already loaded")
        return
    torch.ops.load_library(str(path))
    for family in ("prefill", "decode"):
        def make_fake(selected):
            def fake(x, b, counts):
                torch._check(x.is_cuda and b.is_cuda and counts.is_cuda)
                torch._check(x.device == b.device == counts.device)
                torch._check(x.ndim == 2 and b.ndim == 3 and counts.ndim == 1)
                torch._check(x.is_contiguous() and b.is_contiguous() and counts.is_contiguous())
                torch._check(x.dtype in (torch.float16, torch.bfloat16) and b.dtype == x.dtype)
                torch._check(counts.dtype == torch.int64)
                torch._check((b.shape[0] >= 1) & (b.shape[0] <= 16) & (len(counts) == b.shape[0]))
                torch._check(x.shape[1] == b.shape[1])
                torch._check((x.shape[1] == 768 and b.shape[2] == 2432) or
                             (x.shape[1] == 2432 and b.shape[2] == 768) or
                             (x.shape[1] == 1024 and b.shape[2] == 1024))
                torch._check(not x.requires_grad and not b.requires_grad)
                if selected == "decode":
                    torch._check(x.shape[0] <= 256)
                return x.new_empty((x.shape[0], b.shape[2]))
            return fake
        torch.library.register_fake(f"my_cuda_kernels_split::{family}")(make_fake(family))
        torch.library.register_fake(f"my_cuda_kernels_split::{family}_out")(lambda x,b,counts,out,metadata,partial: None)
        def backward(ctx, grad):
            raise RuntimeError("split Grouped GEMM has no backward")
        torch.library.register_autograd(f"my_cuda_kernels_split::{family}", backward)
    _loaded = path

def validate_counts(x, b, counts):
    if counts.dtype != torch.int64 or counts.device != x.device or counts.shape != (b.shape[0],):
        raise ValueError("counts must be same-device int64 [E]")
    values = counts.cpu().tolist()
    if any(v < 0 for v in values) or sum(values) != len(x):
        raise ValueError("counts must be nonnegative and sum to M")

def grouped_gemm(x, packed_b, counts, *, family="auto", debug_counts=False):
    if debug_counts:
        validate_counts(x, packed_b, counts)
    return getattr(torch.ops.my_cuda_kernels_split, family_for(len(x), family))(x, packed_b, counts)

@dataclass
class GemmWorkspace:
    """Reusable capacity; separate workspaces are required for concurrent streams."""
    out: torch.Tensor
    metadata: torch.Tensor
    partial: torch.Tensor
    capacity: int

    @classmethod
    def allocate(cls, x, b, *, capacity=None):
        m, k = x.shape
        capacity = m if capacity is None else capacity
        if capacity < m:
            raise ValueError("workspace capacity is smaller than M")
        n = b.shape[2]
        # Maximum Split-K is 8; decode's supported rows are at most 256.
        return cls(torch.empty((capacity,n),device=x.device,dtype=x.dtype),
                   torch.empty(torch.ops.my_cuda_kernels_split.metadata_bytes(),device=x.device,dtype=torch.uint8),
                   torch.empty(8*min(capacity,256)*n,device=x.device,dtype=torch.float32),capacity)

    def run(self, x, b, counts, *, family="auto", debug_counts=False):
        if len(x) > self.capacity:
            raise ValueError("M exceeds workspace capacity")
        if debug_counts:
            validate_counts(x, b, counts)
        out = self.out[:len(x)]
        getattr(torch.ops.my_cuda_kernels_split, family_for(len(x),family)+"_out")(
            x,b,counts,out,self.metadata,self.partial)
        return out

class PackedExpertWeights:
    """Inference cache for native NK or canonical KN Linear weights."""
    def __init__(self, experts, expert_ids, *, dtype=torch.bfloat16, device=None):
        if dtype not in (torch.float16,torch.bfloat16):
            raise ValueError("FP16/BF16 packed cache required")
        self.expert_ids = tuple(expert_ids)
        if not 1 <= len(self.expert_ids) <= 16 or len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("one to sixteen unique local expert ids required")
        self.sources = tuple(experts[i] for i in self.expert_ids)
        if any(e.training for e in self.sources):
            raise RuntimeError("prepare experts after eval()")
        self.dtype = dtype
        self.valid = True
        self.stamp = self.fingerprint()
        self.packed = {}
        with torch.inference_mode():
            for name in ("gate_proj", "up_proj", "down_proj"):
                weights = []
                for e in self.sources:
                    layer = getattr(e,name)
                    if layer.bias is not None:
                        raise ValueError("expert projections must have no bias")
                    w = layer.weight.detach()
                    # LinearKN exposes _nk; its canonical parameter already is KN.
                    weights.append(w if hasattr(layer,"_nk") else w.T)
                self.packed[name] = torch.stack(weights).to(device=device,dtype=dtype).contiguous()
        self.device = self.packed["gate_proj"].device

    def fingerprint(self):
        values = []
        for e in self.sources:
            for name in ("gate_proj","up_proj","down_proj"):
                layer = getattr(e,name);w = layer.weight
                try:
                    version = w._version
                except RuntimeError:
                    version = None
                values.append((id(layer),id(w),w.data_ptr(),version,w.device,w.dtype))
        return tuple(values)

    def get(self, name):
        if not self.valid or any(e.training for e in self.sources) or self.fingerprint() != self.stamp:
            raise RuntimeError("packed expert cache is stale; invalidate and rebuild")
        return self.packed[name]

    @property
    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in self.packed.values())

    def invalidate(self, reason="weights changed"):
        self.valid = False
        self.packed.clear()
