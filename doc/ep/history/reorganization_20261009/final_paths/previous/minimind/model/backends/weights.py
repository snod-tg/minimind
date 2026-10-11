"""MiniMind expert-module weight caches; tensor GEMMs live in my_cuda_kernels."""
import torch

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


class Kernel9PackedExpertWeights:
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
