"""Forward-only BF16 compatibility path for compact expert-major inputs.

Preserves PyTorch's current BF16 reduction policy and contiguous [N,K]
projection weights. The handwritten prefill/decode kernels are separate
FP32-accumulation paths; this fallback does not claim their throughput.
"""
import torch
import torch.nn.functional as F
from contextlib import nullcontext

class NativeBF16GroupedGemm:
    def __init__(self, weights_kn: torch.Tensor, family="prefill"):
        if family not in ("prefill", "decode"):
            raise ValueError("family must be prefill/decode")
        if weights_kn.ndim != 3 or not weights_kn.is_cuda or not weights_kn.is_contiguous():
            raise ValueError("weights require contiguous CUDA [E,K,N]")
        e,k,n=weights_kn.shape
        if not 1<=e<=16 or (k,n) not in ((768,2432),(2432,768),(1024,1024)):
            raise ValueError("unsupported expert count/projection shape")
        if weights_kn.dtype not in (torch.float32,torch.float16,torch.bfloat16):
            raise ValueError("compatibility weights must preserve original floating-point values")
        self.family=family;self.weights_kn=weights_kn
        self.experts,self.k,self.n=e,k,n
        self._native_weights=None;self._stamp=None
        self.prepare()

    def _version(self):
        w=self.weights_kn
        try: version=w._version
        except RuntimeError: version=None
        # Inference tensors have no version counter: invalidate explicitly
        # after modifying those weights in place.
        return (id(w),w.data_ptr(),version,w.device,w.dtype)

    def prepare(self):
        """One-time [K,N]->[N,K] cache. Does not alter canonical checkpoint values."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare weights before graph capture")
        # Normal leaf parameters allow the same outer autocast weight cache
        # as the original model. Forward-only: no gradient to canonical weights.
        with torch.inference_mode(False),torch.no_grad():
            self._native_weights=[torch.nn.Parameter(self.weights_kn[e].T.contiguous(),requires_grad=True)
                                  for e in range(self.experts)]
        self._stamp=self._version()

    def invalidate(self):
        self._native_weights=None;self._stamp=None
        torch.clear_autocast_cache()

    def __call__(self,x:torch.Tensor,counts):
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("compatibility path is forward-only; use inference_mode")
        if x.ndim!=2 or x.shape[1]!=self.k or not x.is_contiguous():
            raise ValueError("input requires contiguous [M,K]")
        if x.device!=self.weights_kn.device or x.dtype!=torch.bfloat16:
            raise ValueError("input requires BF16 on the weights' CUDA device")
        if self.family=="decode" and len(x)>256:
            raise ValueError("decode supports total M<=256")
        if self._stamp!=self._version():
            # Refreshing while an outer autocast cache remains live can reuse
            # stale casts, so clear it explicitly with the weight cache.
            self.invalidate();self.prepare()
        if isinstance(counts,torch.Tensor):
            if counts.ndim!=1 or len(counts)!=self.experts or counts.dtype!=torch.int64:
                raise ValueError("counts requires int64 [E]")
            if counts.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("dynamic GPU counts compatibility fallback requires host synchronization")
            host=counts.detach().cpu().tolist()
        else:host=list(counts)
        if len(host)!=self.experts or any(type(m)!=int or m<0 for m in host) or sum(host)!=len(x):
            raise ValueError("counts must be nonnegative and sum to M")
        if not len(x):return x.new_empty((0,self.n))
        pieces=[];off=0
        # Preserve the actual default BF16 policy; never disable reduced
        # precision globally to make a failed compatibility test pass.
        active=torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda")==torch.bfloat16
        with nullcontext() if active else torch.autocast("cuda",dtype=torch.bfloat16):
            for e,m in enumerate(host):
                if m:pieces.append(F.linear(x[off:off+m],self._native_weights[e]))
                off+=m
        return torch.cat(pieces,dim=0)
