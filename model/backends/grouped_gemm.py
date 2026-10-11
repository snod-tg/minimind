"""Backend used at MiniMind's explicit local-expert extension point."""
import torch
from .grouped_expert_mlp import GroupedExpertMLP

class MiniMindEPSplitBackend:
    MODES = ("original","native_bf16","cuda_split","cuda_split_last_layer")

    def __init__(self, model, backend="native_bf16"):
        if backend not in self.MODES:
            raise ValueError("unknown expert backend")
        self.model = model
        self.backend = backend
        self.layers = [layer.mlp for layer in model.model.layers]
        self.states = {}
        self.capture = False
        self.rows = {}
        self.routes = {}
        self.calls = {}
        self.prepare_events = 0
        self.load_hook = model.register_load_state_dict_post_hook(lambda module, incompatible: self.invalidate())
        for i,mlp in enumerate(self.layers):
            if not hasattr(mlp,"local_expert_ids"):
                raise ValueError("split backend requires Expert Parallel layers")
            if getattr(mlp,"_grouped_expert_backend",None) is not None:
                raise RuntimeError("an expert backend is already attached")
            object.__setattr__(mlp,"_grouped_expert_backend",self)
            mlp._grouped_expert_layer = i

    def can_use(self, mlp, x):
        if mlp.training:
            self.invalidate(mlp._grouped_expert_layer)
            return False
        # Includes ordinary no_grad/evaluation in the native fallback.
        if not torch.is_inference_mode_enabled():
            return False
        return self.backend != "original" or self.capture

    def state(self, mlp, dtype=torch.bfloat16):
        i = mlp._grouped_expert_layer
        old = self.states.get(i)
        if old is not None and (not old.cache.valid or old.cache.dtype != dtype or old.cache.fingerprint() != old.cache.stamp):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("weights changed during Graph capture")
            self.invalidate(i)
            old = None
        if old is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare expert weights before Graph capture")
            experts = {j:mlp.experts[str(j)] for j in mlp.local_expert_ids}
            old = GroupedExpertMLP(experts,mlp.local_expert_ids,dtype=dtype)
            self.states[i] = old
        return old

    def __call__(self, mlp, x, counts):
        return self.experts(mlp,x,counts)

    def experts(self, mlp, x, counts):
        i = mlp._grouped_expert_layer
        supported = x.is_cuda and x.dtype in (torch.bfloat16,torch.float16) and x.ndim == 2 and x.shape[1] == 768
        supported = supported and mlp.config.moe_intermediate_size == 2432 and mlp.config.hidden_act == "silu" and len(mlp.local_expert_ids) <= 16
        use_cuda = self.backend in ("cuda_split","cuda_split_last_layer") and supported
        if self.backend == "cuda_split_last_layer":
            use_cuda = use_cuda and i == len(self.layers)-1 and len(x) <= 256
        if use_cuda and not len(x):
            y = x.clone()
            path = "empty"
        elif use_cuda:
            state = self.state(mlp,x.dtype)
            before = state.prepare_events
            y = state._run(x,counts,state.cache.packed["gate_proj"])
            self.prepare_events += state.prepare_events-before
            path = "decode" if len(x) <= 256 else "prefill"
        else:
            # Exact original module calls, also supporting canonical LinearKN.
            values = []
            offset = 0
            for expert_id,m in zip(mlp.local_expert_ids,counts.tolist()):
                if m:
                    values.append(mlp.experts[str(expert_id)](x[offset:offset+m]))
                offset += m
            y = torch.cat(values) if values else x.clone()
            path = "native_bf16"
        if self.capture:
            self.calls[i] = path
            self.rows[i] = dict(m=len(x),counts=counts.cpu().tolist(),input=x.detach().cpu(),output=y.detach().cpu(),path=path)
        return y

    def initialize(self):
        for mlp in self.layers:
            self.state(mlp)

    def invalidate(self, layer=None):
        torch.clear_autocast_cache()
        for i in list(self.states):
            if layer is None or layer == i:
                self.states.pop(i).invalidate()

    def release_workspaces(self):
        for state in self.states.values():
            state.release_workspaces()

    def reset_capture(self):
        self.rows = {}
        self.routes = {}
        self.calls = {}

    def uninstall(self):
        self.load_hook.remove()
        self.invalidate()
        for mlp in self.layers:
            del mlp._grouped_expert_backend
            del mlp._grouped_expert_layer
