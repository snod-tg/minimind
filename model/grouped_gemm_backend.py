"""MiniMind adapters; original PyTorch models need no CUDA extension."""
def load_kn_model(model, checkpoint):
    from .checkpoint_kn import load_kn_checkpoint
    return load_kn_checkpoint(model, checkpoint, ep_subset=bool(model.config.use_ep))

def attach_grouped_backend(model, *, backend="native_bf16", library=None):
    if backend in ("cuda_split", "cuda_split_last_layer"):
        from my_cuda_kernels.grouped_gemm import load_library
        load_library(library)
    from .backends.grouped_gemm import MiniMindEPSplitBackend
    return MiniMindEPSplitBackend(model, backend)
