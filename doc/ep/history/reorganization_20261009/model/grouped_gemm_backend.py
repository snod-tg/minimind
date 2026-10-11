"""Opt-in bridge to my_cuda_kernels; original MiniMind has no extra dependency."""
from pathlib import Path
import sys

def _python_path(python_dir):
    if python_dir is None:
        python_dir = Path(__file__).resolve().parents[2]/"my_cuda_kernels/dev/pytorch"
    python_dir = Path(python_dir).resolve()
    if not (python_dir/"minimind_ep_split_backend.py").is_file():
        raise FileNotFoundError(f"split backend Python package missing: {python_dir}")
    if str(python_dir) not in sys.path:
        sys.path.insert(0,str(python_dir))
    return python_dir

def load_kn_model(model, checkpoint, *, python_dir=None):
    _python_path(python_dir)
    from checkpoint_kn import load_kn_checkpoint
    return load_kn_checkpoint(model,checkpoint,ep_subset=bool(model.config.use_ep))

def attach_grouped_backend(model, *, backend="native_bf16", library=None, python_dir=None):
    _python_path(python_dir)
    if backend in ("cuda_split","cuda_split_last_layer") and library is None:
        raise ValueError("CUDA split backend requires its built library")
    from minimind_ep_split_backend import MiniMindEPSplitBackend
    if library is not None:
        from grouped_gemm_split import load_library
        load_library(library)
    return MiniMindEPSplitBackend(model,backend)
