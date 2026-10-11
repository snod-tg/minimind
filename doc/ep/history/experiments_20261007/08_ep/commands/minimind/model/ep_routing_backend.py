"""Explicit bridge to optimized EP; missing libraries fail instead of falling back."""
from pathlib import Path
import sys

def attach_ep_routing(model,*,library,python_dir=None,mode='compact',decode_capacity_tokens=32):
    if library is None:raise ValueError('optimized EP requires --ep_routing_library')
    path=Path(python_dir) if python_dir else Path(__file__).resolve().parents[2]/'my_cuda_kernels/dev/pytorch'
    path=path.resolve()
    if not (path/'minimind_ep_routing.py').is_file():raise FileNotFoundError(path/'minimind_ep_routing.py')
    if str(path) not in sys.path:sys.path.insert(0,str(path))
    from minimind_ep_routing import MiniMindEPRouting
    return MiniMindEPRouting(model,library,mode,decode_capacity_tokens=decode_capacity_tokens)
