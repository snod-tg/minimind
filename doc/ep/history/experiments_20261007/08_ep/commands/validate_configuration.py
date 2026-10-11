"""Runtime partition, configurable fixed capacity and native reduction contracts."""
import argparse
import json
import sys
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--quick', action='store_true')
a = p.parse_args()
sys.path.insert(0, str(Path('source/dev/pytorch').resolve()))
from ep_routing_ops import load_library
from minimind_ep_routing import Workspace, combine_inference

load_library(a.library)
torch.set_num_threads(1)
torch.manual_seed(20261009)
ws = Workspace()
rows = []

def exact(x, y, label, **info):
    assert torch.equal(x, y), (label, info, float((x.float()-y.float()).abs().max()) if x.numel() else 0)
    rows.append(dict(test=label, exact=True, **info))

with torch.inference_mode():
    for bins in [1, 2, 3, 4, 8, 16, 64, 1024]:
        for n in [0, 1, 33, 256, 257, 513, 65537]:
            for masked in [False, True]:
                divisor = 1 if masked else 3
                ids = torch.randint(bins * divisor, (n,), device='cuda')
                if masked:
                    ids[::3] = -1
                perm, inv, counts = ws.sort(ids, divisor, 'partition', bins, masked)
                valid = torch.nonzero(ids >= 0, as_tuple=True)[0]
                expected = valid[torch.argsort(ids[valid] // divisor, stable=True)]
                exact(perm[:len(valid)], expected, 'partition', bins=bins, n=n, masked=masked)
                exact(counts, torch.bincount(ids[valid] // divisor, minlength=bins), 'counts')
                exact(perm[inv[valid]], valid, 'inverse')
                if masked:
                    exact(inv[ids < 0], torch.full_like(inv[ids < 0], -1), 'padding')
    for ranks, experts_per_rank in [(1, 1), (2, 2), (2, 3), (3, 2), (4, 4)]:
        experts = ranks * experts_per_rank
        for topk in sorted({1, min(2, experts), min(3, experts), experts}):
            for n in [0, 1, 33]:
                dtype = torch.bfloat16
                h, capacity = 64, 40 * topk
                x = torch.randn(n, h, device='cuda', dtype=dtype)
                ids = torch.randint(experts, (n * topk,), device='cuda')
                perm, inv, counts = ws.sort(ids, experts_per_rank, 'send', ranks)
                wire = torch.empty(ranks * capacity, h, device='cuda', dtype=dtype)
                meta = torch.empty(ranks * capacity, device='cuda', dtype=torch.int64)
                inverse = torch.empty_like(ids)
                torch.ops.my_cuda_ep.pack_fixed_out(x, ids, perm, counts, topk, experts_per_rank, wire, meta, inverse)
                ep, ei, ec = ws.sort(meta, 1, 'fixed', experts_per_rank, masked=True)
                valid = torch.nonzero(meta >= 0, as_tuple=True)[0]
                expected = valid[torch.argsort(meta[valid], stable=True)]
                exact(ep[:len(valid)], expected, 'fixed_partition', experts=experts, ranks=ranks, topk=topk, n=n)
                exact(ec, torch.bincount(meta[valid], minlength=experts_per_rank), 'fixed_counts')
                packed = torch.empty(len(valid), h, device='cuda', dtype=dtype)
                torch.ops.my_cuda_ep.gather_out(wire, ep[:len(valid)], packed)
                back = torch.empty_like(wire)
                torch.ops.my_cuda_ep.scatter_fixed_out(packed, ei, back)
                exact(back, wire, 'fixed_restore')
                exact(back[inverse], x.repeat_interleave(topk, dim=0), 'fixed_inverse', experts=experts, ranks=ranks, topk=topk, n=n)
                weighted = torch.rand(n * topk, device='cuda', dtype=dtype)
                out = torch.empty_like(x)
                combine_inference(back, weighted, inverse, topk, out)
                reference = torch.zeros_like(x).index_add_(0, torch.arange(n, device='cuda').repeat_interleave(topk), x.repeat_interleave(topk, dim=0) * weighted[:, None])
                if topk <= 2:
                    exact(out, reference, 'fixed_combine', experts=experts, ranks=ranks, topk=topk, n=n)
                else:
                    magnitudes = (x.repeat_interleave(topk, dim=0) * weighted[:, None]).float().abs().reshape(n, topk, h).sum(1)
                    error = (out.float()-reference.float()).abs()
                    assert bool((error <= magnitudes * torch.finfo(dtype).eps * topk).all()), (experts, ranks, topk, n)
                    rows.append(dict(test='fixed_combine_native_rounding', experts=experts, ranks=ranks, topk=topk, n=n, exact=torch.equal(out, reference), bounded=True, max_abs=float(error.max()) if error.numel() else 0))

if not a.quick:
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for topk in [1, 2, 3, 4, 8]:
            n, h, experts_per_rank = 33, 64, 4
            ids = torch.randint(8, (n * topk,), device='cuda')
            with torch.no_grad():
                perm, inv, counts = ws.sort(ids, experts_per_rank, 'grad', 2)
            initial = torch.randn(n, h, device='cuda', dtype=dtype)
            upstream = torch.randn(n * topk, h, device='cuda', dtype=dtype)
            gradients = []
            token_index = torch.arange(n, device='cuda').repeat_interleave(topk)
            for custom in [False, False, True]:
                x = initial.clone().requires_grad_()
                # Match the original two indexing steps, including backward order.
                y = torch.ops.my_cuda_ep.pack(x, ids, perm, inv, topk, experts_per_rank)[0] if custom else x[token_index][perm]
                y.backward(upstream)
                gradients.append(x.grad)
            reference, repeated, actual = gradients
            # Three or more atomic contributions may sum in a different order.
            # Bound error by the sum of magnitudes, so cancellation stays covered.
            bound = torch.zeros_like(initial, dtype=torch.float32).index_add_(0, torch.arange(n, device='cuda').repeat_interleave(topk), upstream[inv].float().abs())
            tolerance = bound * torch.finfo(dtype).eps * max(2, topk)
            error = (actual.float() - reference.float()).abs()
            assert bool((error <= tolerance).all()), (dtype, topk, float(error.max()))
            if topk <= 2:
                exact(actual, reference, 'pack_grad', dtype=str(dtype), topk=topk)
            rows.append(dict(test='pack_grad_rounding', dtype=str(dtype), topk=topk, exact=torch.equal(actual, reference), max_abs=float(error.max()), original_repeat_max_abs=float((reference.float()-repeated.float()).abs().max()), bounded=True))
            returned = torch.randn(n * topk, h, device='cuda', dtype=dtype)
            weight = torch.rand(n * topk, device='cuda', dtype=dtype)
            output_grad = torch.randn(n, h, device='cuda', dtype=dtype)
            values = []
            for custom in [False, False, True]:
                x = returned.clone().requires_grad_(); w = weight.clone().requires_grad_()
                if custom:
                    y = torch.ops.my_cuda_ep.combine(x, w, inv, topk, (torch.float32, torch.float16, torch.bfloat16).index(dtype))
                else:
                    y = torch.zeros(n, h, device='cuda', dtype=dtype).index_add_(0, torch.arange(n, device='cuda').repeat_interleave(topk), x[inv] * w[:, None])
                y.backward(output_grad)
                values.append((y.detach(), x.grad, w.grad))
            for name, actual, reference in zip(['output', 'returned_grad', 'weight_grad'], values[2], values[0]):
                if name != 'output' or topk <= 2:
                    exact(actual, reference, 'combine_contract', dtype=str(dtype), topk=topk, tensor=name)
                else:
                    magnitude = (returned[inv] * weight[:, None]).float().abs().reshape(n, topk, h).sum(1)
                    error = (actual.float()-reference.float()).abs()
                    assert bool((error <= magnitude * torch.finfo(dtype).eps * topk).all()), (dtype, topk)
                    rows.append(dict(test='combine_native_rounding', dtype=str(dtype), topk=topk, exact=torch.equal(actual, reference), bounded=True, max_abs=float(error.max()), original_repeat_max_abs=float((values[0][0].float()-values[1][0].float()).abs().max())))
    # Validate all functional operators, including tracing of their gradients.
    x = torch.randn(33, 64, device='cuda', requires_grad=True)
    ids = torch.randint(8, (66,), device='cuda')
    with torch.no_grad(): perm, inv, _ = ws.sort(ids, 4, 'opcheck')
    returned = torch.randn(66, 64, device='cuda', requires_grad=True)
    weight = torch.rand(66, device='cuda', requires_grad=True)
    for op, args in [('pack', (x, ids, perm, inv, 2, 4)), ('gather', (returned, perm)), ('combine', (returned, weight, inv, 2, 0))]:
        rows.append(dict(test='opcheck', op=op, checks=torch.library.opcheck(getattr(torch.ops.my_cuda_ep, op).default, args)))

a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(checks=rows, torch=torch.__version__, gpu=torch.cuda.get_device_name()), indent=2) + '\n')
print('CONFIGURATION PASSED', len(rows), flush=True)
