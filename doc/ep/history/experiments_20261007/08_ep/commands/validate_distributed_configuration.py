"""Small full-model/gradient checks for real configurable NCCL groups."""
import argparse
import importlib.util
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--experts', type=int, required=True)
p.add_argument('--topks', type=int, nargs='+', default=[1, 2])
a = p.parse_args()
rank, local, world = (int(os.environ[n]) for n in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
torch.cuda.set_device(local)
torch.set_num_threads(1)
dist.init_process_group('nccl', device_id=torch.device('cuda', local), timeout=timedelta(seconds=120))
sys.path.insert(0, str(Path('source/dev/pytorch').resolve()))
spec = importlib.util.spec_from_file_location('ep_config_model', Path('minimind/model/model_minimind.py').resolve())
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
torch.manual_seed(410 + rank)
config = module.MiniMindConfig(hidden_size=64, num_hidden_layers=2, vocab_size=128, num_attention_heads=4, num_key_value_heads=2,
    use_moe=True, use_ep=True, ep_size=world, num_experts=a.experts, num_experts_per_tok=a.topks[0],
    moe_intermediate_size=128, dropout=0., ep_comm_dtype='native', max_position_embeddings=128)
model = module.MiniMindForCausalLM(config).cuda()
for name, param in model.named_parameters():
    if '.mlp.experts.' not in name:
        dist.broadcast(param.data, src=0)
from minimind_ep_routing import MiniMindEPRouting
controller = MiniMindEPRouting(model, a.library)
rows = []
a.output.mkdir(parents=True, exist_ok=True)

def check(y, ref, label, topk, **extra):
    error = y.float() - ref.float()
    maximum = float(error.abs().max()) if error.numel() else 0.
    relative = float(error.norm() / ref.float().norm().clamp_min(1e-12))
    exact = torch.equal(y, ref)
    row = dict(test=label, topk=topk, exact=exact, max_abs=maximum, relative_rms=relative, **extra)
    rows.append(row)
    if topk <= 2:
        assert exact, row
    else:
        # Multi-way native BF16 reductions need not be bitwise deterministic.
        assert torch.isfinite(y).all() and relative < .02, row

for topk in a.topks:
    config.num_experts_per_tok = topk
    model.eval()
    for mode in ['compact', 'fixed'] + (['dedup'] if topk == 2 else []):
        controller.mode = mode
        controller.fixed_decode = mode == 'fixed'
        for length in [1, 17, 33]:
            if mode == 'fixed' and length > controller.decode_capacity_tokens:
                continue
            torch.manual_seed(701 + rank + length)
            ids = torch.randint(3, 128, (1, length), device='cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                controller.enabled = False
                ref = model(ids)
                controller.enabled = True
                y = model(ids)
                check(y.logits, ref.logits, 'model_logits', topk, mode=mode, tokens=length)
                check(y.hidden_states, ref.hidden_states, 'model_hidden', topk, mode=mode, tokens=length)
    controller.mode = 'compact'
    controller.fixed_decode = False
    model.train()
    owner = model.model.layers[0].mlp
    torch.manual_seed(501 + rank)
    initial = torch.randn(1, 17, 64, device='cuda', dtype=torch.bfloat16)
    values = []
    for enabled in [False, False, True]:
        controller.enabled = enabled
        model.zero_grad(set_to_none=True)
        x = initial.clone().requires_grad_()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = owner(x)
            loss = y.float().square().mean() + owner.aux_loss
        loss.backward()
        values.append(dict(output=y.detach(), input=x.grad.clone(), **{name: param.grad.clone() for name, param in owner.named_parameters() if param.grad is not None}))
    assert values[0].keys() == values[2].keys()
    for name in values[0]:
        native_repeat = float((values[0][name].float()-values[1][name].float()).abs().max())
        check(values[2][name], values[0][name], 'training', topk, tensor=name, original_repeat_max_abs=native_repeat)
    # At least one expert can be empty without changing NCCL payload dtypes.
    gate = owner.gate.forward
    def forced(x):
        logits = x.new_full((len(x), a.experts), -8.)
        logits[:, :topk] = torch.arange(topk, device=x.device, dtype=x.dtype) + 8.
        return logits
    owner.gate.forward = forced
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        x = initial[:, :1]
        controller.enabled = False; ref = owner(x)
        controller.enabled = True; y = owner(x)
        check(y, ref, 'empty_receiving_expert', topk)
    owner.gate.forward = gate

(a.output / f'rank{rank}.json').write_text(json.dumps(dict(rank=rank, ranks=world, experts=a.experts, checks=rows, optimized_calls=controller.calls), indent=2)+'\n')
dist.barrier()
dist.destroy_process_group()
print('DISTRIBUTED CONFIGURATION PASSED', rank, world, a.experts, len(rows), flush=True)
