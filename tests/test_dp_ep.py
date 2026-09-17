"""Run with two or four torchrun processes; set DP_EP_TEST_DEVICE=cuda for NCCL."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.distributed as dist

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import (
    clip_ep_grad_norm_, init_ep_dp_groups, lm_checkpoint,
    sync_ep_gradients, sync_ep_shared_parameters,
)


def main():
    use_cuda = os.environ.get('DP_EP_TEST_DEVICE') == 'cuda'
    dist.init_process_group('nccl' if use_cuda else 'gloo')
    if use_cuda:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda' if use_cuda else 'cpu')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size in (2, 4)
    ep_size = int(os.environ.get('EP_TEST_SIZE', '2'))
    parallel = init_ep_dp_groups(ep_size, 4)
    assert (parallel.ep_rank, parallel.dp_rank) == (rank % ep_size, rank // ep_size)

    torch.manual_seed(10 + rank)
    config = MiniMindConfig(
        hidden_size=32, num_hidden_layers=1, vocab_size=64,
        num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, moe_intermediate_size=64,
        max_position_embeddings=32, use_moe=True, use_ep=True,
        ep_size=ep_size, num_experts=4, num_experts_per_tok=2,
        flash_attn=False,
    )
    model = MiniMindForCausalLM(config, ep_group=parallel.ep_group).to(device)
    sync_ep_shared_parameters(model, parallel)
    experts_per_rank = 4 // ep_size
    assert model.model.layers[0].mlp.local_expert_ids == tuple(
        range(parallel.ep_rank * experts_per_rank,
              (parallel.ep_rank + 1) * experts_per_rank)
    )
    ids = (torch.arange(12, device=device).reshape(2, 6) + rank * 7) % 64
    result = model(ids, labels=ids)
    (result.loss + result.aux_loss).backward()
    sync_ep_gradients(model, parallel)
    norm = clip_ep_grad_norm_(model, 1.0, parallel)
    assert torch.isfinite(norm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()

    for name, param in model.named_parameters():
        if '.mlp.experts.' in name and parallel.dp_size > 1:
            gathered = [torch.empty_like(param) for _ in range(parallel.dp_size)]
            dist.all_gather(gathered, param.detach(), group=parallel.dp_group)
            assert torch.equal(gathered[0], gathered[1]), name
        elif name.endswith(('mlp.gate.weight', 'self_attn.q_proj.weight')):
            gathered = [torch.empty_like(param) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, param.detach())
            assert all(torch.equal(gathered[0], other) for other in gathered[1:]), name

    path = os.environ['DP_EP_TEST_DIR']
    lm_checkpoint(config, weight='test', model=model, optimizer=optimizer,
                  save_dir=path, parallel=parallel)
    resume = lm_checkpoint(config, weight='test', save_dir=path, parallel=parallel)
    assert resume['ep_size'] == ep_size and resume['dp_size'] == world_size // ep_size
    assert resume['rank'] == rank
    if rank == 0:
        full = torch.load(os.path.join(path, 'test_32_moe.pth'), map_location='cpu', weights_only=True)
        for expert_id in range(4):
            assert any(f'.mlp.experts.{expert_id}.' in key for key in full)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
