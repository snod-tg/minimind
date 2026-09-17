"""Run with two torchrun processes; set EP_TEST_DEVICE=cuda for NCCL."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.distributed as dist

from model.model_minimind import (
    MiniMindConfig, MiniMindForCausalLM,
    differentiable_all_to_all, differentiable_fp8_all_to_all,
)
from trainer.trainer_utils import sync_ep_gradients


def check_ragged_exchange(device):
    rank = dist.get_rank()
    send_splits = [2, 0] if rank == 0 else [1, 1]
    send_counts = torch.tensor(send_splits, device=device)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    recv_splits = recv_counts.tolist()
    values = torch.tensor(
        [[0.25, -1.0, 3.0], [2.5, 0.125, -4.0]] if rank == 0
        else [[-0.5, 8.0, 1.0], [1.25, -2.0, 0.0625]],
        device=device,
    )
    native_x = values.clone().requires_grad_()
    fp8_x = values.clone().requires_grad_()
    native = differentiable_all_to_all(native_x, recv_splits, send_splits, dist.group.WORLD)
    compressed = differentiable_fp8_all_to_all(fp8_x, recv_splits, send_splits, dist.group.WORLD)
    torch.testing.assert_close(compressed, native, rtol=0.07, atol=0.02)
    native.square().sum().backward()
    compressed.square().sum().backward()
    torch.testing.assert_close(fp8_x.grad, native_x.grad, rtol=0.14, atol=0.05)


def check_model_step(device, comm_dtype):
    config = MiniMindConfig(
        hidden_size=32, num_hidden_layers=1, vocab_size=64,
        num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, moe_intermediate_size=64,
        max_position_embeddings=32, use_moe=True, use_ep=True,
        ep_size=2, num_experts=4, num_experts_per_tok=2,
        ep_comm_dtype=comm_dtype, flash_attn=False,
    )
    torch.manual_seed(10 + dist.get_rank())
    model = MiniMindForCausalLM(config).to(device)
    ids = (torch.arange(12, device=device).reshape(2, 6) + dist.get_rank() * 7) % 64
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        result = model(ids, labels=ids)
        loss = result.loss + result.aux_loss
    assert torch.isfinite(loss)
    loss.backward()
    sync_ep_gradients(model)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    return loss.item()


def main():
    use_cuda = os.environ.get('EP_TEST_DEVICE') == 'cuda'
    dist.init_process_group('nccl' if use_cuda else 'gloo')
    assert dist.get_world_size() == 2
    if use_cuda:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda' if use_cuda else 'cpu')
    check_ragged_exchange(device)
    native_loss = check_model_step(device, 'native')
    fp8_loss = check_model_step(device, 'fp8')
    assert abs(fp8_loss - native_loss) < 0.05
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
