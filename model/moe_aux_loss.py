"""Explicit auxiliary-loss contract for MoE/EP comparison."""
import torch.distributed as dist
import torch.nn.functional as F


def moe_aux_loss(scores, expert_ids, config, process_group=None):
    mode = getattr(config, 'router_aux_loss_mode', 'legacy')
    if mode == 'local_topk_sum' or process_group is None:
        # Preserve the original non-EP objective, including the Top-K sum.
        load = F.one_hot(expert_ids, config.num_experts).float().mean(0)
    else:
        # Existing EP default: global load, averaged across Top-K slots.
        load = F.one_hot(expert_ids, config.num_experts).float().mean(dim=(0, 1))
        dist.all_reduce(load, group=process_group)
        load.div_(dist.get_world_size(process_group))
    return (load * scores.mean(0)).sum() * config.num_experts * config.router_aux_loss_coef
