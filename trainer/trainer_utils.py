"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    local_expert_ids = {
        n.split('.mlp.experts.', 1)[1].split('.', 1)[0]
        for n, _ in model.named_parameters()
        if '.mlp.experts.' in n
    }
    local_expert_total = sum(
        p.numel()
        for n, p in model.named_parameters()
        if '.mlp.experts.' in n
    ) / 1e6
    expert = local_expert_total / max(len(local_expert_ids), 1)
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - local_expert_total - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if getattr(config, 'use_ep', False):
        global_total = base + (expert * n_routed) + (shared_expert * n_shared)
        Logger(f'Model Params/rank: {total:.2f}M-Global{global_total:.2f}M-Active{active:.2f}M')
    elif active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_ep_expert_parameter(name):
    return '.mlp.experts.' in name


@torch.no_grad()
def sync_ep_shared_parameters(model, group=None):
    """让所有 EP rank 的共享参数从 rank 0 获得完全相同的初值。"""
    if not getattr(model.config, 'use_ep', False):
        return
    group = group if group is not None else dist.group.WORLD
    for name, param in model.named_parameters():
        if not is_ep_expert_parameter(name):
            dist.broadcast(param.data, src=0, group=group)


@torch.no_grad()
def sync_ep_gradients(model, group=None):
    """同步共享参数梯度，并对只在本 rank 保存的 Expert 梯度做全局平均。"""
    if not getattr(model.config, 'use_ep', False):
        return
    group = group if group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if is_ep_expert_parameter(name):
            # 每个本地 Expert 已经通过 dispatch 收到了所有 rank 的 Token，
            # 参数梯度是各 rank loss 梯度之和，只需除以 EP world size。
            param.grad.div_(world_size)
        else:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
            param.grad.div_(world_size)


@torch.no_grad()
def clip_ep_grad_norm_(model, max_norm, group=None):
    """计算去重后的全局 EP 梯度范数，并在所有 rank 使用同一缩放系数。"""
    group = group if group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    device = next(model.parameters()).device
    local_squared_norm = torch.zeros((), dtype=torch.float32, device=device)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        squared_norm = param.grad.detach().float().pow(2).sum()
        # 共享梯度在每个 rank 都相同，求全局范数时只计算一份。
        if not is_ep_expert_parameter(name):
            squared_norm.div_(world_size)
        local_squared_norm.add_(squared_norm)
    dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM, group=group)
    total_norm = local_squared_norm.sqrt()
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(clip_coef)
    return total_norm


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        if lm_config.use_ep:
            if not dist.is_initialized():
                raise RuntimeError('EP checkpoint requires torch.distributed')
            rank, world_size = dist.get_rank(), dist.get_world_size()
            local_expert_state = {
                key: value
                for key, value in state_dict.items()
                if is_ep_expert_parameter(key)
            }
            gathered_expert_states = [None] * world_size if rank == 0 else None
            dist.gather_object(
                local_expert_state,
                gathered_expert_states,
                dst=0,
            )

            export_path = kwargs.pop('export_path', None)
            if rank == 0:
                full_state_dict = dict(state_dict)
                for expert_state in gathered_expert_states:
                    full_state_dict.update(expert_state)
                for path in (ckp_path, export_path):
                    if path:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        tmp_path = path + '.tmp'
                        torch.save(full_state_dict, tmp_path)
                        os.replace(tmp_path, path)
                del full_state_dict

            wandb_id = getattr(wandb, 'id', None) if wandb else None
            rank_resume_path = resume_path.replace(
                '_resume.pth',
                f'_ep_rank{rank}_resume.pth',
            )
            resume_data = {
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'step': step,
                'world_size': world_size,
                'wandb_id': wandb_id,
            }
            for key, value in kwargs.items():
                if value is not None:
                    resume_data[key] = value.state_dict() if hasattr(value, 'state_dict') else value
            rank_resume_tmp = rank_resume_path + '.tmp'
            torch.save(resume_data, rank_resume_tmp)
            os.replace(rank_resume_tmp, rank_resume_path)
            dist.barrier()
            del state_dict, resume_data, local_expert_state
            torch.cuda.empty_cache()
            return

        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if lm_config.use_ep:
            if not dist.is_initialized():
                raise RuntimeError('EP resume requires torch.distributed')
            rank = dist.get_rank()
            rank_resume_path = resume_path.replace(
                '_resume.pth',
                f'_ep_rank{rank}_resume.pth',
            )
            if not os.path.exists(rank_resume_path):
                return None
            ckp_data = torch.load(rank_resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size()
            if saved_ws != current_ws:
                raise RuntimeError(
                    f'EP resume要求相同world_size，checkpoint={saved_ws}, current={current_ws}'
                )
            return ckp_data
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location='cpu')
        if lm_config.use_ep:
            local_keys = set(model.state_dict().keys())
            weights = {
                key: value
                for key, value in weights.items()
                if key in local_keys
            }
            missing_keys = local_keys - set(weights.keys())
            if missing_keys:
                raise RuntimeError(
                    f'checkpoint缺少当前EP rank需要的参数: {sorted(missing_keys)[:5]}'
                )
            model.load_state_dict(weights, strict=True)
        else:
            model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    model = model.to(device)
    if lm_config.use_ep:
        sync_ep_shared_parameters(model)
    return model, tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
