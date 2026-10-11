import math, torch, torch.distributed as dist, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        self.use_ep = kwargs.get("use_ep", False)
        self.ep_size = kwargs.get("ep_size", None)
        self.ep_comm_dtype = kwargs.get("ep_comm_dtype", "native")

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0: # 向外扩展
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base)) # 计算“高频”和“低频”的维度分界线
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1) # 生成平滑过渡的斜坡（ramp）
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)


def _all_to_all_single(x, output_splits, input_splits, group):
    """执行一次支持不等长 split 的 All-to-All。"""
    output = x.new_empty((sum(output_splits), *x.shape[1:]))
    dist.all_to_all_single(
        output,
        x.contiguous(),
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )
    return output


class _AllToAll(torch.autograd.Function):
    """可求导 All-to-All：反向就是交换输入、输出 split 后再通信一次。"""

    @staticmethod
    def forward(ctx, x, output_splits, input_splits, group):
        ctx.output_splits = tuple(output_splits)
        ctx.input_splits = tuple(input_splits)
        ctx.group = group
        return _all_to_all_single(
            x,
            ctx.output_splits,
            ctx.input_splits,
            group,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _all_to_all_single(
            grad_output,
            ctx.input_splits,
            ctx.output_splits,
            ctx.group,
        )
        return grad_input, None, None, None


def differentiable_all_to_all(x, output_splits, input_splits, group):
    return _AllToAll.apply(x, output_splits, input_splits, group)


def _fp8_all_to_all_single(x, output_splits, input_splits, group):
    """Communicate E4M3 payloads as uint8, with one FP32 scale per peer."""
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    send_chunks = []
    send_scales = []
    for chunk in x.split(input_splits, dim=0):
        if chunk.numel():
            chunk_fp32 = chunk.float()
            scale = (chunk_fp32.abs().amax() / fp8_max).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            bits = (chunk_fp32 / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype).view(torch.uint8)
        else:
            scale = torch.ones((), dtype=torch.float32, device=x.device)
            bits = chunk.to(fp8_dtype).view(torch.uint8)
        send_chunks.append(bits)
        send_scales.append(scale)

    send_scales = torch.stack(send_scales)
    recv_scales = torch.empty_like(send_scales)
    dist.all_to_all_single(recv_scales, send_scales, group=group)
    recv_bits = _all_to_all_single(
        torch.cat(send_chunks, dim=0), output_splits, input_splits, group
    )
    recv_chunks = recv_bits.view(fp8_dtype).float().split(output_splits, dim=0)
    return torch.cat(
        [chunk * scale for chunk, scale in zip(recv_chunks, recv_scales)], dim=0
    ).to(x.dtype)


class _FP8AllToAll(torch.autograd.Function):
    """FP8 all-to-all with a straight-through backward, also sent as FP8."""

    @staticmethod
    def forward(ctx, x, output_splits, input_splits, group):
        ctx.output_splits = tuple(output_splits)
        ctx.input_splits = tuple(input_splits)
        ctx.group = group
        return _fp8_all_to_all_single(x, ctx.output_splits, ctx.input_splits, group)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _fp8_all_to_all_single(
            grad_output, ctx.input_splits, ctx.output_splits, ctx.group
        )
        return grad_input, None, None, None


def differentiable_fp8_all_to_all(x, output_splits, input_splits, group):
    return _FP8AllToAll.apply(x, output_splits, input_splits, group)


class ExpertParallelMOEFeedForward(nn.Module):
    """支持训练与推理的 Top-K Expert Parallel MoE。"""

    def __init__(self, config: MiniMindConfig, process_group=None):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Expert Parallel requires torch.distributed to be initialized "
                "before constructing the model"
            )
        if not 1 <= config.num_experts_per_tok <= config.num_experts:
            raise ValueError(
                "num_experts_per_tok must be between 1 and num_experts"
            )
        if config.ep_comm_dtype not in ("native", "fp8"):
            raise ValueError("ep_comm_dtype must be 'native' or 'fp8'")

        self.config = config
        self.communicate = (
            differentiable_fp8_all_to_all
            if config.ep_comm_dtype == "fp8" else differentiable_all_to_all
        )
        self.process_group = (
            process_group if process_group is not None else dist.group.WORLD
        )
        self.ep_rank = dist.get_rank(self.process_group)
        self.ep_size = dist.get_world_size(self.process_group)

        if config.ep_size is not None and config.ep_size != self.ep_size:
            raise ValueError(
                f"config.ep_size={config.ep_size}, but process group size={self.ep_size}"
            )
        if config.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts={config.num_experts} must be divisible by "
                f"ep_size={self.ep_size}"
            )

        self.experts_per_rank = config.num_experts // self.ep_size
        self.first_expert_id = self.ep_rank * self.experts_per_rank
        self.local_expert_ids = tuple(
            range(
                self.first_expert_id,
                self.first_expert_id + self.experts_per_rank,
            )
        )

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # Global IDs are used as keys so sharded state_dict names retain their
        # global Expert identity (for example experts.2.* on rank 1).
        self.experts = nn.ModuleDict({
            str(global_id): FeedForward(
                config,
                intermediate_size=config.moe_intermediate_size,
            )
            for global_id in self.local_expert_ids
        })
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)
        num_tokens = x_flat.shape[0]

        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_expert_id = torch.topk(
            scores,
            k=self.config.num_experts_per_tok,
            dim=-1,
            sorted=False,
        )
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (
                topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            )

        # Top-K turns every Token into K independent routing assignments.
        # assignment_token_indices remembers which original Token each returned
        # Expert output must be accumulated into.
        assignment_token_indices = torch.arange(
            num_tokens,
            device=x_flat.device,
        )[:, None].expand(-1, self.config.num_experts_per_tok).reshape(-1)
        assignment_weight = topk_weight.reshape(-1)
        global_expert_id = topk_expert_id.reshape(-1)
        assignment_x = x_flat[assignment_token_indices]
        num_assignments = assignment_x.shape[0]

        target_rank = global_expert_id // self.experts_per_rank
        local_expert_id = global_expert_id % self.experts_per_rank

        rank_sort_indices = torch.argsort(target_rank, stable=True)
        send_x = assignment_x[rank_sort_indices]
        send_local_expert_id = local_expert_id[rank_sort_indices]

        send_counts = torch.bincount(
            target_rank,
            minlength=self.ep_size,
        )
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(
            recv_counts,
            send_counts,
            group=self.process_group,
        )
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        # dispatch：把 Token 发送到持有目标 Expert 的 rank。该通信参与 autograd，
        # backward 会执行反向 All-to-All，把输入梯度送回 Token 的来源 rank。
        recv_x = self.communicate(
            send_x,
            recv_splits,
            send_splits,
            self.process_group,
        )
        recv_local_expert_id = torch.empty(
            sum(recv_splits),
            dtype=send_local_expert_id.dtype,
            device=send_local_expert_id.device,
        )
        dist.all_to_all_single(
            recv_local_expert_id,
            send_local_expert_id.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.process_group,
        )

        expert_sort_indices = torch.argsort(
            recv_local_expert_id,
            stable=True,
        )
        expert_sorted_x = recv_x[expert_sort_indices]
        local_expert_counts = torch.bincount(
            recv_local_expert_id,
            minlength=self.experts_per_rank,
        ).tolist()

        expert_outputs = []
        offset = 0
        for local_id, count in enumerate(local_expert_counts):
            end = offset + count
            if count > 0:
                global_id = self.first_expert_id + local_id
                expert_outputs.append(
                    self.experts[str(global_id)](expert_sorted_x[offset:end])
                )
            offset = end

        if expert_outputs:
            expert_sorted_output = torch.cat(expert_outputs, dim=0)
        else:
            # Keep a correctly shaped empty output when this rank receives no Token.
            expert_sorted_output = expert_sorted_x.clone()

        inverse_expert_indices = torch.empty_like(expert_sort_indices)
        inverse_expert_indices[expert_sort_indices] = torch.arange(
            expert_sort_indices.numel(),
            device=expert_sort_indices.device,
        )
        recv_order_output = expert_sorted_output[inverse_expert_indices]

        # combine：把 Expert 输出送回 Token 来源 rank。它的 backward 同样是
        # 一次方向相反的 All-to-All，把输出梯度送回 Expert 所在 rank。
        returned_sorted_output = self.communicate(
            recv_order_output,
            send_splits,
            recv_splits,
            self.process_group,
        )
        inverse_rank_indices = torch.empty_like(rank_sort_indices)
        inverse_rank_indices[rank_sort_indices] = torch.arange(
            num_assignments,
            device=rank_sort_indices.device,
        )
        returned_assignment_output = returned_sorted_output[inverse_rank_indices]
        weighted_assignment_output = (
            returned_assignment_output * assignment_weight[:, None]
        ).to(x_flat.dtype)
        y_flat = torch.zeros_like(x_flat)
        y_flat.index_add_(
            0,
            assignment_token_indices,
            weighted_assignment_output,
        )

        if self.training and self.config.router_aux_loss_coef > 0:
            # top-k 选择本身不可导；负载统计只作为 gate 概率的平衡目标。
            load = F.one_hot(
                topk_expert_id,
                self.config.num_experts,
            ).float().mean(dim=(0, 1))
            if self.ep_size > 1:
                dist.all_reduce(load, op=dist.ReduceOp.SUM, group=self.process_group)
                load.div_(self.ep_size)
            self.aux_loss = (
                load * scores.mean(dim=0)
            ).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(())

        return y_flat.reshape(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not config.use_moe:
            self.mlp = FeedForward(config)
        elif config.use_ep:
            self.mlp = ExpertParallelMOEFeedForward(config)
        else:
            self.mlp = MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([
            l.mlp.aux_loss
            for l in self.layers
            if isinstance(l.mlp, (MOEFeedForward, ExpertParallelMOEFeedForward))
        ], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
