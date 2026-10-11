# MiniMind Expert Parallel（EP）实现说明

> 作者：tongleiwen  
> 项目：在 MiniMind 原生 PyTorch MoE 上实现可训练、可推理的 Expert Parallel  
> 当前状态：已完成 Top-K 路由、动态不等长 All-to-All、反向传播、训练同步、Checkpoint 和多卡推理闭环

## 1. 项目概述

MiniMind 原有的 `MOEFeedForward` 会在每个进程中创建全部 Expert。即使一个 Token 只激活少量 Expert，每张 GPU 仍然保存所有 Expert 参数。Expert 数量继续增加时，单卡显存会成为主要限制。

我在原有 MoE 的基础上新增了 Expert Parallel（EP）：

- Router、Attention、Embedding、RMSNorm 和 LM Head 等共享参数仍在每个 Rank 保留一份。
- 全局 Expert 被平均切分到多个 Rank，每个 Rank 只保存自己的本地 Expert。
- Router 仍对全部全局 Expert 做 Top-K 选择。
- Token 通过动态、不等长 All-to-All 被发送到目标 Expert 所在的 Rank。
- 本地 Rank 按 Expert 对 Token 分组并执行 Expert 前向。
- Expert 输出通过第二次 All-to-All 返回 Token 的来源 Rank，再恢复原始顺序。
- 自定义可求导 All-to-All，使 EP 可以参与完整反向传播。
- 训练阶段手动同步共享参数梯度，并正确处理本地 Expert 梯度。
- 推理阶段通过 `eval_llm.py --use_ep` 启动多卡 EP。

这份实现以正确性、可读性和完整闭环为优先目标，不依赖 DeepEP、Megatron-LM 或 vLLM 的 fused EP kernel。

## 2. 已实现的能力

| 能力 | 状态 | 说明 |
|---|---|---|
| Expert 参数分片 | ✅ | 每个 Rank 只创建 `num_experts / ep_size` 个 Expert |
| 全局 Top-K Router | ✅ | 支持 Top-1 和参数化 Top-K |
| 动态不等长 All-to-All | ✅ | 每轮根据实际路由结果动态计算 split sizes |
| Token dispatch/combine | ✅ | Token 发往 Expert Rank，结果返回来源 Rank |
| 原 Token 顺序恢复 | ✅ | 两级 inverse indices 恢复 Expert 排序和 Rank 排序 |
| EP 反向传播 | ✅ | 自定义 autograd Function 执行反向 All-to-All |
| Router 辅助损失 | ✅ | 跨 Rank 汇总负载，保留负载均衡训练目标 |
| 多卡预训练 | ✅ | 已接入 `train_pretrain.py` |
| 多卡 Full SFT | ✅ | 已接入 `train_full_sft.py` |
| EP Checkpoint | ✅ | 完整权重合并保存，并支持逐 Rank 断点续训 |
| 多卡生成推理 | ✅ | 已接入 `eval_llm.py --use_ep` |
| Top-1 / Top-2 backward | ✅ | Gloo 与 NCCL 均已验证 |

## 3. 整体设计

下面以 2 个 Rank、4 个全局 Expert 为例：

| Rank | 实际保存的全局 Expert | 本地 Expert ID |
|---|---|---|
| Rank 0 / GPU 0 | Expert 0、Expert 1 | 0、1 |
| Rank 1 / GPU 1 | Expert 2、Expert 3 | 0、1 |

全局 Expert 与目标 Rank 的映射关系为：

```python
experts_per_rank = num_experts // ep_size
target_rank = global_expert_id // experts_per_rank
local_expert_id = global_expert_id % experts_per_rank

# 在目标 Rank 上恢复全局 Expert ID
global_expert_id = rank * experts_per_rank + local_expert_id
```

完整数据流如下：

```text
+--------------------------------------------------------------------------+
| 1. SOURCE RANK: Router                                                   |
+--------------------------------------------------------------------------+
| x [batch, seq_len, hidden_size]                                          |
|             |                                                            |
|             v  reshape                                                   |
| x_flat [num_tokens, hidden_size]                                         |
|             |                                                            |
|             v  global Router + softmax + top-k                           |
| topk_expert_id [num_tokens, top_k]                                       |
| topk_weight    [num_tokens, top_k]                                       |
|             |                                                            |
|             v  expand each Token into top_k assignments                  |
| assignment_x / global_expert_id / assignment_weight                      |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
| 2. SOURCE RANK: Prepare Dispatch                                         |
+--------------------------------------------------------------------------+
| global_expert_id                                                         |
|        |                                                                 |
|        +--> target_rank    = global_expert_id // experts_per_rank        |
|        +--> local_expert_id = global_expert_id %  experts_per_rank       |
|                                                                          |
| stable argsort(target_rank)                                              |
|        |                                                                 |
|        +--> send_x:               [to Rank 0][to Rank 1] ...             |
|        +--> send_local_expert_id: [to Rank 0][to Rank 1] ...             |
|        +--> keep rank_sort_indices on the source Rank                    |
|                                                                          |
| bincount(target_rank) -> send_splits                                     |
| exchange send_counts    -> recv_splits                                   |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
| 3. DISPATCH: Dynamic Uneven All-to-All                                   |
+--------------------------------------------------------------------------+
|                                                                          |
|  Rank 0 send buffer                         Rank 1 send buffer            |
|  +--------------+--------------+            +--------------+-----------+ |
|  | for Rank 0   | for Rank 1   |            | for Rank 0   | for Rank 1| |
|  +------+-------+------+-------+            +------+-------+------+----+ |
|         |              |                           |              |       |
|         +--------------+------ All-to-All ---------+--------------+       |
|                        |                                                 |
|              +---------+-------------------+                             |
|              v                             v                             |
|  Rank 0 receives Tokens for E0/E1   Rank 1 receives Tokens for E2/E3     |
|                                                                          |
|  hidden states: differentiable_all_to_all                               |
|  local Expert IDs: ordinary dist.all_to_all_single                      |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
| 4. TARGET RANK: Local Expert Computation                                 |
+--------------------------------------------------------------------------+
| recv_x + recv_local_expert_id                                            |
|             |                                                            |
|             v  stable argsort(local_expert_id)                           |
| expert_sorted_x: [Expert 0 Tokens][Expert 1 Tokens] ...                  |
|             |                                                            |
|             +--> local Expert 0 --> output 0                             |
|             +--> local Expert 1 --> output 1                             |
|             +--> ...                                                     |
|             |                                                            |
|             v  concatenate Expert outputs                               |
| expert_sorted_output                                                     |
|             |                                                            |
|             v  inverse_expert_indices                                    |
| recv_order_output: restore the order received from source Ranks          |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
| 5. COMBINE: Dynamic Uneven All-to-All                                    |
+--------------------------------------------------------------------------+
| Expert outputs are sent back to the Rank where each Token originated     |
|                                                                          |
| recv_order_output                                                        |
|             |                                                            |
|             v  differentiable_all_to_all (reverse direction)            |
| returned_sorted_output                                                   |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
| 6. SOURCE RANK: Restore and Merge                                        |
+--------------------------------------------------------------------------+
| returned_sorted_output                                                   |
|             |                                                            |
|             v  inverse_rank_indices                                      |
| returned_assignment_output: restore original assignment order           |
|             |                                                            |
|             v  multiply assignment_weight                               |
| weighted_assignment_output                                               |
|             |                                                            |
|             v  index_add by assignment_token_indices                     |
| y_flat [num_tokens, hidden_size]                                         |
|             |                                                            |
|             v  reshape                                                   |
| y [batch, seq_len, hidden_size]                                          |
+--------------------------------------------------------------------------+
```

需要始终保留的三类索引分别是：

- `assignment_token_indices`：Top-K 的每条 assignment 属于哪个原始 Token。
- `inverse_expert_indices`：撤销目标 Rank 内按 `local_expert_id` 进行的排序。
- `inverse_rank_indices`：撤销来源 Rank 内按 `target_rank` 进行的排序。

## 4. 我修改了哪些文件

| 文件 | EP 相关改动 |
|---|---|
| [`model/model_minimind.py`](./model/model_minimind.py) | EP 配置、Expert 分片、Top-K 路由、dispatch/combine、可求导 All-to-All、负载均衡损失 |
| [`trainer/trainer_utils.py`](./trainer/trainer_utils.py) | 共享参数初始化、EP 梯度同步、全局梯度裁剪、完整权重合并和逐 Rank resume |
| [`trainer/train_pretrain.py`](./trainer/train_pretrain.py) | `--use_ep` 预训练入口和 EP 训练流程 |
| [`trainer/train_full_sft.py`](./trainer/train_full_sft.py) | `--use_ep` Full SFT 入口和 EP 训练流程 |
| [`eval_llm.py`](./eval_llm.py) | `--use_ep` 推理入口、分布式初始化、分片加载和多 Rank 同步生成 |

## 5. EP Forward 是如何实现的

### 5.1 新增配置

我在 `MiniMindConfig` 中加入：

```python
self.use_ep = kwargs.get("use_ep", False)
self.ep_size = kwargs.get("ep_size", None)
```

EP 还会使用原 MoE 已有的配置：

```python
self.num_experts
self.num_experts_per_tok
self.norm_topk_prob
self.router_aux_loss_coef
```

当前必须满足：

```text
1 <= num_experts_per_tok <= num_experts
num_experts % ep_size == 0
```

TransformerBlock 会根据配置选择不同的 FFN：

```python
if not config.use_moe:
    self.mlp = FeedForward(config)
elif config.use_ep:
    self.mlp = ExpertParallelMOEFeedForward(config)
else:
    self.mlp = MOEFeedForward(config)
```

因此原来的 Dense 和普通 MoE 路径仍然保留，只有显式开启 `use_ep` 才进入 EP。

### 5.2 每个 Rank 只创建本地 Expert

每个 Rank 先计算自己负责的全局 Expert 范围：

```python
self.experts_per_rank = config.num_experts // self.ep_size
self.first_expert_id = self.ep_rank * self.experts_per_rank
self.local_expert_ids = range(
    self.first_expert_id,
    self.first_expert_id + self.experts_per_rank,
)
```

本地 Expert 使用 `ModuleDict` 保存，并把全局 Expert ID 作为 key：

```python
self.experts = nn.ModuleDict({
    str(global_id): FeedForward(config)
    for global_id in self.local_expert_ids
})
```

这样做可以让 EP 权重名称继续保留全局身份。例如 Rank 1 保存的是 `experts.2.*` 和 `experts.3.*`，多个 Rank 合并后仍能直接兼容普通 MoE checkpoint。

### 5.3 Router 生成 Top-K assignment

输入先从 `[batch, seq_len, hidden_size]` 展平为 `[num_tokens, hidden_size]`：

```python
x_flat = x.reshape(-1, hidden_size)
scores = F.softmax(self.gate(x_flat), dim=-1)
topk_weight, topk_expert_id = torch.topk(
    scores,
    k=self.config.num_experts_per_tok,
    dim=-1,
)
```

当 Top-K 大于 1 时，一个 Token 会产生 K 条 assignment：

```text
num_assignments = num_tokens * top_k
```

我额外保存 `assignment_token_indices`，记录每条 assignment 来自哪个原始 Token。Expert 结果返回后，依靠它把 K 路输出累加回原 Token。

### 5.4 动态计算通信数量

根据全局 Expert ID 得到目标 Rank：

```python
target_rank = global_expert_id // experts_per_rank
local_expert_id = global_expert_id % experts_per_rank
```

All-to-All 的发送缓冲区需要让发往同一 Rank 的 Token 连续，因此先稳定排序：

```python
rank_sort_indices = torch.argsort(target_rank, stable=True)
send_x = assignment_x[rank_sort_indices]
send_local_expert_id = local_expert_id[rank_sort_indices]
```

每个 Rank 实际收到的 Token 数量由 Router 决定，通常并不相等。我使用 `bincount` 动态计算：

```python
send_counts = torch.bincount(target_rank, minlength=ep_size)
dist.all_to_all_single(recv_counts, send_counts)

send_splits = send_counts.tolist()
recv_splits = recv_counts.tolist()
```

这就是“不等长 All-to-All”：

- `input_split_sizes` 表示当前 Rank 分别向各目标 Rank 发送多少行。
- `output_split_sizes` 表示当前 Rank 分别从各来源 Rank 接收多少行。

### 5.5 Dispatch：把 Token 发给目标 Expert

隐藏状态使用可求导 All-to-All：

```python
recv_x = differentiable_all_to_all(
    send_x,
    output_splits=recv_splits,
    input_splits=send_splits,
    group=self.process_group,
)
```

`local_expert_id` 是整数路由元数据，不需要梯度，因此使用普通 All-to-All 一并发送：

```python
dist.all_to_all_single(
    recv_local_expert_id,
    send_local_expert_id,
    output_split_sizes=recv_splits,
    input_split_sizes=send_splits,
)
```

我没有通信 `sort_indices`。它只需要保留在 Token 的来源 Rank，等 Expert 输出回来后再用于恢复顺序；目标 Rank 只需要收到隐藏状态和 `local_expert_id`。

### 5.6 在目标 Rank 内按 Expert 分组

经过第一次 All-to-All 后，Token 是按“来源 Rank”分段的，还没有按本地 Expert 连续排列。因此必须再按 `local_expert_id` 排序：

```python
expert_sort_indices = torch.argsort(
    recv_local_expert_id,
    stable=True,
)
expert_sorted_x = recv_x[expert_sort_indices]
local_expert_counts = torch.bincount(
    recv_local_expert_id,
    minlength=experts_per_rank,
)
```

相同 Expert 的 Token 变为连续内存片段后，再一次性送入对应 Expert：

```python
for local_id, count in enumerate(local_expert_counts):
    global_id = first_expert_id + local_id
    expert_output = self.experts[str(global_id)](expert_input)
```

这里不能让整个 Rank 的 Token 直接进入同一个 Expert，因为一个 Rank 通常保存多个 Expert，`local_expert_id` 才能确定每个 Token 应由哪个本地 Expert 处理。

### 5.7 Combine：返回来源 Rank 并恢复顺序

Expert 计算结束后，顺序已经发生两次变化：

1. 来源 Rank 按 `target_rank` 做过一次排序。
2. 目标 Rank 按 `local_expert_id` 做过一次排序。

因此需要按相反顺序撤销：

```python
# 先撤销目标 Rank 内的 Expert 排序
recv_order_output = expert_sorted_output[inverse_expert_indices]

# 再通过 All-to-All 把输出发回 Token 来源 Rank
returned_sorted_output = differentiable_all_to_all(...)

# 最后撤销来源 Rank 上的 target_rank 排序
returned_assignment_output = returned_sorted_output[inverse_rank_indices]
```

不能直接 `reshape` 恢复，因为 `reshape` 只改变张量形状，不会恢复被排序打乱的数据语义。

最后乘 Router 权重，并把同一 Token 的 Top-K 输出累加起来：

```python
weighted_output = (
    returned_assignment_output * assignment_weight[:, None]
).to(x_flat.dtype)

y_flat = torch.zeros_like(x_flat)
y_flat.index_add_(
    0,
    assignment_token_indices,
    weighted_output,
)
```

## 6. EP Backward 是如何实现的

普通的 `dist.all_to_all_single` 不会自动建立这里需要的反向通信。我封装了 `_AllToAll(torch.autograd.Function)`：

```python
class _AllToAll(torch.autograd.Function):
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
```

关键点是：反向传播交换 forward 的输入、输出 split，再做一次方向相反的 All-to-All。

```text
dispatch.forward 的反向 = combine
combine.forward 的反向  = dispatch
```

因此训练时隐藏状态实际经历四次通信：forward 的 dispatch/combine，以及 backward 中对应的两次反向通信。

## 7. 为什么 EP 训练没有直接套普通 DDP

普通 DDP 假设不同 Rank 拥有相同名称、数量和顺序的参数，但 EP 中不同 Rank 保存的是不同 Expert：

```text
Rank 0: experts.0.*, experts.1.*
Rank 1: experts.2.*, experts.3.*
```

直接使用普通 DDP，既可能因为参数结构不一致失败，也可能错误地把不同 Expert 的梯度相加。因此 EP 训练使用手动同步：

### 7.1 共享参数初始化

模型创建后，把 Rank 0 的非 Expert 参数广播到所有 Rank，保证 Router、Attention、Embedding 等共享参数初值一致：

```python
if not is_ep_expert_parameter(name):
    dist.broadcast(param.data, src=0)
```

### 7.2 梯度同步

- 共享参数：对梯度执行 All-Reduce 并除以 `world_size`。
- 本地 Expert：不能与其他 Rank 的不同 Expert 做 All-Reduce；其梯度只除以 `world_size`，保持与全局平均 loss 相同的尺度。

```python
if is_ep_expert_parameter(name):
    param.grad.div_(world_size)
else:
    dist.all_reduce(param.grad)
    param.grad.div_(world_size)
```

### 7.3 全局梯度裁剪

`clip_ep_grad_norm_()` 会计算去重后的全局范数：

- 本地 Expert 梯度在各 Rank 求和。
- 共享参数在所有 Rank 都有副本，计算全局范数时只计一次。
- 所有 Rank 使用相同的裁剪系数。

## 8. Router 负载均衡损失

EP 保留了原 MoE 的 Router auxiliary loss。先统计各 Expert 的 Top-K 路由比例，再跨 Rank 做 All-Reduce，得到全局负载分布：

```python
load = F.one_hot(
    topk_expert_id,
    num_experts,
).float().mean(dim=(0, 1))

dist.all_reduce(load)
load /= ep_size

aux_loss = (
    load * scores.mean(dim=0)
).sum() * num_experts * router_aux_loss_coef
```

Top-K 的离散 Expert ID 不参与梯度，梯度通过连续的 `scores` 返回 Router gate。

## 9. Checkpoint 如何保存和恢复

### 9.1 完整模型权重

每个 Rank 的 `state_dict` 只包含共享参数和本地 Expert shard。保存时：

1. 每个 Rank 提取自己的 Expert 参数。
2. 使用 `dist.gather_object` 把 Expert shard 发给 Rank 0。
3. Rank 0 合并全部全局 Expert。
4. 写出普通 MoE 也可以加载的完整权重：`{weight}_{hidden_size}_moe.pth`。

因此同一份完整 checkpoint 可以用于：

- 单卡普通 MoE 推理。
- 多卡 EP 推理。
- 从普通 MoE 切换到 EP 训练。

### 9.2 EP 断点续训

Optimizer state 与参数 shard 绑定，所以每个 Rank 分别保存：

```text
{weight}_{hidden_size}_moe_ep_rank0_resume.pth
{weight}_{hidden_size}_moe_ep_rank1_resume.pth
...
```

逐 Rank resume 包含模型 shard、optimizer、scaler、epoch、step 和保存时的 `world_size`。当前恢复训练要求 `world_size` 与保存时一致。

## 10. 推理如何实现

`eval_llm.py` 新增了：

```text
--use_ep
--num_experts
--num_experts_per_tok
```

开启 `--use_ep` 后会自动开启 MoE，并要求通过 `torchrun` 启动。CUDA 使用 NCCL，CPU 调试使用 Gloo。

完整 checkpoint 会先加载到 CPU，然后每个 Rank 只筛选共享参数和自己负责的 Expert，再把本地模型放入 GPU。

交互式生成只由 Rank 0 读取 Prompt 和打印结果，但 Prompt 与随机种子会广播到所有 Rank。所有 Rank 必须同步执行每一层 MoE 的 collective，否则会发生通信死锁。

## 11. 使用方法

以下命令假设当前目录是 MiniMind 项目根目录。

### 11.1 双卡 EP 推理

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  eval_llm.py \
  --use_ep \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --weight pretrain \
  --save_dir out
```

`--use_ep` 会自动打开 MoE，不需要再传 `--use_moe 1`。

### 11.2 从普通 MoE 权重开始 EP 预训练

```bash
cd trainer

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  train_pretrain.py \
  --use_ep \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --from_weight pretrain \
  --from_resume 0 \
  --save_weight pretrain_ep \
  --batch_size 32 \
  --use_compile 0
```

首次切换到 EP 时应从完整 MoE 权重加载，并使用新的 `save_weight`，避免覆盖原权重。

### 11.3 EP 断点续训

```bash
cd trainer

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  train_pretrain.py \
  --use_ep \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --from_resume 1 \
  --save_weight pretrain_ep \
  --batch_size 32 \
  --use_compile 0
```

恢复时应保持相同的 `WORLD_SIZE`、`num_experts` 和模型结构。

### 11.4 双卡 EP Full SFT

```bash
cd trainer

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  train_full_sft.py \
  --use_ep \
  --num_experts 4 \
  --num_experts_per_tok 1 \
  --from_weight pretrain \
  --from_resume 0 \
  --save_weight full_sft_ep \
  --use_compile 0
```

### 11.5 FP8 通信压缩

预训练、Full SFT 和 `eval_llm.py` 的 EP 模式可加入 `--ep_comm_dtype fp8`。
默认 `native` 保留原有通信精度。FP8 模式把 dispatch/combine 的隐藏状态按
目标 Rank 分段，以每段的绝对最大值计算 FP32 缩放系数，再编码为 E4M3。
E4M3 位模式以 `uint8` 执行 All-to-All，接收后反量化回输入 dtype；反向传播的
隐藏状态通信也使用相同方式。Token 数量和 Expert ID 等路由元数据不压缩。

```bash
cd trainer

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train_pretrain.py --use_ep --ep_comm_dtype fp8 \
  --num_experts 4 --from_weight pretrain --use_compile 0
```

FP8 是有损压缩，反向传播对量化采用直通近似；输出和梯度不会与原精度逐位相同。
每次通信还会交换少量 FP32 缩放系数，因此小 batch 不一定更快。正式训练应先
比较 loss 与吞吐。模型权重和 checkpoint 格式不变。

## 12. 验证结果

### 12.1 普通 MoE 与 EP 输出一致

在默认 `native` 通信模式下，使用相同权重、相同 Prompt 和相同解码设置对比普通 MoE 与双卡 EP：

- 多个 Prompt 的输出文本哈希完全一致。
- 说明 Router、dispatch、本地 Expert、combine、inverse reorder 和权重合并正确。

双卡不会让同一 checkpoint 的模型质量自动提高。EP 改变的是 Expert 参数的放置和计算位置；在数学上，它应与未分片的普通 MoE 保持一致。

### 12.2 Forward / Backward 验证

以下项目均已通过双进程 Gloo 和双卡 NCCL 测试：

| 测试项 | 结果 |
|---|---|
| Top-1 forward/backward | 通过 |
| Top-2 forward/backward | 通过 |
| 本地 Expert 梯度非空且有限 | 通过 |
| 共享参数同步后梯度一致 | 通过 |
| 全局梯度范数与裁剪 | 通过 |
| Optimizer step 更新参数 | 通过 |
| 完整 checkpoint 合并全部 Expert | 通过 |
| 各 Rank model/optimizer/scaler resume | 通过 |

### 12.3 BF16 多 Epoch 训练

在双 RTX 4080 SUPER、PyTorch 2.10.0+cu128 环境下，Tiny Top-2 模型连续训练 5 个 epoch：

| Epoch | Loss | Aux loss |
|---:|---:|---:|
| 1 | 8.8426 | 0.0010 |
| 2 | 8.1929 | 0.0010 |
| 3 | 7.7300 | 0.0011 |
| 4 | 7.6172 | 0.0010 |
| 5 | 7.4878 | 0.0010 |

Loss 持续下降，Router auxiliary loss 保持稳定，说明 forward、backward、梯度同步与 optimizer step 已经形成完整训练闭环。

### 12.4 真实模型 Smoke Test

真实 `pretrain_768_moe.pth`（8 层、4 Expert、Top-1）连续运行 3 个 epoch、共 6 个 optimizer step：

- 无 NaN。
- 无 NCCL 错误。
- 无通信死锁。
- Router auxiliary loss 稳定在约 `0.0040`。
- 完整权重保存和逐 Rank resume 正常。

双 Rank 参数统计结果：

```text
Model Params/rank: 108.76M-Global198.42M-Active63.94M
```

这说明每个 Rank 只保存自己的 Expert shard；共享参数仍会在每个 Rank 复制，所以“每卡总参数量”不会简单降为全局参数量的二分之一。

### 12.5 性能结论

当前验证的核心结论是功能正确和可训练，而不是宣称 EP 一定比单卡更快：

- EP 能把 Expert 参数分片到多张 GPU，从而扩大可容纳的 Expert 数量和模型容量。
- 同一权重下，EP 不会自动提升准确率或生成质量。
- 小模型、小 batch 或逐 Token decode 时，All-to-All 通信可能大于 Expert 计算收益，双卡 EP 甚至可能更慢。
- 当 Expert 参数很大、batch/Token 数量足够且通信占比可控时，EP 的显存扩展价值才更明显。

## 13. 当前限制

1. 当前直接使用 `dist.group.WORLD`，还没有拆分独立的 EP group 与 DP group。
2. Expert 必须平均分配，`num_experts` 必须能被 `WORLD_SIZE` 整除。
3. 没有实现 Expert capacity、Token dropping 或溢出策略。
4. Rank/Expert 分组使用 `argsort`、`bincount` 和逐 Expert Python 循环。
5. Expert 计算尚未使用 grouped GEMM 或 fused kernel。
6. 通信和 Expert 计算尚未通过独立 CUDA stream 重叠。
7. FP8 压缩是可选的 E4M3 通信路径；尚无误差反馈或 FP8 fused kernel，默认保持原精度。
8. 动态 All-to-All 训练当前不支持 `torch.compile`。
9. 加载时每个 Rank 的 CPU 内存仍会暂时读取完整 checkpoint。
10. EP resume 暂不支持改变 `WORLD_SIZE` 后自动重分片 optimizer state。
11. 当前只接入预训练和 Full SFT，DPO、PPO、GRPO 等训练器尚未接入。
12. EP 只切分 Expert；Attention、Embedding、LM Head 等共享参数仍完整复制。

## 14. 常见问题

### 开 EP 后为什么没有自动变得更准？

因为 EP 是并行执行和参数放置方案，不是新的训练目标。相同权重、相同输入下，EP 应尽量复现普通 MoE 的结果。

### 为什么必须使用 `torchrun`？

EP 的每个 Rank 都需要加入同一个 distributed process group，并在相同位置执行 All-to-All collective。直接运行普通 `python` 没有这些 Rank 信息。

### 为什么 `num_experts` 必须整除 GPU 数？

当前实现采用静态平均分片，每个 Rank 保存相同数量的 Expert。未来可以扩展非均匀 Expert placement，但当前版本没有实现。

### 为什么不能直接使用 DDP？

不同 Rank 保存的 Expert 参数不同。普通 DDP 假设各 Rank 参数结构一致，直接使用会造成结构检查失败或错误的 Expert 梯度同步。

### 为什么不能直接 reshape Expert 输出？

因为数据已经按目标 Rank 和本地 Expert 排序。`reshape` 只改变形状，不改变元素顺序，必须通过 inverse indices 撤销两次排序。

### 为什么推理时所有 Rank 都要执行 generate？

All-to-All 是 collective。只让 Rank 0 进入模型、其他 Rank 不进入，会导致 Rank 0 永久等待其他进程，形成通信死锁。

## 15. 后续优化方向

1. 拆分 DP group 与 EP group，支持 DP × EP 的二维并行。
2. 把路由元数据封装为可复用 handle，减少重复排序和分配。
3. 移除 `.tolist()` 带来的 CPU/GPU 同步。
4. 扩展可选 grouped GEMM 推理路径的全层数值兼容性和整体性能；训练仍用原专家前向。
5. 用独立 CUDA stream 重叠 dispatch/combine 与 Expert 计算。
6. 增加 capacity、Token dropping 和负载不均衡保护。
7. 将完整 checkpoint 改为真正的分片读写。
8. 在更大模型与合适硬件上评估 DeepEP 或其他 fused EP backend。

## 16. 总结

这次扩展把 MiniMind 的 MoE 从“每个 Rank 保存全部 Expert”改造成了完整的 Expert Parallel 实现，打通了以下关键环节：

```text
全局 Top-K Router
→ 动态不等长 Token dispatch
→ 本地 Expert 分组计算
→ Expert 输出 combine 与顺序恢复
→ Top-K 权重合并
→ 可求导反向 All-to-All
→ 共享参数与 Expert 梯度同步
→ 全局梯度裁剪
→ 完整权重合并和逐 Rank 续训
→ 多卡同步生成
```

当前实现已经可以完成 EP 推理、预训练和 Full SFT，并通过普通 MoE 输出一致性、Top-1/Top-2 backward、BF16 多 epoch 训练、真实模型 smoke test 和 checkpoint resume 验证。


## 17. 可选 Prefill / Decode Grouped GEMM 推理

本地专家前向新增显式扩展点：GPU bincount 后交给可选后端；原路由、dispatch、combine 与顺序恢复沿用原实现。默认 `--expert_backend original`，无需额外算子依赖。训练和非 inference_mode 调用回退原路径，保持反向能力。

`native_bf16` 用原专家模块保持原数值；`cuda_split_last_layer` 仅最后一层 local M≤256 使用新算子；`cuda_split` 对全部支持层启用 Prefill/Decode，目前全层数值检查未通过，只作实验。新算子支持 BF16/FP16 输入输出、FP32 累加，默认选择 local M≤256 的 Decode 策略，其余为 Prefill。BF16 输出不代表与原中间 BF16 归约逐位相同。

```bash
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
cd /data/user4/tongleiwen/minimind
CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=2 eval_llm.py \
  --use_ep --weight pretrain_ep --hidden_size 768 --num_hidden_layers 8 \
  --num_experts 4 --num_experts_per_tok 1 --model_dtype bf16 \
  --expert_backend cuda_split_last_layer \
  --grouped_gemm_library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04_07_split_20261009/results/R16_release_build/my_cuda_grouped_split.so \
  --grouped_gemm_python_dir /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04_07_split_20261009/source/dev/pytorch
```

连续 KN 权重可额外传 `--kn_checkpoint /absolute/path/checkpoint.pth`，直接加载转换后的全模型参数。普通 NK 权重也可使用，由初始化缓存打包专家权重。后端注册不会增加 state_dict 参数键；权重重新加载时清除打包缓存。未合并 LoRA 不使用该扩展。

完整专家 MLP 已验证100份真实输入；双卡完整模型、分布式边界和逐步生成测试分层检查。单算子收益尚未稳定传到模型整体前向，因此保持显式选择，不把重新训练作为本次接入的前置条件。
