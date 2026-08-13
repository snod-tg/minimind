# MiniMind Expert Parallel 新增说明（by tongleiwen）

> 作者：tongleiwen  
> 更新时间：2026-08-13  
> 适用范围：MiniMind 原生 PyTorch MoE 模型的 Expert Parallel 推理与训练

## 1. 新增目标

MiniMind 原有的 `MOEFeedForward` 会在每个进程、每张 GPU 上创建全部 Expert。假设模型有 4 个 Expert，即使每个 Token 只选择其中 1 个 Expert，每张 GPU 仍然需要保存全部 4 个 Expert 的参数。

本次新增 Expert Parallel（下文简称 EP），目标是把 Expert 分片保存到多个进程中：

- Router、Attention、Embedding、RMSNorm、LM Head 等共享参数仍然在每个 Rank 上保留一份。
- 全局 Expert 平均分配到不同 Rank。
- Router 仍然对全局 Expert 进行 Top-K 选择。
- Token 根据目标 Expert 被发送到对应 Rank。
- 对应 Rank 使用本地 Expert 完成计算。
- Expert 输出再发送回 Token 原来的 Rank，并恢复原 Token 顺序。
- 推理支持 Top-1 和参数化 Top-K。
- 训练支持完整 forward、backward、梯度同步、优化器更新、梯度裁剪、权重保存和断点续训。

以 4 个 Expert、2 个 Rank 为例：

| Rank | 保存的全局 Expert | 本地 Expert 数量 |
|---|---|---:|
| Rank 0 / GPU 0 | Expert 0、Expert 1 | 2 |
| Rank 1 / GPU 1 | Expert 2、Expert 3 | 2 |

Expert 映射关系为：

```python
experts_per_rank = num_experts // ep_size
target_rank = global_expert_id // experts_per_rank
local_expert_id = global_expert_id % experts_per_rank
global_expert_id = rank * experts_per_rank + local_expert_id
```

## 2. 涉及的正式代码文件

本次 EP 功能主要修改以下文件：

| 文件 | 新增内容 |
|---|---|
| `model/model_minimind.py` | EP 配置、可求导 All-to-All、Token dispatch/combine、本地 Expert 执行、Top-K 合并、辅助负载均衡损失 |
| `eval_llm.py` | `--use_ep` 推理参数、分布式初始化、分片权重加载、Rank 0 交互与多 Rank 同步生成 |
| `trainer/trainer_utils.py` | EP 参数统计、共享参数初始化、梯度同步、全局梯度裁剪、分片 checkpoint 合并与恢复 |
| `trainer/train_pretrain.py` | EP 预训练入口、训练参数、手动梯度同步、EP checkpoint 保存 |
| `trainer/train_full_sft.py` | EP Full SFT 入口、训练参数、手动梯度同步、EP checkpoint 保存 |
| `.gitignore` | 忽略 `playground/` 学习实验目录 |

`playground/` 中保存了从 Router、Rank、All-to-All、不等长 All-to-All 到完整 EP 的学习代码。该目录用于学习和调试，不属于正式提交内容。

## 3. 配置项新增

在 `MiniMindConfig` 中新增：

```python
self.use_ep = kwargs.get("use_ep", False)
self.ep_size = kwargs.get("ep_size", None)
```

相关配置含义如下：

| 配置 | 含义 |
|---|---|
| `use_moe` | 是否使用 MoE；开启 EP 时会自动设置为 `True` |
| `use_ep` | 是否使用 Expert Parallel |
| `ep_size` | EP 通信组包含的 Rank 数量，当前实现等于 `WORLD_SIZE` |
| `num_experts` | 全局 Expert 总数 |
| `num_experts_per_tok` | 每个 Token 选择的 Expert 数量，即 Top-K |
| `norm_topk_prob` | 是否归一化 Top-K Expert 权重 |
| `router_aux_loss_coef` | Router 负载均衡辅助损失系数 |

必须满足：

```text
1 <= num_experts_per_tok <= num_experts
num_experts % ep_size == 0
```

## 4. EP 模块构造

新增 `ExpertParallelMOEFeedForward`，用于代替原来的 `MOEFeedForward`。

模型层的选择逻辑变为：

```python
if not config.use_moe:
    self.mlp = FeedForward(config)
elif config.use_ep:
    self.mlp = ExpertParallelMOEFeedForward(config)
else:
    self.mlp = MOEFeedForward(config)
```

### 4.1 每个 Rank 只创建本地 Expert

每个 Rank 根据自身编号计算负责的全局 Expert 范围：

```python
self.experts_per_rank = config.num_experts // self.ep_size
self.first_expert_id = self.ep_rank * self.experts_per_rank
self.local_expert_ids = range(
    self.first_expert_id,
    self.first_expert_id + self.experts_per_rank,
)
```

本地 Expert 使用 `ModuleDict` 保存，并直接使用全局 Expert ID 作为 key：

```python
self.experts = nn.ModuleDict({
    str(global_id): FeedForward(config)
    for global_id in self.local_expert_ids
})
```

使用全局 ID 的原因是让 EP 模型的 `state_dict` 名称与普通 MoE 模型保持兼容。例如 Rank 1 保存的 Expert 参数仍然叫：

```text
model.layers.0.mlp.experts.2.gate_proj.weight
model.layers.0.mlp.experts.3.gate_proj.weight
```

这样不同 Rank 的 Expert shard 合并后，可以直接由普通 MoE 或 EP 推理加载。

## 5. EP Forward 完整流程

设输入形状为：

```text
x: [batch_size, seq_len, hidden_size]
```

### 5.1 展平 Token

```python
x_flat = x.reshape(-1, hidden_size)
```

展平后：

```text
num_tokens = batch_size * seq_len
x_flat: [num_tokens, hidden_size]
```

### 5.2 Router 选择全局 Expert

```python
scores = softmax(gate(x_flat))
topk_weight, topk_expert_id = torch.topk(
    scores,
    k=num_experts_per_tok,
)
```

形状为：

```text
scores:          [num_tokens, num_experts]
topk_weight:     [num_tokens, top_k]
topk_expert_id:  [num_tokens, top_k]
```

Top-K 大于 1 时，一个 Token 会产生 K 条独立的路由 assignment：

```python
assignment_token_indices = ...
assignment_weight = topk_weight.reshape(-1)
global_expert_id = topk_expert_id.reshape(-1)
assignment_x = x_flat[assignment_token_indices]
```

因此：

```text
num_assignments = num_tokens * top_k
```

`assignment_token_indices` 会一直保留，用于最后把 K 个 Expert 输出累加回原 Token。

### 5.3 计算目标 Rank 和本地 Expert ID

```python
target_rank = global_expert_id // experts_per_rank
local_expert_id = global_expert_id % experts_per_rank
```

例如 4 个 Expert、2 个 Rank：

| 全局 Expert | target_rank | local_expert_id |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 1 | 0 |
| 3 | 1 | 1 |

### 5.4 按目标 Rank 排序

All-to-All 的发送缓冲区必须按照目标 Rank 连续排列：

```python
rank_sort_indices = torch.argsort(target_rank, stable=True)
send_x = assignment_x[rank_sort_indices]
send_local_expert_id = local_expert_id[rank_sort_indices]
```

同时动态计算每个目标 Rank 的发送数量：

```python
send_counts = torch.bincount(target_rank, minlength=ep_size)
```

各 Rank 先交换 `send_counts`，获得自己将从其他 Rank 接收多少 Token：

```python
dist.all_to_all_single(recv_counts, send_counts)
```

最终得到：

```text
send_splits = 当前 Rank 向每个目标 Rank 发送多少 assignment
recv_splits = 当前 Rank 从每个来源 Rank 接收多少 assignment
```

这实现了动态、不等长 All-to-All，不要求每张 GPU 收到相同数量的 Token。

### 5.5 Dispatch：发送 Token 和 Expert ID

第一次 All-to-All 发送隐藏状态：

```python
recv_x = differentiable_all_to_all(
    send_x,
    output_splits=recv_splits,
    input_splits=send_splits,
)
```

第二次普通 All-to-All 发送 `local_expert_id` 元数据：

```python
dist.all_to_all_single(
    recv_local_expert_id,
    send_local_expert_id,
    output_split_sizes=recv_splits,
    input_split_sizes=send_splits,
)
```

隐藏状态需要参与反向传播，因此使用可求导通信；Expert ID 是整数路由元数据，不需要梯度。

### 5.6 按本地 Expert 再次排序

收到的 Token 最初按照“来源 Rank”分段，还没有按照本地 Expert 连续排列，因此需要第二次排序：

```python
expert_sort_indices = torch.argsort(
    recv_local_expert_id,
    stable=True,
)
expert_sorted_x = recv_x[expert_sort_indices]
```

再统计每个本地 Expert 收到多少 Token：

```python
local_expert_counts = torch.bincount(
    recv_local_expert_id,
    minlength=experts_per_rank,
)
```

相同 Expert 的 Token 现在在内存中连续，可以成批输入对应 Expert：

```python
for local_id, count in enumerate(local_expert_counts):
    global_id = first_expert_id + local_id
    output = self.experts[str(global_id)](expert_input)
```

### 5.7 恢复接收顺序

Expert 计算完成后，先使用 `inverse_expert_indices` 撤销本地 Expert 排序：

```python
inverse_expert_indices[expert_sort_indices] = arange(...)
recv_order_output = expert_sorted_output[inverse_expert_indices]
```

不能直接 `reshape`，因为当前数据顺序已经不是原始 Token 顺序。

### 5.8 Combine：把 Expert 输出发回来源 Rank

第二次可求导 All-to-All 把 Expert 输出返回 Token 的来源 Rank：

```python
returned_sorted_output = differentiable_all_to_all(
    recv_order_output,
    output_splits=send_splits,
    input_splits=recv_splits,
)
```

返回后使用 `inverse_rank_indices` 撤销最初的目标 Rank 排序：

```python
inverse_rank_indices[rank_sort_indices] = arange(...)
returned_assignment_output = returned_sorted_output[inverse_rank_indices]
```

### 5.9 Top-K 加权并合并回原 Token

Expert 输出乘以 Router 权重：

```python
weighted_assignment_output = (
    returned_assignment_output * assignment_weight[:, None]
).to(x_flat.dtype)
```

显式转换到 `x_flat.dtype` 是为了兼容 BF16 autocast。Router 权重可能保持 FP32，如果不转换，`index_add_` 会因为 BF16/FP32 类型不同而报错。

最后把同一 Token 的 K 个 Expert 输出累加：

```python
y_flat.index_add_(
    0,
    assignment_token_indices,
    weighted_assignment_output,
)
```

恢复形状：

```python
return y_flat.reshape(batch_size, seq_len, hidden_size)
```

## 6. 可求导 All-to-All

PyTorch 原始 `dist.all_to_all_single` 不会自动为当前使用方式建立所需的 autograd 反向通信，因此新增 `_AllToAll(torch.autograd.Function)`。

Forward 保存：

```text
input_splits
output_splits
process_group
```

Backward 接收到 `grad_output` 后，再执行一次方向相反的 All-to-All：

```python
grad_input = all_to_all_single(
    grad_output,
    output_splits=forward_input_splits,
    input_splits=forward_output_splits,
)
```

其本质是：

```text
dispatch.forward  的反向 = combine
combine.forward   的反向 = dispatch
```

因此训练中实际会出现四次隐藏状态通信：

```text
Forward:   dispatch → Expert → combine
Backward:  dispatch ← Expert ← combine
```

这一数学关系与 DeepSeek DeepEP 的 dispatch/combine backward 设计一致；当前 MiniMind 实现使用 PyTorch NCCL/Gloo collective，没有引入 DeepEP CUDA 库。

## 7. Router 辅助负载均衡损失

EP 训练保留并扩展了原 MoE 的 Router auxiliary loss。

先统计 Top-K 路由到各 Expert 的比例：

```python
load = one_hot(topk_expert_id).float().mean(...)
```

不同 Rank 处理不同训练样本，因此对 `load` 执行 All-Reduce，获得全局负载分布：

```python
dist.all_reduce(load)
load /= ep_size
```

最终计算：

```python
aux_loss = (
    load * scores.mean(dim=0)
).sum() * num_experts * router_aux_loss_coef
```

Top-K ID 是离散选择，不参与梯度；梯度通过 `scores` 返回 Router gate。

## 8. 为什么 EP 训练不能直接使用普通 DDP

普通 DDP 假设每个 Rank 的参数名称、数量和顺序完全一致，并对同位置参数做梯度 All-Reduce。

但 EP 中：

```text
Rank 0: experts.0.*, experts.1.*
Rank 1: experts.2.*, experts.3.*
```

如果直接套 DDP，会遇到两类问题：

1. 不同 Rank 的参数结构不一致，DDP 初始化可能失败。
2. 即使强行对齐，也可能错误地把 Expert 0 的梯度与 Expert 2 的梯度相加。

因此 EP 训练不再使用：

```python
DistributedDataParallel(model)
```

而是新增手动的共享参数同步和梯度同步。

## 9. EP 参数初始化与梯度同步

### 9.1 共享参数初始化

不同 Rank 使用不同随机种子，以便 DataLoader 和其他随机过程具备 Rank 差异；但共享模型参数必须完全相同。

`sync_ep_shared_parameters()` 在模型创建后，把 Rank 0 的非 Expert 参数广播到所有 Rank：

```python
for name, param in model.named_parameters():
    if not is_ep_expert_parameter(name):
        dist.broadcast(param.data, src=0)
```

Expert 参数不广播，因为每个全局 Expert 只存在于负责它的 Rank。

### 9.2 共享参数梯度

Router、Attention、Embedding 等共享参数会在每个 Rank 上产生本地梯度，需要 All-Reduce 求平均：

```python
dist.all_reduce(param.grad)
param.grad /= world_size
```

### 9.3 Expert 参数梯度

每个 Expert 只保存在一个 Rank。经过 dispatch 后，该 Expert 已经收到来自所有 Rank 的 Token，因此本地 Expert 梯度已经包含整个 EP 组的贡献。

Expert 梯度不能再与其他 Rank 的不同 Expert 做 All-Reduce，只需要除以 `world_size`，使梯度尺度与“各 Rank loss 求平均”的训练语义一致：

```python
param.grad /= world_size
```

### 9.4 全局梯度裁剪

普通 `clip_grad_norm_` 只能看到当前 Rank 参数，并且会重复计算每个 Rank 都存在的共享参数。

新增 `clip_ep_grad_norm_()`：

- 当前 Rank 计算本地 Expert 梯度平方和。
- 共享参数梯度平方和先除以 `world_size`，避免重复计算。
- 所有 Rank 对平方和执行 All-Reduce。
- 得到去重后的全局梯度范数。
- 所有 Rank 使用同一个裁剪系数。

## 10. 训练入口新增

`train_pretrain.py` 和 `train_full_sft.py` 都新增：

```text
--use_ep
--num_experts
--num_experts_per_tok
```

开启 `--use_ep` 后：

- 自动开启 MoE，不需要再传 `--use_moe 1`。
- 必须通过 `torchrun` 启动。
- `WORLD_SIZE` 必须至少为 2。
- `num_experts` 必须能被 `WORLD_SIZE` 整除。
- 当前不支持 `--use_compile 1`。
- 不使用普通 DDP，改用 EP 手动同步逻辑。

训练数据仍然使用 `DistributedSampler`。每个 Rank 读取不同样本，形成数据并行 batch；MoE Token 再通过同一组 Rank 做 Expert Parallel 通信。

因此当前实现可以理解为：

```text
共享参数：手动 Data Parallel
Expert 参数：Expert Parallel
EP group：WORLD group
```

有效全局 batch size 为：

```text
batch_size_per_rank × world_size × accumulation_steps
```

## 11. EP Checkpoint 设计

### 11.1 完整推理权重

保存时，每个 Rank 先取得自己的 `state_dict`，其中包含：

- 一份共享参数。
- 当前 Rank 的本地 Expert 参数。

各 Rank 使用 `dist.gather_object` 把本地 Expert shard 发送给 Rank 0。Rank 0 合并所有 Expert，写出完整权重：

```text
{weight}_{hidden_size}_moe.pth
```

完整文件包含全部全局 Expert，可用于：

- 普通单卡 MoE 推理。
- 多卡 EP 推理。
- 下一阶段普通 MoE 训练。
- 下一阶段 EP 训练的初始权重。

写文件使用 `.tmp` 加 `os.replace()`，减少保存中断后留下半个 checkpoint 的风险。

### 11.2 EP 断点续训权重

Optimizer state 与参数 shard 绑定，因此每个 Rank 分别保存自己的续训状态：

```text
{weight}_{hidden_size}_moe_ep_rank0_resume.pth
{weight}_{hidden_size}_moe_ep_rank1_resume.pth
...
```

每个 Rank resume 文件包含：

```text
model       当前 Rank 的共享参数和本地 Expert shard
optimizer   当前 Rank 的 optimizer state
scaler      AMP GradScaler state
epoch       当前 epoch
step        当前 step
world_size  保存时的 Rank 数量
wandb_id    实验追踪 ID（如有）
```

EP resume 要求恢复时 `world_size` 与保存时一致。因为 Expert shard 和 optimizer state 已按照原 Rank 切分，当前版本没有实现改变 EP world size 后的自动重分片。

### 11.3 从普通 MoE 权重首次切换到 EP

每个 Rank 先把完整 MoE checkpoint 加载到 CPU，然后只筛选本 Rank 模型实际存在的 key：

```python
local_state_dict = {
    key: value
    for key, value in full_state_dict.items()
    if key in local_model_keys
}
```

筛选后使用 `strict=True` 加载，确保当前 Rank 所需的共享参数和本地 Expert 参数都存在。

首次从普通 MoE 切换到 EP 时，应使用完整权重作为 `--from_weight`，不要使用旧的普通 DDP resume 文件。EP 自己保存过一次之后，才能使用 EP rank resume。

## 12. 推理入口新增

`eval_llm.py` 新增：

```text
--use_ep
--num_experts
--num_experts_per_tok
```

### 12.1 分布式初始化

`init_ep()` 完成：

- 检查必须由 `torchrun` 启动。
- CUDA 环境使用 NCCL，CPU 环境使用 Gloo。
- 根据 `LOCAL_RANK` 绑定当前 GPU。
- 自动开启 `use_moe`。

### 12.2 分片加载权重

推理时完整 checkpoint 先加载到 CPU，再只选择当前 Rank 的共享参数和本地 Expert，避免把远端 Expert 放入当前 GPU。

注意：当前仍会在每个 Rank 的 CPU 内存中暂时读取一次完整 checkpoint，只减少 GPU Expert 显存，还没有实现真正的按文件 shard 读取。

### 12.3 多 Rank 同步生成

交互式推理只允许 Rank 0：

- 读取自动/手动模式。
- 读取用户 Prompt。
- 创建 `TextStreamer`。
- 打印回答和速度。

Rank 0 会把以下信息广播到所有 Rank：

```text
input_mode
prompt
random_seed
```

所有 Rank 必须使用相同 Prompt 和随机种子执行 `model.generate()`，才能在每一层 MoE 中同时进入对应的 All-to-All collective，避免通信死锁并保证生成 Token 一致。

推理过程包裹在：

```python
with torch.inference_mode():
```

退出时执行 barrier 并销毁 process group。

## 13. 参数统计修复

原 `get_model_params()` 假设每个进程都有 `experts.0`，在 EP Rank 1 中可能找不到 Expert 0，从而打印负的 Active 参数量。

新增逻辑会识别当前 Rank 实际拥有的 Expert ID，分别计算：

```text
Model Params/rank  当前 Rank 实际保存的参数量
Global             合并全部 Expert 后的全局参数量
Active             每个 Token 实际激活的参数量
```

真实 768/8 层、4 Expert、Top-1、双 Rank 测试打印为：

```text
Model Params/rank: 108.76M-Global198.42M-Active63.94M
```

## 14. 使用方法

以下命令假设当前目录是 MiniMind 项目根目录。

### 14.1 普通单卡 MoE 推理

```bash
CUDA_VISIBLE_DEVICES=0 python eval_llm.py \
  --use_moe 1 \
  --weight pretrain \
  --save_dir out
```

### 14.2 双卡 EP 推理

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

### 14.3 首次从普通 MoE 权重开始 EP 预训练

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

建议第一次使用新的 `save_weight` 名称，避免覆盖原普通 MoE 权重。

### 14.4 EP 断点续训

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

恢复时必须使用与保存 checkpoint 时相同的 `WORLD_SIZE`、`num_experts` 和模型结构。

### 14.5 双卡 EP Full SFT

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

## 15. 已完成验证

### 15.1 Forward 与推理一致性

- 普通 MoE 与双卡 EP 使用相同权重、相同 Prompt、贪心解码。
- 多个 Prompt 的输出文本哈希完全一致。
- 说明 Router、dispatch、本地 Expert、combine 和 inverse reorder 的结果正确。

### 15.2 Backward 验证

以下项目均通过双进程 Gloo 和双卡 NCCL 测试：

- Top-1 forward/backward。
- Top-2 forward/backward。
- 本地 Expert 获得非空且有限的梯度。
- 共享参数在各 Rank 同步后梯度一致。
- 全局梯度范数计算与裁剪。
- Optimizer step 后参数发生更新。
- 完整 checkpoint 包含 Expert 0、1、2、3。
- 每个 Rank 的 model、optimizer、scaler resume 成功加载。

### 15.3 BF16 多 Epoch 训练验证

双 RTX 4080 SUPER、PyTorch 2.10.0+cu128 环境下完成：

Tiny Top-2 模型连续训练 5 个 epoch：

```text
Epoch 1: loss 8.8426, aux_loss 0.0010
Epoch 2: loss 8.1929, aux_loss 0.0010
Epoch 3: loss 7.7300, aux_loss 0.0011
Epoch 4: loss 7.6172, aux_loss 0.0010
Epoch 5: loss 7.4878, aux_loss 0.0010
```

真实 `pretrain_768_moe.pth`、8 层、4 Expert、Top-1 连续运行 3 个 epoch，共 6 个 optimizer step：

- 无 NaN。
- 无 NCCL 错误。
- 无通信死锁。
- Router aux loss 稳定在约 `0.0040`。
- 完整权重保存和逐 Rank resume 正常。

测试使用独立名称，结束后已删除全部 smoke-test 权重；正式 `pretrain_768_moe*.pth` 哈希未变化。

## 16. 当前限制

当前实现以正确性、可读性和学习 EP 完整流程为主，仍有以下限制：

1. EP group 当前直接使用 `dist.group.WORLD`，还不支持把 WORLD 切分为独立 EP group 和 DP group。
2. Expert 必须平均分配，`num_experts` 必须能被 `WORLD_SIZE` 整除。
3. 没有 Expert capacity、Token dropping 或溢出策略。
4. 使用 Python `argsort`、`bincount`、`.tolist()` 和逐 Expert 循环，存在 GPU/CPU 同步与 kernel launch 开销。
5. Expert 计算没有 grouped GEMM，也没有把多个 Expert 融合为单个高效 kernel。
6. 通信与 Expert 计算尚未使用独立 CUDA stream 重叠。
7. 通信数据未使用 FP8 压缩，dispatch/combine 使用输入 Tensor 的常规精度。
8. 当前动态 All-to-All 训练不支持 `torch.compile`。
9. 完整 checkpoint 会先被每个 Rank 读取到 CPU，再筛选本地参数，CPU 峰值内存尚未优化。
10. EP resume 暂不支持改变 `WORLD_SIZE` 后自动重分片 optimizer state。
11. 正式训练入口目前只接入 `train_pretrain.py` 和 `train_full_sft.py`，DPO、PPO、GRPO 等训练器尚未接入 EP。
12. Attention、Embedding、LM Head 等共享参数仍然在每张 GPU 上完整复制；EP 只切分 Expert 参数。
13. 小模型、batch size 较小或逐 Token decode 时，All-to-All 通信开销可能大于 Expert 并行收益，因此 EP 不保证比单卡更快。

## 17. 与 DeepSeek DeepEP 的关系

当前新增代码没有安装或依赖 DeepEP，但在数学通信模型上采用同样的核心抽象：

```text
dispatch：Token → Expert 所在 Rank
combine： Expert 输出 → Token 来源 Rank
dispatch backward = combine
combine backward = dispatch
```

DeepEP 在此基础上进一步提供：

- 专用高吞吐和低延迟 CUDA 通信 kernel。
- `ElasticBuffer` 复用通信缓冲区。
- routing handle 复用 dispatch 布局。
- 独立通信流和 EventOverlap。
- FP8 dispatch、BF16 combine。
- grouped GEMM 友好的 Expert 数据布局。
- NVLink/RDMA 分层通信。
- 更低 SM 占用和更大 EP 规模。

当前 MiniMind EP 更适合作为一份可运行、可训练、便于逐变量学习的参考实现。后续优化方向可以按照以下顺序推进：

1. 把动态路由元数据封装成可复用 handle。
2. 减少 `.tolist()` 导致的 CPU 同步。
3. 引入 grouped GEMM。
4. 使用独立 CUDA stream 重叠 dispatch/combine 与计算。
5. 增加 capacity 和负载不均衡保护。
6. 拆分 EP group 与 DP group。
7. 在硬件满足条件时评估 DeepEP 或其他 fused EP backend。

DeepEP 官方仓库：<https://github.com/deepseek-ai/DeepEP>

## 18. 总结

本次新增使 MiniMind MoE 从“每张 GPU 保存全部 Expert”扩展为完整的 Expert Parallel 实现，并打通了：

```text
全局 Router
→ 动态不等长 dispatch
→ 本地 Expert 分组计算
→ combine 与顺序恢复
→ Top-K 加权合并
→ 可求导反向 All-to-All
→ 共享参数/Expert 梯度同步
→ 全局梯度裁剪
→ 完整权重合并
→ 分 Rank 断点续训
→ eval_llm 多 Rank 同步推理
```

它已经具备正确运行 EP 推理、预训练和 Full SFT 的基本闭环，同时保留普通 Dense、普通 MoE 和单卡推理路径。
