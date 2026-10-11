> 历史实验记录：保留该轮配置和结果；当前执行入口见项目 README。

# 原生精度 EP 路由接口

保持模型原 Router 和 PyTorch 专家，优化路由分桶、输入打包和输出合并。输入、输出支持 FP32/FP16/BF16；模型训练沿用 FP32 参数与 BF16 autocast。

```python
from model.ep_routing_backend import attach_ep_routing

controller = attach_ep_routing(
    model,
    library="/absolute/path/my_cuda_ep.so",
    python_dir="/absolute/path/my_cuda_kernels/dev/pytorch",
    mode="compact",
    decode_capacity_tokens=32,
)
```

专家数与 Top-K 使用模型的 `num_experts`、`num_experts_per_tok`，Rank 数来自实际进程组。要求专家总数能被 Rank 数整除，`1 <= Top-K <= num_experts`。运行时分桶支持每轴 1～1024 桶；实际整模型验证范围为 Rank1/2/3/4、专家4/6/8、Top-K1/2/3/4。

- `compact`：推理和训练均支持。训练使用有梯度注册的功能式 pack/gather/combine，不复用活跃反向张量。
- `fixed`：推理 Decode 固定容量；所有 Rank 同步设置 `controller.fixed_decode=True`。每 Rank Token 数不能超过 `decode_capacity_tokens`，超出时报错。Prefill 设置为 False；训练明确使用 compact。
- `dedup`：Top-2 推理实验；最终测量慢于 compact，不作为默认方案。

推理需要 `torch.inference_mode()`。Top-1/2 使用融合合并，保留乘法和 BF16 舍入边界；Top-K>2 使用融合加权加原 PyTorch `index_add_`，不承诺原生多路浮点归约逐位确定。每层每步重新计算路由，工作区缓存只保存空间。

训练建议对每个原专家 `expert.compile(dynamic=True, fullgraph=True)`，对模型 `model.compile(dynamic=True, fullgraph=False)`；动态 EP 控制流保留显式边界。优化训练不支持 FP8 通信，缺库或不合法配置直接报错。

`controller.uninstall()` 可移除扩展点。不得在训练图仍待反向时修改路由配置或替换控制器。
