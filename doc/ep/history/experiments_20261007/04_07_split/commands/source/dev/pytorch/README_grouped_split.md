> 历史实验记录：保留该轮配置和结果；当前执行入口见项目 README。

# Prefill / Decode 前向接口

BF16/FP16 输入、权重和输出，FP32 累加。它与 PyTorch 默认 BF16 中间归约不是同一数值契约。原模型默认仍是 `original`；`native_bf16` 沿用原专家模块，保留原结果（FP16 模型也沿用其原精度）。不支持反向；MiniMind 在训练或非 inference_mode 调用中使用原实现。

```python
from grouped_gemm_split import load_library, grouped_gemm, GemmWorkspace
load_library("/absolute/path/my_cuda_grouped_split.so")
y = grouped_gemm(x, packed_b, counts)  # 自动选路；独立分配输出
workspace = GemmWorkspace.allocate(x, packed_b, capacity=4096)
y = workspace.run(x, packed_b, counts) # 复用输出与工作区
```

输入 `x[M,K]` 按专家依次打包，权重 `packed_b[E,K,N]` 连续，GPU int64 `counts[E]` 非负且总和等于 M，E=1～16。支持 (K,N)=(768,2432)/(2432,768)/(1024,1024)。M≤256 自动使用 Decode，其他 M 使用 Prefill；也可显式 `family="prefill"` 或 `"decode"`。小 M 的 Prefill 请求仍可使用 Decode 计算策略。动态专家分配来自 GPU counts，不要求固定每个专家 M。

`debug_counts=True` 会同步检查 counts 的值；正常性能路径由调用方保证值合法，结构和工作区安全检查仍执行。输入/权重需16字节对齐、counts需8字节对齐；输出和可写工作区必须与输入及彼此不重叠。

`torch.ops.my_cuda_kernels_split` 下分别注册 `prefill`、`decode`、`prefill_out`、`decode_out`，支持 FakeTensor，分配接口支持 torch.compile；Graph 前先预热和预分配。工作区返回的输出是复用视图，后续调用会覆盖，需要长期持有时调用 `clone()`。

完整专家 MLP 使用 `grouped_expert_mlp_split.GroupedExpertMLP(experts, expert_ids)`，按传入专家顺序缓存 gate/up/down 权重，保留 BF16 SiLU 和乘法的舍入边界。首次使用每条 stream 分配容量，按至少256行、2的幂扩容。不同 stream 使用独立工作区；跨 stream 数据依赖仍由调用方建立。常规参数版本改变会使缓存失效；没有版本计数器的 inference 参数修改后必须显式 `invalidate()` 并重建。直接 MLP 缓存发现权重变动会报错，MiniMind 控制器则在 Graph 外自动重建。

MiniMind 可直接在 `eval_llm.py` 使用：

- `--expert_backend original`：默认原实现。
- `--expert_backend native_bf16`：原数值兼容路径。
- `--expert_backend cuda_split_last_layer`：仅最后一层、local M≤256 使用新 CUDA；其余原实现。
- `--expert_backend cuda_split`：所有支持形状的层使用新 CUDA；目前是实验选项，全层模型数值准入未通过。

另传 `--model_dtype bf16 --grouped_gemm_library /absolute/path/library.so --grouped_gemm_python_dir /absolute/path/my_cuda_kernels/dev/pytorch`。已有连续 KN checkpoint 用 `--kn_checkpoint /absolute/path/checkpoint.pth` 加载；普通 NK checkpoint 仍可使用，首次调用打包权重。未合并 LoRA 不支持新后端。整个 EP 含动态 All-to-All 与原路由，目前没有完成全链路 CUDA Graph。
