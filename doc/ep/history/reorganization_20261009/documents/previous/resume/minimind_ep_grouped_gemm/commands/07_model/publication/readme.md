# My CUDA Kernels

从数学公式和参考实现出发，实践 CUDA / TileLang 算子优化。保留朴素版与各阶段 kernel，比较向量化、warp 归约、寄存器复用、共享内存布局、异步拷贝、流水线和分组调度的实际收益。

项目包含 RMSNorm、Fused Residual RMSNorm 的前向/反向，以及普通 GEMM 和 MoE Grouped GEMM 前向。手写 CUDA 支持 FP32/FP16/BF16；TileLang 示例目前使用 FP16 存储、FP32 归约。GEMM 使用 cuBLAS 参考，Grouped GEMM 另有可选 CUTLASS 对照。

## 目录

```text
dev/
  cuda/       # 手写 CUDA 算子与共用头文件
  tilelang/   # 四个 TileLang 算子示例
  pytorch/    # kernel9 推理接口、工作区与权重缓存
analysis/     # 算子分析报告
figures/      # 报告引用的统一风格图表
.gitignore
readme.md
```

## 分析入口

| 实现 | 算子 | 报告 |
|---|---|---|
| CUDA | RMSNorm forward | [分析](analysis/cuda_rmsnorm_forward.md) |
| CUDA | RMSNorm backward | [分析](analysis/cuda_rmsnorm_backward.md) |
| CUDA | Fused Residual RMSNorm forward | [分析](analysis/cuda_fused_residual_rmsnorm_forward.md) |
| CUDA | Fused Residual RMSNorm backward | [分析](analysis/cuda_fused_residual_rmsnorm_backward.md) |
| TileLang | RMSNorm forward | [分析](analysis/tilelang_rmsnorm_forward.md) |
| TileLang | RMSNorm backward | [分析](analysis/tilelang_rmsnorm_backward.md) |
| TileLang | Fused Residual RMSNorm forward | [分析](analysis/tilelang_fused_residual_rmsnorm_forward.md) |
| TileLang | Fused Residual RMSNorm backward | [分析](analysis/tilelang_fused_residual_rmsnorm_backward.md) |
| CUDA | GEMM，固定 1024³ | [分析](analysis/gemm.md) |
| CUDA | MoE Grouped GEMM，固定总 M、可变专家分配 | [分析](analysis/grouped_gemm.md) |
| CUDA | MoE Grouped GEMM kernel9，动态总 M、BF16 模型维度 | [分析](analysis/grouped_gemm_kernel9.md) |
| PyTorch | kernel9 注册接口、工作区与权重缓存 | [分析](analysis/grouped_gemm_api.md) |
| PyTorch / CUDA | MiniMind-EP 真实完整专家 MLP 回放 | [分析](analysis/grouped_expert_mlp.md) |
| PyTorch / CUDA | MiniMind-EP 后端接入、数值传播与整体模型评估 | [分析](analysis/minimind_ep_integration.md) |

各报告注明形状、精度、硬件、计时方法、验证范围和数据日期。3060 Laptop 的历史 RMSNorm 数据与 4090D 的 GEMM / TileLang 数据分别呈现；不同测量方法或硬件的数据不直接混算加速比。

## 编译与运行

CUDA 示例使用 C++17。4090D 使用 `sm_89`，3060 Laptop 使用 `sm_86`。

```bash
mkdir -p result/build
nvcc -O3 -std=c++17 -arch=sm_89 dev/cuda/rmsnorm_forward.cu -o result/build/rmsnorm_forward
./result/build/rmsnorm_forward 10

nvcc -O3 -std=c++17 -arch=sm_89 dev/cuda/gemm_forward.cu -lcublas -o result/build/gemm_forward
./result/build/gemm_forward --dtype fp16 --kernel 9
./result/build/gemm_forward --self-test

nvcc -O3 -std=c++17 -arch=sm_89 dev/cuda/moe_grouped_gemm_forward.cu -lcublas -o result/build/moe_grouped_gemm_forward
./result/build/moe_grouped_gemm_forward --counts 64,96,160,192,64,96,160,192,64,96,160,192,64,96,160,192 --n 1024 --k 1024 --dtype fp16 --kernel 7
./result/build/moe_grouped_gemm_forward --self-test --kernel 7

# kernel8：总M=2048、E=16、N=K=1024，专家分配可变。
./result/build/moe_grouped_gemm_forward --m 2048 --experts 16 --n 1024 --k 1024 --dtype fp16 --kernel 8
./result/build/moe_grouped_gemm_forward --self-test --kernel 8

# kernel9：动态总 M，GPU counts 调度；支持 BF16/FP16 和三组固定 N/K。
./result/build/moe_grouped_gemm_forward --m 65536 --experts 2 --n 2432 --k 768 --dtype bf16 --kernel 9 --no-bench
./result/build/moe_grouped_gemm_forward --counts 0,1 --n 768 --k 2432 --dtype bf16 --kernel 9 --no-bench
```

CUTLASS 对照编译时追加 `--expt-relaxed-constexpr -DUSE_CUTLASS -I "$CUTLASS_ROOT/include"`；普通编译无需 CUTLASS。TileLang 示例需要可用的 TileLang/PyTorch/CUDA 环境及匹配的 C++ 工具链：

```bash
python dev/tilelang/rmsnorm_forward.py
python dev/tilelang/rmsnorm_backward.py
python dev/tilelang/fused_residual_rmsnorm_forward.py
python dev/tilelang/fused_residual_rmsnorm_backward.py
```

默认 TileLang backward 示例包含编译成本较高的 BLOCK_N=128；本次报告明确列出已验证的配置范围。

Grouped GEMM 输入已按专家打包，单算子执行一个专家线性层；现已增加真实完整专家 MLP 回放，包含三个 GEMM 与原激活，见 [MLP 分析](analysis/grouped_expert_mlp.md)。端到端 routing、dispatch/combine、通信、整体模型性能和反向仍属于后续实践范围。

PyTorch 前向接口为 `torch.ops.my_cuda_kernels.grouped_gemm(x, packed_b, counts)`，另提供预分配调用。构建、加载与缓存约定见 [接口分析](analysis/grouped_gemm_api.md)。

完整专家 MLP 包装为 `GroupedExpertMLP`；提供分配版、预分配版、空输入路径及缓存失效约束。回放验证与模型兼容性的区别、适用性能范围见 [MLP 分析](analysis/grouped_expert_mlp.md)。

双卡 EP 提供可切换专家后端，默认 `original`；仅最后一层小 M 的 `last_layer` 在本次模型与生成测试中通过数值门槛。整体性能与复测见 [接入分析](analysis/minimind_ep_integration.md)。
