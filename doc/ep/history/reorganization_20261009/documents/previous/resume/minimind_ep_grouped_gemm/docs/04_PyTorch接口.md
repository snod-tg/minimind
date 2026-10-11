# 04 PyTorch 接口

## 目的
让 kernel9 从独立基准算子变成模型可直接调用的 PyTorch 算子，并确认张量检查、内存管理与当前设备/stream 不破坏计算语义。单 GEMM 的吞吐优势尚不能代表正式调用或模型收益，本阶段单独测量接口路径。

## 执行方案
1. 在 C++ 用 TORCH_LIBRARY/TORCH_LIBRARY_IMPL 注册 CUDA 实现。CUDA launcher 复用已验证的 kernel9，保持三组 N/K、两种精度与 shape 默认策略；不改 GEMM 数学核心。
2. 检查 CUDA device、dtype、shape、contiguous、向量访问对齐与无梯度输入；预分配路径另检查输出/工作区大小、类型和读写内存互不重叠。
3. 用 CUDAGuard 切换到输入设备、在该设备当前 stream 上启动；输出与临时存储由 PyTorch 分配器管理，不在调用中同步或读取 GPU counts。counts 的合法性遵循阶段 02 契约，另提供显式同步调试检查。
4. 提供自动分配接口、预分配工作区接口和一次性专家权重缓存。缓存按显式专家映射打包，权重原件保留；更新、重新加载、迁移或训练时失效后重建。
5. 验证空输出、动态 M/counts、数值、非默认 stream、设备恢复、Graph、FakeTensor、opcheck、梯度拒绝与负向参数。用动态 M 的 eager 编译后端验证形状路径，不宣称 Inductor 优化或完整模型编译通过。
6. 对相同输入轮换测量 native C ABI、PyTorch 预分配和自动分配路径，分别记录 CUDA Graph 执行与 eager 调用。补充 PyTorch 工作区路径的 memcheck。

[完整执行指令](../commands/04_PyTorch接口.md) · [环境](../测试环境.md) · [结果与数据](../results/04_PyTorch接口.md)。注册方法参考 [PyTorch C++/CUDA 算子教程](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)，FakeTensor、Autograd 和 opcheck 使用 [torch.library 文档](https://docs.pytorch.org/docs/stable/library.html)；实际构建/验证版本为 2.9.0，未采用新文档中 2.13 才引入的 mutable Python API。

## 接口约定
普通调用：`torch.ops.my_cuda_kernels.grouped_gemm(x, packed_b, counts) -> y`。

预分配调用：`torch.ops.my_cuda_kernels.grouped_gemm_out(x, packed_b, counts, out, metadata, partial) -> None`。输出、metadata、partial 在 schema 中声明可变；不与输入或彼此重叠。metadata 为至少 280 字节的连续 uint8；partial 为连续 FP32，容量由默认 Split-K 路径决定，小 M 最多 8×M×N×4 字节，普通路径为空。

Python 包装提供 `GemmWorkspace.allocate(x,b)` 与 `workspace.run(x,b,counts)`。每个工作区同时只能被一条有序执行链使用；并发 stream 使用各自工作区。改变 M/输出形状时重新准备工作区与 Graph；同一 shape 内改变合法 counts 可复用。

x/packed_b 为 BF16/FP16，均不得 requires_grad；counts 为同设备连续 int64。输入矩阵起始地址需 16 字节对齐，输出需 4 字节，metadata/partial 需 8 字节对齐。连续张量的非零 storage_offset 也可能违反向量对齐，因此检查地址。

默认调用不会把 counts 数值读回 CPU。非法 GPU counts 会被原 kernel 的 valid 分支拦截，输出可能保持未写入状态；该分支只用于防越界，不能当作同步 Python 参数报错。需要完整数值检查时调用包装的 `validate_counts=True`，该调试选项含同步，不纳入正常路径计时。

`PackedExpertWeights` 按指定 expert_ids 建立 gate/up/down 的连续 `[E,K,N]` 缓存。重新加载或原地更新权重时必须显式 `invalidate()` 并新建缓存；训练、参数对象替换、device/dtype 改变可在 get() 时失效。对象/指针检查不能识别原地参数值更新，也不依赖 inference tensor 的版本号。不要复用已经失效的缓存。

## 结果
状态：已完成，2026年10月7日。100 条成功检查（含 72 个数值案例、16 次 Graph 分配变化）和 23 条预期拒绝通过；functional 的四项 opcheck 与 out 的两项检查通过。R08 接口 memcheck 零错误。

五个负载的 Graph 执行与原 native/full 接近。小 M 的 eager 预分配调用约 9.2～9.4 μs，自动分配约 13.4～13.5 μs；后续优先评估工作区复用。完整同轮数据见 [结果文档](../results/04_PyTorch接口.md)。

## 调整闭环
| 轮次 | 证据/问题 | 调整 | 验证 |
| --- | --- | --- | --- |
| R01→R02 | 环境缺少 Ninja，标准 extension load 无法构建 | 改为已安装 GCC12/nvcc 直接编译，链接当前 PyTorch ABI | 编译与 load_library 成功 |
| R03 | 空张量 data_ptr 均为 0，测试错误地判定为 alias | 空输出不按非空指针规则验收 | 重新运行完整测试 |
| R04→R05 | 手写 Autograd 实现碰触 FakeTensor 的 symbolic numel | 移除该分发，注册 Python Autograd/FakeTensor；CUDA 入口拒绝梯度输入 | R06 的 opcheck 与动态 eager 编译通过 |
| R06 | 工作区、cache、stream 和错误输入需独立证明 | 72 个数值案例、Graph 改 counts、设备恢复、调试合法性与缓存生命周期测试 | 全部通过 |
| R07/R08 | 接口成本和内存安全不能直接继承阶段 03 的性能结论 | 分开测 Graph/eager、自动/预分配调用，并 memcheck 接口工作区 | 数据见结果文档 |

下一阶段回放完整 gate/up → SiLU×乘法 → down 专家 MLP，保持 BF16 中间舍入与原参考一致，再判断 EP 接入价值。
