# MoE Grouped GEMM：prefill / decode

## 目的
针对真实 MiniMind 专家投影，分别优化大 M 的紧凑 Prefill 和小 M 的 Decode；第三轮新增 kernel9_prefill / kernel9_decode；第四轮按80%CUTLASS门槛仅优化M137 gate/up。Prefill实现保留，Decode最新默认数据为2026-10-09。

## 执行方案
借鉴本地 DeepGEMM 的 contiguous / masked 分工、分组 Tile 调度和资源约束。在 RTX4090D 上使用 cp.async、ldmatrix、mma.sync，不直接使用面向 SM90 的 TMA/WGMMA。

- Prefill：紧凑 A[M,K]、B[E,K,N]、C[M,N]；GPU counts 生成前缀和与 Tile 元数据。向量化加载、XOR 共享内存布局、2～4阶段流水线、寄存器双缓冲，以及跨 stage 预取和共享内存中转的连续输出存储。最终保留8个配置，按 M/N/K 选路；专家的 M 分配可变化。
- Decode：M=0～256，计算 CTA 内直接验证 counts 和定位专家，消除独立元数据启动；保留7个原小 Tile/Split-K 配置，新增有效Tile/persistent配置，gate/up M>64正常默认选用，部分和为 FP32，最后只转换一次输出。支持紧凑布局及 masked A[E,capacity,K]，固定容量内实际总 M 可以变化。
- 固定三组 (K,N)=(768,2432)/(2432,768)/(1024,1024)，E=1～16；FP16/BF16 输入输出、FP32 累加。共享辅助逻辑统一放在 common.h。实现仅含前向。

## 结果
### 同轮 CUTLASS 对比
Decode为2026-10-09第四轮，Prefill为2026-10-07第三轮（实现未变）；各行只与该行同轮CUTLASS比较。GPU4 RTX4090D、CPU28；CUDA12.4 / GCC12.4 / Python3.13.5 / torch2.9.0+cu128，TF32关闭。每形状扫描 CUTLASS device、host 与 extra 家族：162个有效配置/CTA组合，另6个不支持组合；最优候选复测后与最终默认算子同轮计时。自己的正式路径不使用逐场景手工指定最优配置。

CUDA Graph 64节点，9轮轮换，每轮10次重放，取中位数。吞吐占比= CUTLASS耗时 / 本算子耗时 ×100%。CUTLASS 描述符预建；Prefill“计算”不含元数据启动，“完整调用”包含；Decode 包含 GPU counts 和 Split-K 归约。均为已打包输入上的投影计时。

| 场景 | 专家 M | (K,N) | 最终算子计算 μs | 完整调用 μs | CUTLASS μs | CUTLASS 吞吐占比 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| decode single gate/up | [0, 1] | (768,2432) | 3.830 | 3.834 | 4.626 | 120.8% |
| decode single down | [0, 1] | (2432,768) | 4.563 | 4.566 | 10.634 | 233.0% |
| decode batch gate/up | [8, 8] | (768,2432) | 4.842 | 4.846 | 4.725 | 97.6% |
| decode batch down | [8, 8] | (2432,768) | 4.878 | 4.883 | 10.635 | 218.0% |
| decode 29 gate/up | [18, 11] | (768,2432) | 5.918 | 5.918 | 4.832 | 81.6% |
| decode 29 down | [18, 11] | (2432,768) | 6.048 | 6.050 | 10.846 | 179.3% |
| decode 137 gate/up | [64, 73] | (768,2432) | 8.283 | 8.283 | 7.154 | 86.4% |
| decode 137 down | [64, 73] | (2432,768) | 12.576 | 12.570 | 11.480 | 91.3% |
| prefill medium gate/up | [1480, 1432] | (768,2432) | 84.094 | 85.229 | 79.285 | 94.3% |
| prefill medium down | [1480, 1432] | (2432,768) | 93.797 | 94.523 | 79.811 | 85.1% |
| prefill large gate/up | [10986, 11477] | (768,2432) | 631.168 | 633.157 | 594.232 | 94.1% |
| prefill large down | [10986, 11477] | (2432,768) | 647.074 | 650.221 | 606.822 | 93.8% |
| regression 2048 | [64,96,160,192]×4 | (1024,1024) | 44.429 | 48.290 | 41.477 | 93.4% |
| variable 4097 | 含空专家，最大3136 | (1024,1024) | 74.835 | 78.760 | 67.622 | 90.4% |

大 Prefill 约93.8%～94.3%，中等 Prefill down 为85.1%；M137 gate/up现为86.36%，独立复测86.02%，完成本轮80%门槛；具体调度与验收见[Decode第四轮分析](grouped_gemm_decode.md)。极小 M 的 down 可高于本次 CUTLASS 配置，不能由此推断优于所有开源实现，也不能推断整体模型收益。

![最终默认算子与CUTLASS](../figures/grouped_gemm_prefill_decode.png)

### 数值契约与兼容路径
按用户确认，默认模型继续保留原 BF16 结果和 PyTorch reduced-precision reduction 设置。手写 CUDA 路径按 FP32 累加参考验收；原生兼容路径 NativeBF16GroupedGemm 使用原连续 NK 投影及原 BF16 autocast，在 KN 权重准备后缓存 NK 权重，权重变更需刷新缓存。

最新CUDA回归：2160组合/分配、384结构化输入、75次Graph有效M/分配/非法counts、1368次真实投影配置检查，FP32参考失败为0，另有CPU double抽查；新增82项M137动态专家检查。第三轮34组安全检查通过，第四轮另23组Decode安全检查全部零错误，7项Decode独立CLI通过。BF16参考阈值 atol=2e-4、rtol=.008；FP16 rtol=.001，另有限值检查。

原 BF16 对照单独审计：最新真实投影有24个配置检查未过原逐元素阈值，对应2个唯一回放投影；最大相对RMS约0.285%。兼容路径342次真实投影对拍逐位一致，并验证空输入、非法 counts、权重更新和独立 autocast。兼容路径保留原归约偏差，因此相对 FP32 参考仍可不同。这两项验证不能互相替代。

兼容路径逐专家调用 PyTorch，GPU counts 会发生主机同步，不能复用手写 CUDA 吞吐数据；动态 GPU counts 不支持该兼容路径的 Graph 捕获。原模型默认后端保持 original，本轮未把新 CUDA 路径直接替换为模型默认后端。

兼容路径的单独 eager wall time（μs，含分配；7轮中位数，CPU counts 与GPU counts区分）：

| M | 投影 | 原实现CPU counts | 兼容CPU counts | 兼容GPU counts |
| ---: | --- | ---: | ---: | ---: |
| 1 | gate/up | 20.120 | 27.775 | 40.671 |
| 1 | down | 20.367 | 27.923 | 40.761 |
| 137 | gate/up | 32.456 | 39.537 | 53.941 |
| 137 | down | 33.708 | 41.178 | 55.860 |
| 2912 | gate/up | 100.091 | 100.970 | 125.873 |
| 2912 | down | 96.583 | 97.625 | 121.467 |
| 22463 | gate/up | 817.024 | 818.106 | 843.542 |
| 22463 | down | 653.483 | 655.453 | 681.844 |

此表保留原BF16语义，不能与上面Graph中的FP32手写算子直接混算加速比。

### Masked 整理成本
8组紧凑/预整理masked/GPU pack+decode+unpack单独比较。M=1 gate/up 为4.210 / 4.221 / 6.470μs，down 为5.018 / 5.027 / 7.286μs；预整理布局没有天然消除原输入的转换成本，当前模型不能把 pack/unpack 当作免费操作。

## 调整闭环
| 观察 | 调整 | 实测与结论 |
| --- | --- | --- |
| 大 Tile 超过默认共享内存额度 | 显式动态共享内存 opt-in | 启动错误修复，重新冒烟；最终保留较小配置 |
| 非连续 NK 参考改变原 PyTorch 算法；长 K 原 BF16 与 FP32参考冲突 | 参考使用连续 NK，数值契约拆成原生兼容与 FP32验证 | CUDA FP32参考通过；原生兼容342/342逐位通过，未修改默认归约策略 |
| 寄存器输出写回零散 | 复用已消费的共享内存并连续16B输出 | 调参中 Prefill up 约90→84μs，保留并全量复测 |
| stage切换中的加载延迟 | 下一stage操作数提前进入寄存器，再执行当前stage最后的 MMA | 调参中 medium down 约97→92μs；正式同轮为93.797μs |
| Decode单独元数据与空 Tile开销 | counts读取合入 CTA、按形状选择小Tile和Split-K | 小M down改善明显；第四轮M137 up有效Tile+persistent默认达到86.36% |
| 兼容路径重复进入 BF16 autocast | 已有同dtype autocast时复用上下文 | 位级对拍及完整调用延迟再次验证 |
| 需要确认剩余吞吐差距 | 按 guide 尝试 NCU | ERR_NVGPUCTRPERM，未获得硬件指标；不能宣称bank conflict或stall已实测消除 |

当前差距的候选原因是中等 M 尾块效率、warp 分工与流水线延迟，以及小 M 的任务映射和分组边界读取成本。编译记录无寄存器溢出，但尚缺硬件计数器证据。第四轮仅处理M137 gate/up，正式及独立复测均超过80%；Prefill/common.h源码未变。后续新CUDA路径需回归第4～7步的接口、MLP、EP与模型测量。
