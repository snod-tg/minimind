# Decode MoE Grouped GEMM：有效Tile与persistent调度

## 目的
仅优化M=137的gate/up投影，目标为所测最优CUTLASS吞吐的80%。这是“算子实现与优化”第3步的第四轮闭环；Prefill与common.h保持原实现。

## 执行方案
参考本地DeepGEMM的`deep_gemm/include/deep_gemm/scheduler/gemm.cuh`：根据GPU有效M跳过空专家，固定CTA集合按步长领取下一个Tile。4090D采用cp.async/ldmatrix/mma.sync，未使用SM90 TMA/WGMMA。

新Decode配置BM32/BN64/BK64、4warps、3阶段流水线、寄存器双缓冲；GPU counts计算各专家有效Tile前缀，E=2用shuffle快速定位，E<=16用warp前缀和。紧凑网格上界`(ceil(M/BM)+min(M,E)-1)*(N/BN)`并限制最多2CTA/SM，循环按gridDim.x前进；无独立元数据kernel或CPU读取counts。M137、E2目标由原380个容量CTA改为228个上界CTA，实际190个有效Tile。

复用common.h的完整K加载掩码、跨stage寄存器预取和共享内存中转16B写回；每次复用共享内存前同步CTA。gate/up M>64默认选择新配置，小M继续原Split-K；共保留8配置，专家分配未写死。Decode仍支持总M=0～256、E=1～16、FP16/BF16、三组固定(K,N)=(768,2432)/(2432,768)/(1024,1024)，仅前向。

## 结果
2026-10-09，GPU4 RTX4090D/114SM、CPU28；Python3.13.5 / torch2.9.0+cu128 / nvcc12.4.99 / GCC12.4，sm89/O3/C++17、TF32关闭、无fast-math。CUDA Graph64节点，9轮轮换，每轮10次重放，取中位数。计入GPU counts及必要的Split-K归约，输入已按专家打包、权重为连续KN。

CUTLASS device/host/extra每形状扫描162个有效配置/CTA组合，另6个不支持组合，top5复测选最佳；描述符预建。吞吐占比=CUTLASS耗时/本算子耗时×100%。自己的路径使用正常默认选路。

| 场景 | 专家 M | (K,N) | 默认算子 μs | CUTLASS μs | CUTLASS吞吐占比 |
| --- | --- | --- | ---: | ---: | ---: |
| decode single gate up | [0, 1] | (768,2432) | 3.830 | 4.626 | 120.76% |
| decode single down | [0, 1] | (2432,768) | 4.563 | 10.634 | 233.03% |
| decode batch gate up | [8, 8] | (768,2432) | 4.842 | 4.725 | 97.59% |
| decode batch down | [8, 8] | (2432,768) | 4.878 | 10.635 | 218.01% |
| decode 29 gate up | [18, 11] | (768,2432) | 5.918 | 4.832 | 81.64% |
| decode 29 down | [18, 11] | (2432,768) | 6.048 | 10.846 | 179.34% |
| decode 137 gate up | [64, 73] | (768,2432) | 8.283 | 7.154 | 86.36% |
| decode 137 down | [64, 73] | (2432,768) | 12.576 | 11.480 | 91.28% |

目标默认8.283μs，达到86.36%；独立复测8.238μs / CUTLASS 7.086μs，86.02%。本次8场景均超过80%，范围不外推至任意专家分配或模型端到端收益。目标CUTLASS winner为host调度、BM64/BN64/BK32、warp32×32、3阶段、228CTA。

![最终Decode与CUTLASS](../figures/grouped_gemm_decode.png)

数值回归：2160组形状/配置、384组结构化输入、75组Graph/有效M/非法counts、1368次真实投影配置检查，FP32参考失败为0，另有CPU double抽查；M137动态专家分配另有82项通过，包含E=1/2/4/8/16、空专家、[65,72]、极端分配、Graph内counts变更和非法counts不写输出。23组memcheck/racecheck/synccheck均零错误；7项独立CLI通过。BF16阈值atol=2e-4/rtol=.008，FP16 rtol=.001，阈值未放宽。

FP32累加仍不同于原PyTorch默认BF16降精度归约：24个真实配置检查未过原兼容阈值，对应2个唯一投影，最大相对RMS0.285%。原BF16兼容路径及模型默认后端保留，本轮CUDA未默认替换原模型。原兼容验证与FP32验证不能互相代替。

M137 gate/up同轮布局比较：compact8.240μs、预整理masked8.170μs、pack+decode+unpack12.122μs。继续保留当前compact布局；预整理masked略快不能抵消即时转换成本。硬件计数器权限不足，不能宣称bank conflict或stall已经实测消除。

## 调整闭环
先移植完整K流水线，改善不足；减少容量网格的空CTA后，再比较flat与persistent。1CTA/SM并行度不足，2CTA/SM实测更合适，最终删去实验配置。新旧库同时加载暴露GNU unique配置表冲突，隔离新数组符号并重新构建，重新完成数值、计时和安全验收。两次正式测量均超过用户80%门槛，完成本轮；新CUDA接口、完整MLP、EP和模型接入回归属于后续第4～7步。

编译运行命令见[guide](../guide.md)的“20261009：Decode第四轮优化”。Prefill和兼容路径详情见[Prefill/Decode分析](grouped_gemm_prefill_decode.md)。
