# MiniMind 与算子项目的关联

## 目的
让模型实现、训练入口和实验结论都能在 MiniMind 内找到，稳定调用独立算子包。

## 执行方案
| 目录 | 职责 |
| --- | --- |
| model/model_minimind.py | 模型、原 Router/专家、EP 模型定义 |
| model/ep/routing.py | MiniMind 路由、NCCL、负载均衡损失和专家调用 |
| model/backends | 专家权重缓存、完整 MLP、可选 Grouped GEMM 适配 |
| model/checkpoint_kn.py | 模型 KN 权重转换/加载/保存 |
| trainer | 实际训练器 |
| configs/ep | 独立可编辑的专家数/Top-K 和训练配置 |
| scripts/ep | 训练/tmux、benchmark、绘图和权重转换入口 |
| tests/ep | 模型、分布式梯度与配置边界测试 |
| analysis / figures | EP、MLP、模型分析与配图 |
| doc/ep/design / commands / results | 目的方案闭环、具体指令、结果 |
| runs/<run_id> | config、logs、checkpoints/full、checkpoints/resume、state |

两个仓库分别为同级 minimind 和 my_cuda_kernels。用训练所用解释器安装后者的 editable 包；调用通过普通 import，库默认由包解析，也可用 library 参数覆盖。算子仓库不导入模型，MiniMind 不复制 CUDA 源码或通过 sys.path 加载阶段文件。

训练路径：原 Router → compact CUDA pack/gather → NCCL → 原 PyTorch 专家 → NCCL → 融合 combine → 原辅助损失/反向/梯度同步。每层每步重新路由；工作区只缓存空间。推理 Grouped GEMM 是独立的显式后端选项，不混入本次训练入口。

## 结果
当前 API、模型适配、脚本和文档已按职责归位。历史阶段快照只用于理解对应实验；当前入口不导入快照。

## 调整闭环
目录修改先检查包导入、模型参数键和权重布局，再跑 CPU 契约测试；CUDA 环境补充算子与多卡训练回归。模型全层 BF16 误差问题继续保留为独立准入项；本轮不改变数值算法或服务器运行版本。
