# DeepEP机制迁移：训练EP优化

## 目的

以默认模型尺寸、6专家Top-2、2 Rank为问题场景，先保留原PyTorch专家，优化路由/通信编排，减少重复派发和推理工作区占用。以相同初值、数据、辅助损失和更新频率比较训练吞吐，避免将模型规模、数值契约或compile差异算作EP收益。

## 执行方案

### 第1轮：定位瓶颈并明确内存生命周期

为rank_layout、pack、counts_exchange、host_splits、dispatch、expert_layout、experts、return和combine标注profile范围。profile只用于定位；性能数字来自关闭profile的独立测量。

训练先删除立即被functional算子返回值替换的空缓冲区。分组的perm/inverse/counts每次前向独立分配，反向保存的映射不得被下一层覆盖；只复用positions、block_counts、offsets等非反向保存的scratch。推理在同一device/stream上的串行层共用容量按2次幂增长的工作区，最终输出仍独立拥有存储。新Token/new forward必须重新计算路由。

### 第2轮：Top-2按Rank去重并直接展开

1. 稳定分组专家分配，标记每个Token/目标Rank的第一条记录；GPU计算前缀与去重counts。
2. 同一Token的两个专家在同一Rank时只打包一次输入，元数据保存一至两个本地专家ID；counts通信同时携带unique和expanded数量。
3. compact输入按unique数量通信；接收端生成虚拟专家记录及wire映射，通过一次gather直接得到专家分组输入，省去重复的展开输入中间张量。
4. 专家仍调用原PyTorch模块。输出维持逐专家返回，再按原顺序乘路由权重和Top-K合并；不在远端预先合并输出，以保留BF16舍入。
5. 映射独立属于本次forward；返回通信沿用expanded splits。反向在接收端累加共享wire输入梯度，再发送一次去重输入梯度，源端恢复Token梯度。

通用CUDA打包/展开/autograd属于my_cuda_kernels；MiniMind只负责NCCL、模型专家、路由handle和训练。compact保持可配置的专家数/Top-K；dedup_train明确要求Top-2，其他Top-K使用compact，无隐式训练回退。

### 第3轮：安全与收益验收

单算子55组值/梯度边界，Compute Sanitizer三项；新增算子的FakeTensor/AOT/torch.compile检查；两Rank空输入、空接收Rank和6/8专家对拍；FP32两层模型梯度与两次AdamW更新；完整已训练8层模型Prefill/Decode逐位对拍。最后四种方案轮换4次：无EP DDP、旧compact、新compact、训练去重版，每次4个warmup更新、8个计时更新。

通信重叠需要独立工作与单独验收。当前模型无共享专家分支，接收→本地专家→回传→合并形成同一批次依赖链；辅助损失虽独立，本次profile尚未给出其重叠收益。本轮不凭新增stream声称重叠收益。跨块流水会改变缓冲区/反向通信调度，应作为下一轮独立候选。

具体命令见[操作指令](../commands/14_DeepEP优化闭环.md)。

## 结果

见[本轮结果](../results/14_DeepEP优化闭环.md)。安全、数值、吞吐、缓存和模型总显存分别统计，不将缓存缩减比例套用到整模型。

## 调整闭环

初次4方案测试在进程退出时被监测误判：nvidia-smi尚显示旧PID、/proc命令已消失。监测现限定本次output，并仅对已观察属于本任务且命令消失的PID保留归属；PID显示其他命令仍判为外部任务。增加CPU回归测试，正式性能全部重测。

DeepEP主线的Hopper/NVLink依赖不能直接套用到4090D PCIe；迁移的是按Rank去重、专家布局和handle生命周期。动态PyTorch专家切片仍需要CPU读取counts，完整EP路由不能宣称fullgraph无CPU同步；functional张量算子的compile能力另行测试。吞吐变化须结合逐轮差异和显存评估后决定默认实现。

参考：[DeepEP接口](https://github.com/deepseek-ai/DeepEP#interfaces-and-examples)、[EPHandle/dispatch/combine](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffers/ep.py)。
