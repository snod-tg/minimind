# DeepEP机制迁移：验收结果

## 目的

确认共享scratch/推理池和Top-2按Rank去重能够训练，保留原PyTorch专家及数值规则，并量化真实收益。

## 执行方案

GPU5/7、6专家Top-2、hidden768、8层、专家中间2432、batch32/Rank、seq340、累积8、AdamW LR5e-4、裁剪1。FP32参数/状态，BF16 autocast，native通信和原PyTorch专家。四方案同初值、同每Rank数据、local_topk_sum辅助损失、每8微批一次梯度同步；4轮轮换顺序，每轮4个warmup更新和8个计时更新。计时不含初始化/warmup/checkpoint，不开profile或compile。

## 结果

### 数值、安全与完整模型

- 本地/服务器9项MiniMind CPU检查和2项通用Workspace检查通过。
- 去重/展开55组前向与反向边界通过，覆盖0/1/17/256/257/10880 Token、非对齐hidden、2/4 Rank、FP32/FP16/BF16；有效payload、metadata和映射与参考一致。
- memcheck、racecheck、synccheck各37组快速边界，0错误/0竞争/0同步错误。
- 原pack/gather/combine三类精度前向和梯度逐位一致；新增去重/展开三类精度torch.compile全图张量测试通过，expand的schema/autograd/FakeTensor/AOT检查通过。这不代表完整动态EP路由无CPU同步。
- 两Rank、6/8专家、同Rank/跨Rank/随机/空来源/全空共20个Rank用例：输出和路由权重梯度逐位一致，输入与专家梯度通过atol2e-5、rtol5e-4对拍。
- FP32两层模型：相对无EP DDP，首个同步梯度最大绝对差2.515e-8；两次AdamW后参数最大绝对差1.809e-6。
- 已训练8层模型：三种优化EP路径均通过长度1/17/128/340 Prefill、缓存Decode和3个提示各16 Token的完整对拍；logits/hidden/生成Token逐位一致，两Rank均448次调用、回退0。旧模型SHA256仍为457d87a9f46e417ac56ed36ef63b58ee0b84afba84b62a3f9ccd138896306b22。
- 16个正式性能窗口完成，退出0，占用检查通过；所有loss与梯度范数有限，训练回退0。

### 训练性能与显存

| 方案 | 有效Token/s中位数 | 秒/更新中位数 | 峰值allocated GiB | 峰值reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| 无EP DDP | 73084 | 1.4694 | 13.330 | 14.602 |
| 旧compact | 71926 | 1.4922 | 10.383 | 13.373 |
| 新compact（scratch/推理池） | 71806 | 1.4889 | 10.478 | 13.336 |
| Top-2 Rank去重 | 72813 | 1.4789 | 10.332 | 13.350 |


去重版与旧compact逐轮吞吐比分别为1.0344, 1.0061, 1.0027, 1.0088，配对中位提升0.74%。两组吞吐中位数之比为1.0123，不要把这两个统计量混用。相对新compact的配对中位提升1.27%。

去重版吞吐为无EP DDP中位数的99.63%，尚未证明训练快于无EP。峰值allocated低22.50%，这主要来自EP专家/优化器分片，不能全部归因于本轮去重。reserved是分配器缓存，不能等同于实际活跃张量或nvidia-smi进程占用。

12个更新（含warmup）汇总、4轮×2 Rank的派发有效行数由133,693,440降为110,841,448，下降17.09%；含本地Rank的派发记录，不是全NCCL流量下降比例。逐专家返回和其反向通信保持expanded布局。

### 推理工作区

Rank0：58.54→8.09 MiB（下降86.2%）；Rank1：44.97→8.06 MiB（下降82.1%）。输入最大长度340、batch1、8层共用同一device/stream缓存；输出拥有独立存储，后续层/Decode不会覆盖。此数值仅指EP推理工作区容量，不能套用到整模型总显存。新compact未体现稳定训练吞吐/峰值改善，其确定收益是推理缓存池。

### Profiling

独立4更新中，每Rank256次前向EP调用。Rank0/1的host_splits CPU累计176.1/126.3 ms；实际NCCL SendRecv kernel累计GPU时间2097.7/1806.7 ms。stage、record_param_comms和kernel条目有嵌套/重叠，不能相加计算端到端占比。通信值得优先处理；小映射算子的耗时不足以支持巨大吞吐收益。

## 调整闭环

首次测量被自身退出PID误判，监测修正后全量重测。去重确实降低派发行数，但总吞吐只小幅提高：输出仍逐专家返回，路由依然读CPU counts，Attention/专家计算与梯度同步未减少；PCIe NCCL和kernel/同步开销仍在。

BF16短训练轨迹不保证逐位重复：相同旧compact、同初值/数据独立复测12更新，最大loss差0.0698；去重相对旧compact最大0.0668，第一更新8个微批loss均完全一致。旧实现自身也出现该差异，尚未逐项定位重复性来源；它不能替代单次算子/梯度对拍，更不能据此宣称训练质量等价。FP32更新对拍、已训练模型前向和BF16训练有限性分别验收。

当前保留通用compact默认路径；Top-2可显式选择dedup_train。通信与计算重叠未计入收益，跨块流水/辅助损失异步重叠须另做候选对照。本轮未启动完整2 epochs重训，也没有质量评估结论。
