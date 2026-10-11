# 全模型 KN 权重：转换、直接加载与保存

## 目的
将线性层权重从连续[N,K]改为连续[K,N]，消除CUDA快路径反复转置；保留原默认BF16输出及模型共享权重语义。这是第3步算子验收之后、第4步接口与缓存的布局配套闭环。

## 执行方案
1. 由实际 MiniMind 模块类型发现全部nn.Linear权重，转换attention、router、专家gate/up/down、输出头；norm/bias/独立embedding按原语义保留。
2. 原模型lm_head与embedding共享权重。转换后使用LinearKN/TiedEmbeddingKN，共用同一KN Parameter，词嵌入仍输出[vocab,hidden]语义；这一个绑定embedding键与线性头一起保存KN。
3. checkpoint用显式format=minimind-linear-kn-v1、layout、aliases、config、state_dict记录；禁止重复转置或不带标记的模糊加载。原checkpoint不覆盖。
4. load_kn_checkpoint创建KN模块后直接加载KN值；EP可显式ep_subset=True读取本rank专家键。EP保存需所有rank调用，收集并校验共享参数，由rank0合并完整专家权重保存。save_kn_checkpoint保存所有线性参数为KN；已经KN的模型不再转置，后续model.state_dict中的线性参数本身也为KN。
5. EP保存从各rank收集参数，共享参数逐值校验，展开全专家布局并由rank0写完整checkpoint；重载只选择当前rank键。
6. 默认forward继续F.linear的原连续NK路径，首次建立一次NK缓存；CUDA输入可直接取canonical KN。梯度开启时用可求导转置，避免用脱离计算图的推理缓存训练。
7. 逐键恢复原布局对拍；单卡与双卡EP，对照原模型BF16 logits/KV/连续decode；测试共享参数梯度、更新、缓存失效和格式拒绝。

使用接口：
```python
from checkpoint_kn import load_kn_checkpoint, save_kn_checkpoint, prepare_native_caches
model = load_kn_checkpoint(model, "pretrain_ep_768_moe_kn.pth")
model = model.cuda().eval()
prepare_native_caches(model)
# 原推理继续使用inference_mode + BF16 autocast。
# Expert LinearKN.weight直接是连续[K,N]。
save_kn_checkpoint(model, "next_model_kn.pth")
```
加载时model为同架构原MiniMind实例；EP实例使用ep_subset=True。save_kn_checkpoint也能将原NK模型保存成KN格式。新的接口用于此轮全模型权重工作流；既有训练器的optimizer/resume、LoRA/量化和HF导出没有改造，不能把它们的旧保存调用视为已迁移。

## 结果
171个参数键，137个线性权重、138个KN键（多出共享embedding别名）；转换回原布局逐键值完全一致，原checkpoint SHA256不变。转换后的checkpoint SHA256为c2159fbff1813f3c0a60c6272dbc0862e279f1026f111672845c973f7134ccf5。

最终单卡10个(top1/top2、不同batch/序列长度)对照、40步decode，双卡EP每rank10个对照、40步decode，并验证全模型专家合并保存及重新加载：logits/KV及连续生成逐位一致。另通过共享权重梯度、SGD更新、推理缓存刷新、重新保存/回读和2类错误格式拒绝。原BF16 reduced-precision设置保持开启。测试输入为固定种子的模型回归输入，未将投影回放冒充模型回归。

## 调整闭环
| 发现 | 调整 | 结果 |
| --- | --- | --- |
| 简单转置所有2D张量会破坏词嵌入与头部共享 | 基于模块语义发现线性层，保留别名与同一Parameter | KN形状及绑定关系通过单卡/EP检查 |
| KN转置视图改变PyTorch原权重布局 | 原生兼容使用一次性连续NK缓存，CUDA保留KN | 原BF16模型逐位一致；兼容路径额外占用NK缓存，未宣称全部转置成本消失 |
| 缓存可能在权重更新/重新加载后过期 | 版本/device/dtype检查及load后失效，清理autocast缓存 | 更新与回读通过 |
| 新格式再次转换或误用旧loader | 显式format/layout，双转置和旧格式拒绝 | 两类拒绝测试通过 |



EP保存闭环：R07发现已经KN的EP模型重复加载仍按原NK模块发现布局；修正为复用已有布局，并校验checkpoint契约。R08/R09最终复测通过，EP全模型保存、恢复原布局和重新加载全部通过。
