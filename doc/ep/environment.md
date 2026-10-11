# 本轮整理环境

## 目的
区分本地目录/导入验证和服务器 CUDA 数值/性能验证。

## 执行方案
本地：macOS，Python 3.13.12，PyTorch 2.10.0，Transformers 5.3.0；解释器为 /opt/homebrew/Caskroom/miniconda/base/envs/py313/bin/python。使用已有环境，editable 安装不安装额外依赖。

服务器历史实验：RTX 4090D、CUDA PyTorch 2.9.0+cu128、Python 3.13.5，CUDA Toolkit 12.4，SCVideo GCC12.4。服务器编译选择 CXX，部署时必须使用训练所用解释器重新构建并验证 ABI。

## 结果
本轮完成 CPU 导入、权重转换/缓存与配置测试以及编译/训练命令预览。本地没有 CUDA GPU，本轮不计新增 CUDA 编译、安全或性能测试结果。

## 调整闭环
更换解释器、PyTorch ABI、CUDA 或源码后重新构建；GPU 回归单独记录。详细历史条件与当轮测量数据一起保留，避免跨条件比较。

[历史测试环境](environment_history_20261007.md)

## 2026-10-11 服务器验收与基线

物理GPU5/7，两张RTX4090D，当前测量进程独占这两张卡；Python3.13.5、PyTorch2.9.0+cu128，已有SCVideo GCC12构建EP库。已安装正式editable包，默认库在my_cuda_kernels/build/ep/current。数值对拍与生成带训练BOS；6专家Top-2，FP16导出→FP32参数，BF16 autocast。性能时不启用模型compile；实际时序/显存定义见第12步，避免与训练速度或其他模型的旧数据混算。

## 2026-10-11 公平短训练环境

RTX4090D物理GPU5/7，2 Rank；hidden768、8层、专家中间维度2432、6专家Top-2。每Rank batch32、seq340、梯度累积8、LR5e-4、裁剪1、AdamW；FP32参数与优化器状态、BF16 autocast、原PyTorch专家。三组统一eager，不启用compile。全局每次更新512条序列，最多174080个padding内Token；吞吐主指标只计移位后labels非-100的有效Token。

Python3.13.5、PyTorch2.9.0+cu128；GPU5/7占用监测通过。TF32关闭；默认原生通信精度，使用已有EP原生库。辅助损失明确设为local_topk_sum；训练同步频率对齐。只读取真实数据与独立随机初值，已有训练模型不参与参数更新。

## 2026-10-11 DeepEP机制迁移

GPU5/7沿用公平短训练条件；4方案轮换4次，关闭profile和compile计时。独立profile使用同模型、2个warmup/4个测量更新。单算子三类精度与Compute Sanitizer在GPU5验证，独立torch.compile测试在空闲GPU4完成；完整模型与通信边界用GPU5/7验证。正式EP源代码仅取自my_cuda_kernels/dev/cuda，CUDA12.4、sm89、GCC12、--fmad=false、Torch ABI=1；模型继续使用原PyTorch专家。
