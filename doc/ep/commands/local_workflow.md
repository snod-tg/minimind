# MiniMind EP 操作命令

## 目的
在 MiniMind 根目录完成依赖安装、验证、独立训练与评估。

## 执行方案
以下命令在 MiniMind 根目录使用同一个已有 PyTorch 环境执行。

```bash
python -m pip install --no-deps --no-build-isolation -e ../my_cuda_kernels
python ../my_cuda_kernels/script/build.py ep
python -m unittest tests.ep.test_local_contracts -v

# 仅查看启动配置，不执行训练
python -m scripts.ep.train --run-id ep6_top2_new --gpus 2,3 --tmux --dry-run

# 先查看 GPU 占用；从 0..3 或 4..7 同一组内选择两张
nvidia-smi
python -m scripts.ep.train --run-id ep6_top2_new --gpus 2,3 --tmux
tmux attach -t ep6_top2_new

# 单独验证可配置专家数与 Top-K；选择实际空闲卡
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --standalone --nproc_per_node=2 --module tests.ep.validate_configurations   --experts 6 --topks 1 2 3 --library ../my_cuda_kernels/build/ep/current/my_cuda_ep.so --output runs/ep_config_validation

# 模型同输入基线对照；不传 checkpoint 则使用初始化模型
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --standalone --nproc_per_node=2 --module tests.ep.validate_model   --experts 4 --checkpoint out/pretrain_ep_768_moe.pth   --library ../my_cuda_kernels/build/ep/current/my_cuda_ep.so --output runs/ep_model_validation

python -m scripts.ep.run_benchmark --gpus 2,3 --experts 4 --checkpoint out/pretrain_ep_768_moe.pth   --library ../my_cuda_kernels/build/ep/current/my_cuda_ep.so --output runs/ep_benchmark

# 转换完整模型参数，显式提供该模型的配置；不转换优化器恢复状态
python -m scripts.ep.convert_checkpoint_kn --input out/pretrain_ep_768_moe.pth   --output out/pretrain_ep_768_moe_kn.pth --config path/to/training_config.json --report runs/kn_conversion.json
```

默认训练配置是 configs/ep/pretrain_6_top2.json。调整专家数和 Top-K 可复制配置后传 --config；总专家数必须能被两 Rank 整除。新的 run-id 独立保存模型和两 Rank 恢复状态，已存在的运行目录拒绝覆盖。CXX/CC 使用已选工具链，可指定 SCVideo GCC12；启动器通过 tmux 传递该选择。缺库或配置非法直接报错，无训练回退。

每次运行包含 config、logs、checkpoints/full、checkpoints/resume、state。训练结束检查 state/training_exit_code.txt=0、完整模型目录中 training_summary.json 的 completed_epochs=2 和所有 Rank routing_fallbacks=0，另确认完整模型及两份 Rank 恢复文件存在。退出成功与模型质量验证分开。

## 结果
本轮通过本地 CPU 契约与训练命令预览；未启动新训练，正在运行的服务器训练沿用其启动版本。

## 调整闭环
修改路由后先跑算子和多卡边界测试，再跑真实文本短训练及恢复测试，最后独立正式训练。更改动态通信/计数路径必须重新测量模型整体耗时；编译边界变化先验证梯度与参数保存。
