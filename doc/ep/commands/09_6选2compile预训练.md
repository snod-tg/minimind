> 历史实验记录：保留该轮配置和结果；当前执行入口见项目 README。

# 6 选 2 训练指令

## 目的
验证原尺寸真实数据训练和 checkpoint，再用 tmux 执行独立正式预训练。

## 执行方案
```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage09_compile_20261009
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
export OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export EP_TRAIN_GPUS=2,3
bash run_smoke.sh
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --standalone --nproc_per_node=2 verify_smoke_checkpoint.py --library ../stage08_ep_20261009/results/R27_final_build/my_cuda_ep.so --output results/REPRO_trained_reload
# 正式仓库接入相同已验证源码后启动；重复启动会被保护检查拒绝。
bash launch_tmux.sh
# 查看进度。
tmux attach -t ep6_top2_compile_20261009
# 从会话分离：Ctrl-B，然后 D。
tail -n 30 pretrain_ep6_top2_compile.log
# 完成后退出码0才表示训练进程正常完成。
cat training_exit_code.txt
```

run_train.sh 包含完整固定超参数；每 Rank batch32、seq340、2epochs、累积8、BF16、6选2、从头训练。正式权重独立保存为 /data/user4/tongleiwen/minimind/out/ep6_top2_compile_20261009/pretrain_ep6_top2_compile_20261009_768_moe.pth；两 Rank 恢复状态使用同样独立前缀。

## 结果
真实文本短训练 R05完成，优化EP回退0；R06重新加载通过；正式tmux在GPU2/3启动，第1/2个epoch已完成1300/19848个batch，loss4.1534、回退0；训练尚在进行。

## 调整闭环
短训练使用独立名称，不替代正式训练；保存和重新加载通过后再启动 tmux。动态控制流有显式编译边界，遇到真实错误退出并分析，禁止静默切回旧 EP。

正式训练首个1000-batch保存完成：完整新模型与两个Rank恢复状态均已生成；原模型SHA256未变。训练完成/异常通知已设置，每10分钟只读检查一次，正常进行时保持安静。
