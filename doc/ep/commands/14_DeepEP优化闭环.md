# DeepEP机制迁移：操作指令

## 目的

按照本地实现→本地检查→指定文件同步→服务器正式构建→数值/安全→同条件性能的顺序复现。

## 执行方案

### 本地检查与同步

在MiniMind根目录执行CPU检查：

```bash
python -m unittest discover -s tests/ep -p 'test_*.py'
```

在Code根目录只同步本轮相关文件；绘图和文档也按同样方式同步，避免覆盖另一项目的GEMM修改：

```bash
rsync -aR minimind/model/ep/routing.py minimind/scripts/ep/benchmark_training.py \
  minimind/scripts/ep/profile_training.py minimind/scripts/ep/train.py \
  minimind/trainer/train_pretrain.py minimind/tests/ep/test_workspace_pool.py \
  minimind/tests/ep/test_monitor.py minimind/tests/ep/validate_rank_dedup.py \
  minimind/tests/ep/validate_trained_checkpoint.py \
  my_cuda_kernels/dev/cuda/ep_routing.cu my_cuda_kernels/dev/cuda/ep_routing_torch.cpp \
  my_cuda_kernels/dev/pytorch/my_cuda_kernels/ep.py \
  my_cuda_kernels/dev/pytorch/my_cuda_kernels/ep_ops.py \
  my_cuda_kernels/tests/ep/test_workspace.py my_cuda_kernels/tests/ep/validate_dedup.py \
  my_cuda_kernels/tests/ep/validate_dedup_compile.py \
  Server-Internet:/data/user4/tongleiwen/
```

### 服务器环境与构建

```bash
conda activate SCVideo
export EP_GCC_ROOT="$CONDA_PREFIX"
conda deactivate
cd /data/user4/tongleiwen/tilelang-3032
source .venv/bin/activate
export CC="$EP_GCC_ROOT/bin/x86_64-conda-linux-gnu-cc"
export CXX="$EP_GCC_ROOT/bin/x86_64-conda-linux-gnu-c++"
export CUDA_HOME=/usr/local/cuda
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
cd ../my_cuda_kernels
python script/build.py ep
```

先用nvidia-smi确认GPU5/7空闲；不结束其他用户任务。

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv
mkdir -p result/ep_deepep_20261011
CUDA_VISIBLE_DEVICES=5 python tests/ep/validate_dedup.py \
  --output result/ep_deepep_20261011/dedup.json
CUDA_VISIBLE_DEVICES=5 python tests/ep/validate_functional.py \
  --library build/ep/current/my_cuda_ep.so --output result/ep_deepep_20261011/functional.json
CUDA_VISIBLE_DEVICES=5 python tests/ep/validate_dedup_compile.py \
  --output result/ep_deepep_20261011/compile.json
for tool in memcheck racecheck synccheck; do
  CUDA_VISIBLE_DEVICES=5 "$CUDA_HOME/bin/compute-sanitizer" --tool "$tool" \
    --error-exitcode 91 python tests/ep/validate_dedup.py --quick
 done
```

### 两Rank数值与模型验收

```bash
cd ../minimind
CUDA_VISIBLE_DEVICES=5,7 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module tests.ep.validate_rank_dedup --output runs/ep_deepep_boundaries_20261011
MODEL=out/ep6_top2_compile_20261009/pretrain_ep6_top2_compile_20261009_768_moe.pth
CONFIG=out/ep6_top2_compile_20261009/training_config.json
for mode in compact_legacy compact dedup_train; do
  CUDA_VISIBLE_DEVICES=5,7 python -m torch.distributed.run --standalone --nproc_per_node=2 \
    --module tests.ep.validate_trained_checkpoint \
    --checkpoint "$MODEL" --training-config "$CONFIG" --resume-dir checkpoints \
    --reference runs/ep6_top2_acceptance_20261011/02_bos_original --prepend-bos \
    --routing "$mode" --output "runs/ep_deepep_trained_20261011/$mode"
 done
```

### 独立profiling与正式性能

```bash
python -m scripts.ep.profile_training --gpus 5,7 --output runs/ep_deepep_profile_20261011
python -m scripts.ep.benchmark_training --gpus 5,7 \
  --output runs/ep_deepep_acceptance2_20261011 \
  --variants ddp ep_legacy ep_compact ep_dedup \
  --warmup-updates 4 --updates 8 --repeats 4
python -m scripts.ep.plot_training_baseline --run runs/ep_deepep_acceptance2_20261011 \
  --output figures --stem ep_deepep_training
```

长测量放入tmux，退出码单独记录；本轮正式性能未打开profile或compile。复跑必须换新的output，保留已完成测量。新配置可显式选ep_routing=dedup_train；本轮不启动完整预训练。

## 结果

见[验收结果](../results/14_DeepEP优化闭环.md)。

## 调整闭环

CPU测试通过后同步，构建成功后运行GPU数值，安全检查通过才接受性能。选定卡出现其他任务时该窗口无效；测量监测绑定本次output并记住自身已退出PID，不能把别的benchmark进程认作本任务。
