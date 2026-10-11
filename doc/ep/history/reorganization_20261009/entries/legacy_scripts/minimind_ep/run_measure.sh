#!/usr/bin/env bash
set -euo pipefail
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage07
export CUDA_VISIBLE_DEVICES="${BENCH_GPUS:-4,5}" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BENCH_CPU_BASE="${BENCH_CPU_BASE:-28}"
K9_RUN="${K9_RUN:-R01_model_timing}"
test ! -e "results/$K9_RUN"
mkdir -p "results/$K9_RUN"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader > "results/$K9_RUN/gpu_before.csv"
nvidia-smi topo -m > "results/$K9_RUN/topology.txt"
/data/user4/tongleiwen/tilelang-3032/.venv/bin/python monitor_gpu.py --gpus "$CUDA_VISIBLE_DEVICES" --output "results/$K9_RUN/gpu_monitor.jsonl" --stop "results/$K9_RUN/monitor.stop" &
K9_MONITOR_PID=$!
trap 'touch "results/$K9_RUN/monitor.stop"; wait "$K9_MONITOR_PID"' EXIT
/data/user4/tongleiwen/tilelang-3032/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 benchmark_model.py \
 --stage6 /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage06 \
 --library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/R05_dispatch_build/my_cuda_kernel9.so \
 --repo /data/user4/tongleiwen/minimind --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth \
 --baseline /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/results/R04_baseline \
 --output "results/$K9_RUN" "$@" > "results/$K9_RUN.console.log" 2>&1
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader > "results/$K9_RUN/gpu_after.csv"
