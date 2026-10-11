#!/usr/bin/env bash
set -euo pipefail
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04_07_split_20261009
PY=/data/user4/tongleiwen/tilelang-3032/.venv/bin/python
LIB=results/R01_build/my_cuda_grouped_split.so
CKPT=/data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth
BASE=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/results/R04_baseline
export CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 BENCH_CPU_BASE=28 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 validate_ep.py --source source --library "$LIB" --checkpoint "$CKPT" --repo /data/user4/tongleiwen/minimind --baseline "$BASE" --output results/R08_final_validation > R08_final_validation.log 2>&1
mkdir results/R09_final_model
"$PY" monitor_gpu.py --gpus 4,5 --output results/R09_final_model/gpu_monitor.jsonl --stop results/R09_final_model/monitor.stop &
MONITOR_PID=$!
trap 'touch results/R09_final_model/monitor.stop; wait "$MONITOR_PID"' EXIT
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 benchmark_model.py --stage6 . --library "$LIB" --checkpoint "$CKPT" --repo /data/user4/tongleiwen/minimind --baseline "$BASE" --output results/R09_final_model > R09_final_model.log 2>&1
touch results/R09_final_model/monitor.stop
wait "$MONITOR_PID"
trap - EXIT
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 97 "$PY" source/script/check_api_sanitizer.py --source source --library "$LIB" > R10_api_memcheck.log 2>&1
