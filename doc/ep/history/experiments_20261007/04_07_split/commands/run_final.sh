#!/usr/bin/env bash
set -euo pipefail
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04_07_split_20261009
PY=/data/user4/tongleiwen/tilelang-3032/.venv/bin/python
LIB=results/R01_build/my_cuda_grouped_split.so
BASE=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007
export CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 BENCH_CPU_BASE=28 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 verify_cli.py --repo /data/user4/tongleiwen/minimind --library "$LIB" --kn-checkpoint "$BASE/stage04_weights/results/R01_convert/pretrain_ep_768_moe_kn.pth" --output results/R13b_cli > R13b_cli.log 2>&1
CUDA_VISIBLE_DEVICES=4 "$PY" source/script/replay_split_mlp.py --library "$LIB" --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth --baseline "$BASE/results/R04_baseline" --extra "$BASE/stage05/results/R01_extra_capture" --output results/R14_final_mlp > R14_final_mlp.log 2>&1
mkdir results/R15_clean_model
"$PY" monitor_gpu.py --gpus 4,5 --output results/R15_clean_model/gpu_monitor.jsonl --stop results/R15_clean_model/monitor.stop &
MONITOR_PID=$!
trap 'touch results/R15_clean_model/monitor.stop; wait "$MONITOR_PID"' EXIT
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 benchmark_model.py --stage6 . --library "$LIB" --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth --repo /data/user4/tongleiwen/minimind --baseline "$BASE/results/R04_baseline" --output results/R15_clean_model > R15_clean_model.log 2>&1
touch results/R15_clean_model/monitor.stop
wait "$MONITOR_PID"
trap - EXIT
