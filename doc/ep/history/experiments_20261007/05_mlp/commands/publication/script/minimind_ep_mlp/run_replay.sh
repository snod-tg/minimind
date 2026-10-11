#!/usr/bin/env bash
set -euo pipefail
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage05
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
K9_RUN="${K9_RUN:-R02_mlp_validation}"
/data/user4/tongleiwen/tilelang-3032/.venv/bin/python replay_mlp.py \
 --source source --library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/R05_dispatch_build/my_cuda_kernel9.so \
 --baseline /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/results/R04_baseline \
 --extra results/R01_extra_capture --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth \
 --output "results/$K9_RUN" "$@" > "results/$K9_RUN.console.log" 2>&1
