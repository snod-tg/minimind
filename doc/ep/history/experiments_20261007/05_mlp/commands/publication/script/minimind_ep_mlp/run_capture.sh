#!/usr/bin/env bash
set -euo pipefail
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage05
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
K9_PY=/data/user4/tongleiwen/tilelang-3032/.venv/bin/python
mkdir -p results
"$K9_PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 capture_original.py \
 --source source --baseline /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/results/R04_baseline \
 --repo /data/user4/tongleiwen/minimind --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth \
 --output results/R01_extra_capture > results/R01_extra_capture.console.log 2>&1
