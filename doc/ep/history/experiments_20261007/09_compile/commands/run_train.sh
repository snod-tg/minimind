#!/usr/bin/env bash
set -euo pipefail
REPO=/data/user4/tongleiwen/minimind
EP_STAGE=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage08_ep_20261009
RUN=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage09_compile_20261009
SAVE_NAME=pretrain_ep6_top2_compile_20261009
SAVE_DIR="$REPO/out/ep6_top2_compile_20261009"
# Refuse an accidental second launch over an existing trained checkpoint.
test ! -e "$SAVE_DIR/${SAVE_NAME}_768_moe.pth"
test ! -e "$REPO/checkpoints/${SAVE_NAME}_768_moe_ep_rank0_resume.pth"
if compgen -G "$REPO/checkpoints/${SAVE_NAME}_768_moe*" >/dev/null; then
  echo "Refuse to replace existing checkpoints: $SAVE_NAME" >&2
  exit 1
fi
mkdir -p "$SAVE_DIR"
cd "$REPO/trainer"
export CUDA_VISIBLE_DEVICES=${EP_TRAIN_GPUS:-2,3} OMP_NUM_THREADS=1
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
export TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /data/user4/tongleiwen/tilelang-3032/.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 train_pretrain.py \
  --use_ep --num_experts 6 --num_experts_per_tok 2 --ep_comm_dtype native \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 340 \
  --batch_size 32 --epochs 2 --learning_rate 5e-4 --dtype bfloat16 \
  --num_workers 8 --accumulation_steps 8 --grad_clip 1.0 \
  --data_path "$REPO/dataset/pretrain_t2t_mini.jsonl" \
  --from_weight none --from_resume 0 --save_weight "$SAVE_NAME" --save_dir "$SAVE_DIR" \
  --log_interval 100 --save_interval 1000 --use_compile 1 \
  --ep_routing compact --ep_routing_library "$EP_STAGE/results/R27_final_build/my_cuda_ep.so" \
  --ep_routing_python_dir "$EP_STAGE/source/dev/pytorch"
