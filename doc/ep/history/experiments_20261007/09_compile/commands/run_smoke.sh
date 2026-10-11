#!/usr/bin/env bash
set -euo pipefail
RUN=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage09_compile_20261009
EP=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage08_ep_20261009
python=/data/user4/tongleiwen/tilelang-3032/.venv/bin/python
test ! -e "$RUN/minimind/out/ep_compile_smoke_768_moe.pth"
"$python" "$RUN/prepare_smoke.py"
cd "$RUN/minimind/trainer"
export CUDA_VISIBLE_DEVICES=${EP_TRAIN_GPUS:-1,3} OMP_NUM_THREADS=1
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
export TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
exec "$python" -m torch.distributed.run --standalone --nproc_per_node=2 train_pretrain.py \
  --use_ep --num_experts 6 --num_experts_per_tok 2 --ep_comm_dtype native \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 340 \
  --batch_size 32 --epochs 1 --learning_rate 5e-4 --dtype bfloat16 \
  --num_workers 0 --accumulation_steps 1 --grad_clip 1.0 \
  --data_path ../dataset/ep_compile_smoke.jsonl --from_weight none --from_resume 0 \
  --save_dir ../out --save_weight ep_compile_smoke --log_interval 1 --save_interval 2 --use_compile 1 \
  --ep_routing compact --ep_routing_library "$EP/results/R27_final_build/my_cuda_ep.so" \
  --ep_routing_python_dir "$EP/source/dev/pytorch"
