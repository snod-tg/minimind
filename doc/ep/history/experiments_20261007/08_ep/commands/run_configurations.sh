#!/usr/bin/env bash
set -euo pipefail
python=/data/user4/tongleiwen/tilelang-3032/.venv/bin/python
library=${1:-results/R24_configurable_build/my_cuda_ep.so}
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=4 "$python" -m torch.distributed.run --standalone --nproc_per_node=1 validate_distributed_configuration.py --library "$library" --output results/R26_one_rank_config --experts 4 --topks 1 2 3 4 > R26_one_rank_config.log 2>&1
CUDA_VISIBLE_DEVICES=4,5,7 "$python" -m torch.distributed.run --standalone --nproc_per_node=3 validate_distributed_configuration.py --library "$library" --output results/R26_three_rank_config --experts 6 --topks 1 2 3 > R26_three_rank_config.log 2>&1
CUDA_VISIBLE_DEVICES=4,5,7,2 "$python" -m torch.distributed.run --standalone --nproc_per_node=4 validate_distributed_configuration.py --library "$library" --output results/R26_four_rank_config --experts 8 --topks 1 2 4 > R26_four_rank_config.log 2>&1
echo 'ALL RANK CONFIGURATIONS PASSED'
