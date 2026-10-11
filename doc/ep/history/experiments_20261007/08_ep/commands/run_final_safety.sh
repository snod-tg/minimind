#!/usr/bin/env bash
set -euo pipefail
python=/data/user4/tongleiwen/tilelang-3032/.venv/bin/python
export CUDA_VISIBLE_DEVICES=${EP_SAFETY_GPU:-1} OMP_NUM_THREADS=1
for tool in memcheck racecheck synccheck; do
  /usr/local/cuda/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 "$python" validate_configuration.py --quick --library results/R27_final_build/my_cuda_ep.so --output "results/R30_${tool}_configuration.json" > "R30_${tool}_configuration.log" 2>&1
done
echo 'FINAL CONFIGURABLE SAFETY PASSED'
