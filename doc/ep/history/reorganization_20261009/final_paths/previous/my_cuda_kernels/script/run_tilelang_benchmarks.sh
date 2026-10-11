#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; use the existing TileLang Python environment.
: "${TILELANG_PYTHON:?Set TILELANG_PYTHON to your TileLang environment Python}"
: "${CXX:?Set CXX to GCC 12 (SCVideo compiler on the 4090 server)}"
BENCH_GPU="${BENCH_GPU:-1}"
OUT_DIR="${1:-result/20261005/tilelang}"
mkdir -p "$OUT_DIR"
: > "$OUT_DIR/exitcodes.txt"
for op in rmsnorm_forward rmsnorm_backward fused_residual_rmsnorm_forward fused_residual_rmsnorm_backward; do
  CUDA_VISIBLE_DEVICES="$BENCH_GPU" "$TILELANG_PYTHON" -u "dev/tilelang/$op.py" > "$OUT_DIR/$op.log" 2>&1
  printf '%s 0\n' "$op" >> "$OUT_DIR/exitcodes.txt"
done

# Server setup used for this report (run before this script):
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate SCVideo
# export TILELANG_GCC_ROOT="$CONDA_PREFIX"
# conda deactivate
# cd /data/user4/tongleiwen/tilelang-3032
# source .venv/bin/activate
# export CC="$TILELANG_GCC_ROOT/bin/x86_64-conda-linux-gnu-cc"
# export CXX="$TILELANG_GCC_ROOT/bin/x86_64-conda-linux-gnu-c++"
# export TILELANG_PYTHON="$VIRTUAL_ENV/bin/python"
# "$CXX" --version
# python -c 'from tilelang.contrib.cc import get_cplus_compiler; print(get_cplus_compiler())'
# cd /path/to/my_cuda_kernels
