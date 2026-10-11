#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KERNEL8_RESULT="${KERNEL8_RESULT:-$ROOT/result/20261006/moe_grouped_kernel8}"
export BENCH_GPU="${BENCH_GPU:-3}"
export CUDA_VISIBLE_DEVICES="$BENCH_GPU"
NVCC_BIN="${NVCC_BIN:-/usr/local/cuda/bin/nvcc}"
PYTHON_BIN="${PYTHON_BIN:-/data/user4/tongleiwen/tilelang-3032/.venv/bin/python}"
ARCH="${ARCH:-sm_89}"
cd "$ROOT"
mkdir -p "$KERNEL8_RESULT"
"$NVCC_BIN" -O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr -Xptxas=-v dev/cuda/moe_grouped_gemm_forward.cu -lcublas -o "$KERNEL8_RESULT/grouped" >"$KERNEL8_RESULT/build.log" 2>&1
"$NVCC_BIN" -O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr --cudart shared -Xcompiler=-fPIC -shared script/kernel8_bridge.cu -lcublas -o "$KERNEL8_RESULT/kernel8_bridge.so" >"$KERNEL8_RESULT/bridge_build.log" 2>&1
"$KERNEL8_RESULT/grouped" --self-test --kernel 8 >"$KERNEL8_RESULT/selftest8.log" 2>&1
"$PYTHON_BIN" script/benchmark_kernel8.py >"$KERNEL8_RESULT/benchmark.log" 2>&1
bash script/sanitize_kernel8.sh
