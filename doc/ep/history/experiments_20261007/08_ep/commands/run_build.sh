#!/usr/bin/env bash
set -euo pipefail
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage08_ep_20261009
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
/data/user4/tongleiwen/tilelang-3032/.venv/bin/python build_api.py --source source --output results/R01_build
