#!/usr/bin/env bash
set -euo pipefail
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="${RUN_NAME:-R02_baseline}"
OUT="$TASK_ROOT/results/$RUN_NAME"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export BENCH_OUTPUT="$OUT"
python - <<'PY'
import json, os, platform, subprocess, sys, shutil
from pathlib import Path
import numpy, torch, transformers
out = Path(os.environ['BENCH_OUTPUT'])
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.limit', '--format=csv,noheader'], text=True)
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader'], text=True)
topology = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
for line in gpu.splitlines():
    cells = [v.strip() for v in line.split(',')]
    if cells[0] in ('1', '2'):
        if int(cells[4].split()[0]) > 100 or cells[1] in apps:
            raise RuntimeError('Selected GPU is occupied: ' + line)
info = dict(python=sys.version, executable=sys.executable, virtual_env=os.environ.get('VIRTUAL_ENV'),
            conda_prefix=os.environ.get('CONDA_PREFIX'),
            torch=torch.__version__, cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            nccl=torch.cuda.nccl.version(), transformers=transformers.__version__, numpy=numpy.__version__,
            platform=platform.platform(), visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            gpu_query=gpu, compute_apps_before=apps, topology=topology,
            cpp=subprocess.check_output([shutil.which('g++'), '--version'], text=True),
            nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True))
(out/'environment.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
print('Environment recorded; selected GPUs are idle.', flush=True)
PY
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  "$TASK_ROOT/measure_baseline.py" \
  --repo /data/user4/tongleiwen/minimind \
  --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth \
  --source "$TASK_ROOT/model_minimind_snapshot.py" \
  --output "$OUT" \
  --workloads "${BENCH_WORKLOADS:-prefill_small,prefill_medium,prefill_large,decode_single,decode_batch}" \
  --topks "${BENCH_TOPKS:-1,2}" \
  --warmups "${BENCH_WARMUPS:-5}" --steps "${BENCH_STEPS:-30}" \
  --profile-steps "${BENCH_PROFILE_STEPS:-3}" \
  2>&1 | tee "$OUT/console.log"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu_after.csv"
