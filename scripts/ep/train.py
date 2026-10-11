"""Launch independent EP pretraining runs from the MiniMind repository."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/ep/pretrain_6_top2.json"

def validate_gpus(value):
    ids = [int(i) for i in value.split(",")]
    if len(ids) != 2 or len(set(ids)) != 2 or any(i < 0 or i > 7 for i in ids):
        raise ValueError("Choose two distinct physical GPUs from 0..7")
    if ids[0] // 4 != ids[1] // 4:
        raise ValueError("Choose both GPUs from 0..3 or both from 4..7")
    return ids

def load_config(path):
    config = json.loads(Path(path).read_text())
    fixed = {"use_ep": True, "ep_comm_dtype": "native",
             "use_compile": 1, "from_weight": "none", "from_resume": 0}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("This entry requires from-scratch compile training, native experts and optimized EP")
    if config.get("ep_routing") not in ("compact", "dedup_train"):
        raise ValueError("Use compact or dedup_train routing")
    if config["ep_routing"] == "dedup_train" and config["num_experts_per_tok"] != 2:
        raise ValueError("dedup_train requires Top-2")
    e, k = config["num_experts"], config["num_experts_per_tok"]
    if not isinstance(e, int) or not isinstance(k, int) or e < 2 or e % 2 or not 1 <= k <= e:
        raise ValueError("Expert count must be divisible by two; Top-K must be in 1..E")
    return config

def command_for(config, run, run_id, library):
    values = dict(config, save_weight=f"pretrain_{run_id}", save_dir=str(run / "checkpoints/full"),
                  checkpoint_dir=str(run / "checkpoints/resume"))
    values["data_path"] = str((ROOT / values["data_path"]).resolve())
    if library:
        values["ep_routing_library"] = str(Path(library).expanduser().resolve())
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "train_pretrain.py"]
    for name, value in values.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{name}")
        else:
            command += [f"--{name}", str(value)]
    return command

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--run-id", required=True)
    p.add_argument("--gpus", required=True)
    p.add_argument("--library", help="Optional override; defaults to the installed operator package build")
    p.add_argument("--tmux", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    validate_gpus(args.gpus)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        raise ValueError("Use a short alphanumeric run ID with '_' or '-'")
    config = load_config(args.config)
    run = ROOT / "runs" / args.run_id
    command = command_for(config, run, args.run_id, args.library)
    print(f"Working directory: {ROOT / 'trainer'}")
    print(f"Physical GPUs: {args.gpus}")
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if run.exists():
        raise FileExistsError(f"Choose a new run ID; existing run is preserved: {run}")
    from my_cuda_kernels.ep_ops import load_library
    from my_cuda_kernels._native import resolve_library
    import torch
    library = resolve_library("ep", args.library)
    load_library(library)
    if torch.version.cuda is None:
        raise RuntimeError("CUDA-enabled PyTorch is required")
    if not (ROOT / config["data_path"]).is_file():
        raise FileNotFoundError(ROOT / config["data_path"])
    if not (ROOT / "model/tokenizer.json").is_file():
        raise FileNotFoundError(ROOT / "model/tokenizer.json")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv"], text=True)
    print(inventory, flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               TORCHINDUCTOR_COMPILE_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.tmux:
        worker = [sys.executable, "-m", "scripts.ep.train", "--config", str(args.config.resolve()),
                  "--run-id", args.run_id, "--gpus", args.gpus, "--library", str(library)]
        # Explicitly pass the selected interpreter/toolchain rather than relying on tmux's old environment.
        forwarded = {key: value for key, value in env.items() if key in
                     ("PATH", "CC", "CXX", "CUDA_HOME", "MY_CUDA_KERNELS_BUILD", "LD_LIBRARY_PATH")}
        shell = shlex.join(["env", *[f"{k}={v}" for k, v in forwarded.items()], *worker])
        subprocess.run(["tmux", "new-session", "-d", "-s", args.run_id, "-c", str(ROOT), shell], check=True, env=env)
        print(f"Started tmux session: {args.run_id}")
        return
    for name in ("config", "logs", "checkpoints/full", "checkpoints/resume", "state"):
        (run / name).mkdir(parents=True, exist_ok=(name != "config"))
    command = command_for(config, run, args.run_id, library)
    (run / "config/settings.json").write_text(json.dumps(config, indent=2) + "\n")
    (run / "config/launch.json").write_text(json.dumps(dict(command=command, cwd=str(ROOT / "trainer"),
        gpus=args.gpus, library=str(library), inventory=inventory, torch=torch.__version__,
        started_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")
    with (run / "logs/pretrain.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT / "trainer", env=env, stdout=log, stderr=subprocess.STDOUT)
    (run / "state/training_exit_code.txt").write_text(f"{completed.returncode}\n")
    (run / "state/training_finished_utc.txt").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    raise SystemExit(completed.returncode)

if __name__ == "__main__":
    main()
