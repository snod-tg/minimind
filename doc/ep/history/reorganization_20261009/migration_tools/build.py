"""Build native operators from the canonical CUDA sources; no source snapshots."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "ep": ("ep_routing_torch.cpp", "ep_routing.cu", "my_cuda_ep.so"),
    "grouped_gemm": ("moe_grouped_gemm_split_torch.cpp", "moe_grouped_gemm_split_torch.cu", "my_cuda_grouped_split.so"),
    "kernel9": ("moe_grouped_gemm_torch.cpp", "moe_grouped_gemm_torch.cu", "my_cuda_kernel9.so"),
}

def build_plan(kind, arch="sm_89", build_root=None):
    import torch
    from torch.utils.cpp_extension import include_paths
    source = ROOT / "dev/cuda"
    binding, kernel, library = SPECS[kind]
    for name in (binding, kernel):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.iterdir()) if p.suffix in (".cu", ".cpp", ".h", ".cuh")}
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    cxx = os.environ.get("CXX", "g++")
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    metadata = dict(kind=kind, arch=arch, source_sha256=hashes, torch=torch.__version__,
                    torch_cuda=torch.version.cuda, abi=abi, python=sys.version,
                    python_executable=sys.executable, cxx=cxx, cuda_home=str(cuda))
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = Path(build_root or os.environ.get("MY_CUDA_KERNELS_BUILD", ROOT / "build")).expanduser().resolve()
    out = base / kind / f"{stamp}_{digest}"
    torchlib = Path(torch.__file__).parent / "lib"
    flags = ["-O3", "-std=c++17", "-fPIC", f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]
    commands = [
        [cxx, *flags, *[f"-I{p}" for p in include_paths()], f"-I{cuda}/include",
         "-c", str(source / binding), "-o", str(out / "binding.o")],
        [str(cuda / "bin/nvcc"), "-ccbin", cxx, "-O3", "-std=c++17", f"-arch={arch}",
         *(["--fmad=false"] if kind == "ep" else []), "--cudart", "shared", "-Xcompiler", "-fPIC",
         "-c", str(source / kernel), "-o", str(out / "kernel.o")],
        [cxx, "-shared", str(out / "binding.o"), str(out / "kernel.o"), f"-L{torchlib}",
         f"-L{cuda}/lib64", f"-Wl,-rpath,{torchlib}", f"-Wl,-rpath,{cuda}/lib64",
         "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda", "-lcudart", "-lcublas",
         "-o", str(out / library)],
    ]
    return out, library, commands, metadata

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=SPECS)
    p.add_argument("--arch", default="sm_89")
    p.add_argument("--build-root", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out, library, commands, record = build_plan(args.kind, args.arch, args.build_root)
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if record["torch_cuda"] is None:
        raise RuntimeError("Use a CUDA-enabled PyTorch environment to compile these operators")
    current = out.parent / "current"
    if current.exists() and not current.is_symlink():
        raise RuntimeError(f"Refusing to replace a real directory: {current}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    for i, command in enumerate(commands):
        with (out / f"compile_{i}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, "-c", "import torch,sys; torch.ops.load_library(sys.argv[1])", str(out / library)], check=True)
    record["library_sha256"] = hashlib.sha256((out / library).read_bytes()).hexdigest()
    record["status"] = "built_and_loaded"
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    pending = out.parent / f".current_{out.name}"
    pending.symlink_to(out.name, target_is_directory=True)
    os.replace(pending, current)
    print(f"Built {out / library}")

if __name__ == "__main__":
    main()
