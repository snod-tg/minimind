"""Run round-robin end-to-end timings with selected-GPU occupancy evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .train import ROOT, validate_gpus
import time

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--gpus', required=True)
p.add_argument('--experts', type=int, default=4)
p.add_argument('--checkpoint', type=Path)
p.add_argument('--rounds', type=int, default=21)
a = p.parse_args()
validate_gpus(a.gpus)
a.output=a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
gpu_ids = set(a.gpus.split(','))
inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
selected = {uuid.strip() for line in inventory.splitlines() for index, uuid in [line.split(',')] if index.strip() in gpu_ids}
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, OMP_NUM_THREADS='1')
command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
    '--module', 'scripts.ep.benchmark', '--library', str(Path(a.library).resolve()), '--output', str(a.output), '--rounds', str(a.rounds), '--experts', str(a.experts)]
if a.checkpoint is not None: command += ['--checkpoint', str(a.checkpoint.resolve())]
observations = []
with (a.output/'console.log').open('w') as log:
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while True:
        query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True)
        active = []
        for line in query.splitlines():
            uuid, pid, memory = map(str.strip, line.split(','))
            if uuid not in selected: continue
            try:
                cmdline = Path('/proc')/pid/'cmdline'
                cmd = cmdline.read_bytes().replace(b'\0', b' ').decode()
            except OSError:
                cmd = ''
            active.append(dict(uuid=uuid, pid=int(pid), memory=memory, own='scripts.ep.benchmark' in cmd))
        observations.append(dict(time=time.time(), active=active))
        if process.poll() is not None: break
        time.sleep(1)
foreign = [r for row in observations for r in row['active'] if not r['own']]
(a.output/'gpu_monitor.json').write_text(json.dumps(dict(gpus=a.gpus, samples=len(observations), foreign=foreign, observations=observations, exit_code=process.returncode), indent=2)+'\n')
assert process.returncode == 0, process.returncode
assert not foreign, foreign
print('MEASURED VARIANTS PASSED', len(observations), 'occupancy samples', flush=True)
