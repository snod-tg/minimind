"""Prepare a separate repository snapshot and real-text short training run."""
from pathlib import Path
import shutil

repo = Path('/data/user4/tongleiwen/minimind')
stage8 = Path('/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage08_ep_20261009')
stage9 = Path('/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage09_compile_20261009')
smoke = stage9/'minimind'
for directory in ['model', 'dataset']:
    target = smoke/directory
    target.mkdir(parents=True, exist_ok=True)
    for path in (repo/directory).iterdir():
        if path.is_file() and path.suffix in ['.py', '.json']:
            shutil.copy2(path, target/path.name)
for filename in ['model_minimind.py', 'ep_routing_backend.py']:
    shutil.copy2(stage8/'minimind/model'/filename, smoke/'model'/filename)
with (repo/'dataset/pretrain_t2t_mini.jsonl').open() as source, (smoke/'dataset/ep_compile_smoke.jsonl').open('w') as output:
    for index, line in enumerate(source):
        if index == 128: break
        output.write(line)
print('SMOKE PREPARED', smoke, flush=True)
