"""Apply the tested EP hook/trainer without replacing prior user edits."""
from pathlib import Path
import hashlib
import shutil

repo=Path('/data/user4/tongleiwen/minimind')
stage9=Path(__file__).resolve().parent
stage8=stage9.parent/'stage08_ep_20261009'
guards={'model/model_minimind.py':'c7a344373dff56099c15cbb45a1880a6990c512adc0f45366a277884451b5a2c',
        'trainer/train_pretrain.py':'344473d719dca71524fbf6c81adea7bbcbf7500b3035391c4089b1d223dcb395'}
for relative,expected in guards.items():
    assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==expected,relative
assert not (repo/'model/ep_routing_backend.py').exists()
for relative in guards:
    backup=stage9/'before_ep_routing'/relative
    backup.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(repo/relative,backup)
shutil.copy2(stage8/'minimind/model/model_minimind.py',repo/'model/model_minimind.py')
shutil.copy2(stage8/'minimind/model/ep_routing_backend.py',repo/'model/ep_routing_backend.py')
shutil.copy2(stage9/'minimind/trainer/train_pretrain.py',repo/'trainer/train_pretrain.py')
print('SERVER TRAINING REPOSITORY UPDATED')
