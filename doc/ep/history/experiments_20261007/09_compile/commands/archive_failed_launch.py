"""Preserve a startup OOM before retrying a run with no completed updates."""
from pathlib import Path
import shutil
import subprocess

run=Path(__file__).resolve().parent
repo=Path('/data/user4/tongleiwen/minimind')
name='pretrain_ep6_top2_compile_20261009'
assert (run/'training_exit_code.txt').read_text().strip()=='1'
log=(run/'pretrain_ep6_top2_compile.log').read_text()
assert 'torch.OutOfMemoryError' in log and 'Epoch:[' not in log
assert not list((repo/'checkpoints').glob(name+'*'))
assert not (repo/'out/ep6_top2_compile_20261009'/f'{name}_768_moe.pth').exists()
target=run/'results/R08_attempt1_oom'
target.mkdir(parents=True,exist_ok=False)
for filename in ['pretrain_ep6_top2_compile.log','training_exit_code.txt','training_finished_utc.txt']:
    shutil.move(run/filename,target/filename)
shutil.copy2(repo/'out/ep6_top2_compile_20261009/training_config.json',target/'training_config.json')
session='ep6_top2_compile_20261009'
dead=subprocess.check_output(['tmux','display-message','-p','-t',session,'#{pane_dead}'],text=True).strip()
assert dead=='1'
subprocess.run(['tmux','rename-session','-t',session,session+'_attempt1_oom'],check=True)
print('FAILED STARTUP ARCHIVED; NO COMPLETED TRAINING UPDATES')
