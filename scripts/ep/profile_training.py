"""Profile one optimized EP training window; timings are not a benchmark."""
import argparse
from pathlib import Path
import subprocess
from .benchmark_training import ROOT,prepare,run_trial,processes
from .train import validate_gpus


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',default='5,7')
    parser.add_argument('--library')
    parser.add_argument('--data',type=Path,default=ROOT/'dataset/pretrain_t2t_mini.jsonl')
    a=parser.parse_args();a.output=a.output.resolve();a.data=a.data.resolve()
    ids=validate_gpus(a.gpus)
    selected={}
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    for line in query.splitlines():
        index,uuid,memory,util=map(str.strip,line.split(','))
        if int(index) in ids:
            assert int(memory)<100 and int(util)==0,('GPU occupied',line)
            selected[uuid]=int(index)
    assert len(selected)==2 and not processes(selected)
    a.warmup_updates=2;a.updates=4;a.repeats=3;a.profile=True
    prepare(a)
    run_trial(a,'performance','ep_compact',0,selected)
    print('Profile completed; use unprofiled runs for performance acceptance.',flush=True)


if __name__=='__main__':main()
