"""Read-only occupancy samples; other jobs are never stopped or changed."""
import argparse,csv,io,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stop',type=Path,required=True);a=p.parse_args()
selected=set(a.gpus.split(','));a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w') as out:
 while not a.stop.exists():
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  rows=[r for r in csv.reader(io.StringIO(gpu),skipinitialspace=True) if r[0] in selected];uuids={r[1] for r in rows}
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
  active=[r for r in csv.reader(io.StringIO(apps),skipinitialspace=True) if r and r[0] in uuids]
  out.write(json.dumps(dict(unix_seconds=time.time(),gpus=rows,apps=active))+'\n');out.flush();time.sleep(1)
