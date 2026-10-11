"""Read-only sampling of the selected GPU, with competing PID detection."""
import json, os, subprocess, threading, time
class Monitor:
 def __init__(self,path):self.path=path;self.stop_event=threading.Event();self.thread=None
 def start(self):
  self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def run(self):
  gpu=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
  with self.path.open('w') as out:
   while not self.stop_event.is_set():
    def call(args):return subprocess.check_output(['nvidia-smi']+args,text=True).strip()
    try:
     state=call(['-i',gpu,'--query-gpu=uuid,memory.used,utilization.gpu,clocks.sm,pstate,power.draw','--format=csv,noheader,nounits'])
     uuid=state.split(',')[0];apps=call(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
     peers=[]
     for line in apps.splitlines():
      fields=[x.strip() for x in line.split(',')]
      if fields[0]==uuid and int(fields[1])!=os.getpid():peers.append(fields)
     row=dict(time=time.time(),gpu=gpu,state=state,external_processes=peers)
    except Exception as ex:row=dict(time=time.time(),error=str(ex))
    out.write(json.dumps(row)+'\n');out.flush();self.stop_event.wait(1)
 def close(self):
  self.stop_event.set()
  if self.thread:self.thread.join(timeout=5)
