"""Static-count CUTLASS reference: tune two GEMM shapes, retain BF16 boundaries."""
import ctypes
import hashlib
import json
import statistics
from pathlib import Path

import torch
from kernel9_reference import Cutlass,cutlass_libraries,reference,validate,graph_make,graph_time


class CutlassExpertMLP:
    def __init__(self,x,counts,cache,root,output,case):
        self.families=cutlass_libraries(root)
        self.runners=[]
        self.gate=x.new_empty((len(x),2432));self.up=torch.empty_like(self.gate)
        self.activated=torch.empty_like(self.gate);self.hidden=torch.empty_like(self.gate)
        self.out=torch.empty_like(x)
        self.weights=cache.packed;self.counts=counts
        self.records=[];self.unsupported=[]
        self.root=Path(root);self.output=output;self.case=case
        gate=self.select('gate_up',x,self.weights['gate_proj'],self.gate)
        self.runners.append(self.make(gate,x,self.weights['gate_proj'],self.gate))
        self.runners.append(self.make(gate,x,self.weights['up_proj'],self.up))
        self.runners[0].run();self.runners[1].run()
        torch.ops.aten.silu.out(self.gate,out=self.activated)
        torch.mul(self.activated,self.up,out=self.hidden)
        down=self.select('down',self.hidden,self.weights['down_proj'],self.out)
        self.runners.append(self.make(down,self.hidden,self.weights['down_proj'],self.out))
        self.selected=dict(gate_up=gate,down=down)
        self.run()
        with torch.autocast('cuda',enabled=False):
            for operand,weight,out in [(x,self.weights['gate_proj'],self.gate),(x,self.weights['up_proj'],self.up),
                                       (self.hidden,self.weights['down_proj'],self.out)]:
                validate(out,reference(operand,weight,counts))

    def make(self,row,x,b,y):
        return Cutlass(self.families[row['kind']][0],row['config']['id'],row['blocks'],self.counts,x,b,y)

    def select(self,projection,x,b,y):
        with torch.autocast('cuda',enabled=False):ref=reference(x,b,self.counts)
        runtime=ctypes.CDLL('libcudart.so.12');runtime.cudaGetLastError.restype=ctypes.c_int
        candidates=[];sm=torch.cuda.get_device_properties(x.device).multi_processor_count
        for kind,(lib,configs) in self.families.items():
            for config in configs:
                for blocks in [0,sm,2*sm]:
                    runner=None;graph=None
                    try:
                        runner=Cutlass(lib,config['id'],blocks,self.counts,x,b,y)
                        y.fill_(float('nan'));runner.run();torch.cuda.synchronize();validate(y,ref)
                        graph=graph_make(runner.run,4)
                        us=statistics.median(graph_time(graph,4,2) for _ in range(2))
                        row=dict(projection=projection,kind=kind,config=config,blocks=blocks,us=us)
                        candidates.append(row);self.records.append(row)
                    except RuntimeError as error:
                        # Resource-unsupported candidates only. Numerical assertions remain fatal.
                        code=runtime.cudaGetLastError()
                        self.unsupported.append(dict(projection=projection,kind=kind,config=config,blocks=blocks,
                                                     error=str(error),cuda_error_cleared=code))
                    finally:
                        graph=None
                        if runner is not None:runner.close()
        assert candidates
        refined=[]
        for row in sorted(candidates,key=lambda r:r['us'])[:3]:
            runner=self.make(row,x,b,y)
            try:
                graph=graph_make(runner.run,16)
                refined.append(dict(**row,refined_us=statistics.median(graph_time(graph,16,3) for _ in range(3))))
                del graph
            finally:runner.close()
        best=min(refined,key=lambda r:r['refined_us'])
        (self.output/f'{self.case}_cutlass_selection.json').write_text(json.dumps(dict(
            selected=best,latest_projection=projection,candidates=self.records,unsupported=self.unsupported,
            binaries={name:hashlib.sha256((self.root/name).read_bytes()).hexdigest()
                      for name in ['bridge.so','bridge_host.so','bridge_extra.so']}),indent=2)+'\n')
        print('CUTLASS_SELECTED',self.case,projection,json.dumps(best),flush=True)
        return best

    def run(self):
        self.runners[0].run()
        torch.ops.aten.silu.out(self.gate,out=self.activated)
        self.runners[1].run()
        torch.mul(self.activated,self.up,out=self.hidden)
        self.runners[2].run()
        return self.out

    def close(self):
        for runner in self.runners:runner.close()
