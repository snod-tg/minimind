"""Kernel9 validation and same-input tuning against SM80 CUTLASS on SM89."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F


CONFIGS = [dict(id=i, bm=bm, bn=128, bk=32, stages=st)
           for i, (bm, st) in enumerate([(32,3),(16,3),(64,3),(32,2),(16,2),(64,2)])]
CONFIGS += [dict(id=i,bm=16,bn=128,bk=32,stages=3,split_k=parts) for i,parts in [(6,4),(7,8)]]


def bind_ours(path):
    lib = ctypes.CDLL(str(path))
    lib.k9_create.argtypes = [ctypes.c_int]*5 + [ctypes.c_void_p]*4 + [ctypes.c_int]*2
    lib.k9_create.restype = ctypes.c_void_p
    lib.k9_run.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
    lib.k9_run.restype = ctypes.c_int
    lib.k9_destroy.argtypes = [ctypes.c_void_p]
    lib.k9_error.restype = ctypes.c_char_p
    if hasattr(lib,'k9_default_config'):
        lib.k9_default_config.argtypes=[ctypes.c_int]*3
        lib.k9_default_config.restype=ctypes.c_int
    return lib


class Ours:
    def __init__(self, lib, counts, a, b, c, config=0, blocks=0):
        self.lib = lib
        self.keep = (counts,a,b,c)
        self.ptr = lib.k9_create(int(a.dtype==torch.bfloat16),len(counts),len(a),c.shape[1],a.shape[1],
                                 counts.data_ptr(),a.data_ptr(),b.data_ptr(),c.data_ptr(),config,blocks)
        if not self.ptr: raise RuntimeError(lib.k9_error().decode())

    def run(self, mode=0):
        if self.lib.k9_run(self.ptr,torch.cuda.current_stream().cuda_stream,mode):
            raise RuntimeError(self.lib.k9_error().decode())

    def close(self):
        if self.ptr: self.lib.k9_destroy(self.ptr); self.ptr=None


class Cutlass:
    def __init__(self, lib, config, blocks, counts, a, b, c):
        self.lib = lib
        ct = (ctypes.c_int*len(counts))(*counts)
        self.ptr = lib.gmm_create(int(a.dtype==torch.bfloat16),config,blocks,ct,len(counts),c.shape[1],a.shape[1],
                                  a.data_ptr(),b.data_ptr(),c.data_ptr())
        if not self.ptr: raise RuntimeError(lib.gmm_error().decode())

    def run(self):
        if self.lib.gmm_run(self.ptr,torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(self.lib.gmm_error().decode())

    def close(self):
        if self.ptr: self.lib.gmm_destroy(self.ptr); self.ptr=None


def cutlass_libraries(root):
    result = {}
    for kind, filename, configfile in [('device','bridge.so','cutlass_configs.json'),
                                       ('host','bridge_host.so','cutlass_configs.json'),
                                       ('extra','bridge_extra.so','cutlass_extra_configs.json')]:
        lib = ctypes.CDLL(str(root/filename))
        lib.gmm_create.argtypes = [ctypes.c_int]*3+[ctypes.POINTER(ctypes.c_int)]+[ctypes.c_int]*3+[ctypes.c_void_p]*3
        lib.gmm_create.restype = ctypes.c_void_p
        lib.gmm_run.argtypes = [ctypes.c_void_p]*2
        lib.gmm_run.restype = ctypes.c_int
        lib.gmm_destroy.argtypes = [ctypes.c_void_p]
        lib.gmm_error.restype = ctypes.c_char_p
        result[kind] = (lib,json.loads((root/configfile).read_text()))
    return result


def reference(a,b,counts):
    pieces,offset = [],0
    for e,m in enumerate(counts):
        if m: pieces.append((a[offset:offset+m].float()@b[e].float()).to(a.dtype))
        offset += m
    return torch.cat(pieces) if pieces else a.new_empty((0,b.shape[2]))


def validate(c,ref):
    if not c.numel(): return dict(max_abs=0.,relative_rms=0.)
    delta=(c.float()-ref.float()).abs()
    rtol=0.008 if c.dtype==torch.bfloat16 else 0.001
    bad=(~torch.isfinite(c)) | (delta>2e-4+rtol*ref.float().abs())
    if bool(bad.any()): raise AssertionError(f'{int(bad.sum())} mismatches, max abs {float(delta.max())}')
    rms=float(delta.square().mean().sqrt())
    norm=float(ref.float().square().mean().sqrt())
    return dict(max_abs=float(delta.max()),relative_rms=rms/norm if norm else rms)


def cpu_double_check(a,b,c,counts,samples=17):
    # Copy only sampled input rows/weight columns; do not copy large full tensors.
    offset=0; checked=0
    for e,m in enumerate(counts):
        if m:
            total=m*c.shape[1];number=min(samples,total)
            idx=torch.arange(number,device=a.device,dtype=torch.int64)*(total-1)//max(1,number-1)
            rows,cols=idx//c.shape[1],idx%c.shape[1]
            assert bool((rows<m).all()) and bool((cols<c.shape[1]).all())
            expected=(a[offset+rows].cpu().double()*b[e,:,cols].T.cpu().double()).sum(1).to(c.dtype).to(c.device)
            validate(c[offset+rows,cols],expected)
            checked+=len(idx)
        offset+=m
    return checked


def graph_make(fn,nodes):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):
        for _ in range(nodes): fn()
    return graph


def graph_time(graph,nodes,replays=3):
    for _ in range(2): graph.replay()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays): graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/(nodes*replays)


def tensor_case(counts,k,n,dtype,seed=20261007):
    torch.manual_seed(seed)
    a=(torch.rand((sum(counts),k),device='cuda')-.5).to(dtype)
    b=(torch.rand((len(counts),k,n),device='cuda')-.5).to(dtype)
    c=torch.empty((sum(counts),n),device='cuda',dtype=dtype)
    return a,b,c


def check_all(lib,counts,a,b,c,configs=None,blocks=(0,4)):
    if configs is None: configs=range(8 if len(a)<=64 else 6)
    gpu_counts=torch.tensor(counts,device='cuda',dtype=torch.int64)
    ref=reference(a,b,counts)
    rows=[]
    for config in configs:
        for grid in blocks:
            runner=Ours(lib,gpu_counts,a,b,c,config,grid)
            try:
                c.fill_(float('nan'));runner.run();torch.cuda.synchronize()
                row=validate(c,ref)
                if config==0 and grid==0: row['cpu_double_samples']=cpu_double_check(a,b,c,counts)
                rows.append(dict(config=config,blocks=grid,**row))
            finally: runner.close()
    return rows


def validate_synthetic(lib,out,smoke):
    distributions=[[0,0],[0,1],[31,33],[0,129]] if smoke else [
        [0],[1],[0,0],[0,1],[1,0],[31,33],[65,7],[0]*16,[0]*15+[129],
        [1]*16,[64,96,160,192]*4,[0,32769],[32768,32768],[8691,12526]]
    rows=[]
    for counts in distributions:
        for k,n in [(768,2432),(2432,768),(1024,1024)]:
            for dtype in [torch.bfloat16,torch.float16]:
                a,b,c=tensor_case(counts,k,n,dtype)
                checks=check_all(lib,counts,a,b,c)
                rows.append(dict(counts=counts,k=k,n=n,dtype=str(dtype),pattern='random',checks=checks))
                print('VALIDATED',str(dtype),k,n,counts,flush=True)
    if not smoke:
        counts=[0,1,17,0,33,0,1,0,0,1,0,0,0,0,1,0]
        for dtype in [torch.bfloat16,torch.float16]:
            for k,n in [(768,2432),(2432,768),(1024,1024)]:
                a,b,c=tensor_case(counts,k,n,dtype)
                for pattern in ['zero','expert_constant','single_k']:
                    if pattern=='zero': a.zero_();b.fill_(1)
                    elif pattern=='expert_constant':
                        a.fill_(1)
                        for e in range(16): b[e].fill_((e+1)/k)
                    else:
                        a.zero_();b.zero_();a[:,37]=1
                        for e in range(16): b[e,37,:]=torch.arange(n,device='cuda')/n+(e+1)/16
                    checks=check_all(lib,counts,a,b,c)
                    rows.append(dict(counts=counts,k=k,n=n,dtype=str(dtype),pattern=pattern,checks=checks))
    (out/'synthetic_validation.json').write_text(json.dumps(rows,indent=2))
    return rows


def validate_live_counts(lib,out):
    rows=[]
    for k,n in [(768,2432),(2432,768),(1024,1024)]:
        a,b,c=tensor_case([1,64],k,n,torch.bfloat16)
        gpu_counts=torch.tensor([1,64],device='cuda',dtype=torch.int64)
        # Total M=65 deliberately also covers the regular path; Split-K is tested below for M=16.
        for config in range(6):
            for blocks in [0,4]:
                runner=Ours(lib,gpu_counts,a,b,c,config,blocks)
                graph=graph_make(runner.run,1)
                try:
                    for counts in [[0,65],[65,0],[31,34],[1,64]]:
                        gpu_counts.copy_(torch.tensor(counts,device='cuda',dtype=torch.int64))
                        c.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
                        rows.append(dict(k=k,n=n,config=config,blocks=blocks,counts=counts,
                                         **validate(c,reference(a,b,counts))))
                finally: del graph;runner.close()
    (out/'live_counts_validation.json').write_text(json.dumps(rows,indent=2))
    for k,n in [(768,2432),(2432,768),(1024,1024)]:
        a,b,c=tensor_case([1,15],k,n,torch.bfloat16)
        gpu_counts=torch.tensor([1,15],device='cuda',dtype=torch.int64)
        for config in [6,7]:
            runner=Ours(lib,gpu_counts,a,b,c,config,4)
            graph=graph_make(runner.run,1)
            try:
                for counts in [[0,16],[16,0],[7,9],[1,15]]:
                    gpu_counts.copy_(torch.tensor(counts,device='cuda',dtype=torch.int64))
                    c.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
                    rows.append(dict(k=k,n=n,config=config,blocks=4,counts=counts,
                                     **validate(c,reference(a,b,counts))))
            finally: del graph;runner.close()
    (out/'live_counts_validation.json').write_text(json.dumps(rows,indent=2))
    return rows


def validate_replays(lib,out,baseline,checkpoint):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    records=[]
    for path in sorted(baseline.glob('*/capture_rank*.pt')):
        data=torch.load(path,map_location='cpu',weights_only=True)
        for r in data['replays']:
            x=r['x'].cuda();counts=r['counts'];layer=r['layer'];ids=r['local_expert_ids']
            weights={name:torch.stack([state[f'model.layers.{layer}.mlp.experts.{i}.{name}.weight'].float()
                                     for i in ids]).cuda() for name in ['gate_proj','up_proj','down_proj']}
            # Preserve baseline's original BF16 autocast activation input for the down projection.
            parts=[];offset=0
            with torch.autocast('cuda',dtype=torch.bfloat16):
                for e,m in enumerate(counts):
                    if m:
                        a=x[offset:offset+m]
                        parts.append(F.silu(F.linear(a,weights['gate_proj'][e]))*F.linear(a,weights['up_proj'][e]))
                    offset+=m
            middle=torch.cat(parts) if parts else x.new_empty((0,2432))
            for name,a in [('gate_proj',x),('up_proj',x),('down_proj',middle)]:
                b=weights[name].bfloat16().transpose(1,2).contiguous()
                c=a.new_empty((len(a),b.shape[2]))
                checks=check_all(lib,counts,a,b,c)
                records.append(dict(case=path.parent.name,rank=r['rank'],layer=layer,projection=name,
                                    counts=counts,k=a.shape[1],n=b.shape[2],checks=checks))
            print('REPLAY_VALIDATED',path.parent.name,r['rank'],layer,flush=True)
    assert len(records)==108
    (out/'real_projection_validation.json').write_text(json.dumps(records,indent=2))
    return records


def benchmark(lib,baselines,out,selection_path=None,default_policy=False):
    sm=torch.cuda.get_device_properties(0).multi_processor_count
    families=cutlass_libraries(baselines)
    previous={r['case']:r for r in json.loads(Path(selection_path).read_text())} if selection_path else {}
    runtime=ctypes.CDLL('libcudart.so.12')
    runtime.cudaGetLastError.restype=ctypes.c_int
    cases=[('decode_single',[0,1]),('decode_batch',[8,8]),('prefill_medium',[1480,1432]),
           ('prefill_large',[10986,11477])]
    scenarios=[(f'{name}_{projection}',counts,k,n,torch.bfloat16)
               for name,counts in cases for projection,k,n in [('gate_up',768,2432),('down',2432,768)]]
    scenarios += [('regression_2048',[64,96,160,192]*4,1024,1024,torch.float16),
                  ('variable_4097',[0,1,31,33,129,256,0,511,0,0,0,0,0,0,0,3136],1024,1024,torch.float16)]
    results,tuning,failures=[],[],[]
    for name,counts,k,n,dtype in scenarios:
        a,b,c=tensor_case(counts,k,n,dtype)
        gpu_counts=torch.tensor(counts,device='cuda',dtype=torch.int64)
        ref=reference(a,b,counts)
        own_rows=[]
        configs=CONFIGS
        if default_policy:
            selected=lib.k9_default_config(sum(counts),n,k)
            configs=[dict(CONFIGS[selected],id=-1,resolved_id=selected,default_policy=True)]
        for config in configs:
            if config['id']>=6 and sum(counts)>64: continue
            for blocks in [0] if default_policy else [0,sm,2*sm]:
                runner=Ours(lib,gpu_counts,a,b,c,config['id'],blocks)
                try:
                    runner.run();torch.cuda.synchronize();validate(c,ref)
                    graph=graph_make(lambda:runner.run(1),16)
                    us=statistics.median(graph_time(graph,16) for _ in range(3))
                    del graph
                    row=dict(case=name,kind='kernel9',config=config,blocks=blocks,us=us)
                    own_rows.append(row);tuning.append(row)
                finally: runner.close()
        cut_rows=[]
        for kind,(clib,configs) in families.items():
            if previous:
                saved=previous[name]
                assert (saved['counts'],saved['k'],saved['n'],saved['dtype'])==(counts,k,n,str(dtype))
                best=saved['cutlass']
                if kind!=best['kind']:continue
                configs=[best['config']]
            for config in configs:
                for blocks in [best['blocks']] if previous else [0,sm,2*sm]:
                    runner=None
                    try:
                        runner=Cutlass(clib,config['id'],blocks,counts,a,b,c)
                        c.fill_(float('nan'));runner.run();torch.cuda.synchronize();validate(c,ref)
                        graph=graph_make(runner.run,16)
                        us=statistics.median(graph_time(graph,16) for _ in range(3))
                        del graph
                        row=dict(case=name,kind=kind,config=config,blocks=blocks,us=us)
                        cut_rows.append(row);tuning.append(row)
                    except RuntimeError as ex:
                        # Unsupported resources are reported; numerical failures are fatal.
                        code=runtime.cudaGetLastError()
                        failures.append(dict(case=name,kind=kind,config=config,blocks=blocks,error=str(ex),cuda_error_cleared=code))
                        (out/'unsupported_cutlass.json').write_text(json.dumps(failures,indent=2))
                        print('CUTLASS_UNSUPPORTED',name,kind,config['id'],blocks,str(ex),code,flush=True)
                    finally:
                        if runner is not None: runner.close()
        assert cut_rows
        def refine(rows,is_ours):
            best=None
            for row in sorted(rows,key=lambda r:r['us'])[:3]:
                runner=(Ours(lib,gpu_counts,a,b,c,row['config']['id'],row['blocks']) if is_ours else
                        Cutlass(families[row['kind']][0],row['config']['id'],row['blocks'],counts,a,b,c))
                try:
                    if is_ours: runner.run()
                    graph=graph_make((lambda:runner.run(1)) if is_ours else runner.run,64)
                    us=statistics.median(graph_time(graph,64,5) for _ in range(3));del graph
                    if best is None or us<best['refined_us']: best=dict(**row,refined_us=us)
                finally: runner.close()
            return best
        own_best,cut_best=refine(own_rows,True),refine(cut_rows,False)
        own=Ours(lib,gpu_counts,a,b,c,own_best['config']['id'],own_best['blocks'])
        cut=Cutlass(families[cut_best['kind']][0],cut_best['config']['id'],cut_best['blocks'],counts,a,b,c)
        graphs=[]
        try:
            own.run();torch.cuda.synchronize();validate(c,ref);cpu_double_check(a,b,c,counts,257)
            cut.run();torch.cuda.synchronize();validate(c,ref);cpu_double_check(a,b,c,counts,257)
            graphs=[('kernel9_compute',graph_make(lambda:own.run(1),64)),
                    ('kernel9_full',graph_make(own.run,64)),('CUTLASS_compute',graph_make(cut.run,64))]
            samples={label:[] for label,_ in graphs}
            for iteration in range(7):
                for j in range(3):
                    label,graph=graphs[(iteration+j)%3]
                    samples[label].append(graph_time(graph,64,10))
            medians={label:statistics.median(times) for label,times in samples.items()}
            result=dict(case=name,counts=counts,m=sum(counts),k=k,n=n,dtype=str(dtype),
                        medians_us=medians,samples_us=samples,ours=own_best,cutlass=cut_best,
                        compute_throughput_pct=100*medians['CUTLASS_compute']/medians['kernel9_compute'],
                        own_tflops=2*sum(counts)*n*k/(medians['kernel9_compute']*1e6),
                        input_seed=20261007,cutlass_selection_source=selection_path,
                        full_caveat='Full kernel9 includes device metadata; CUTLASS descriptors are prepared before timing.')
            results.append(result)
            print('RESULT',json.dumps(result),flush=True)
        finally:
            graphs.clear();own.close();cut.close()
        (out/'benchmark.json').write_text(json.dumps(results,indent=2))
        (out/'tuning.json').write_text(json.dumps(tuning,indent=2))
        (out/'unsupported_cutlass.json').write_text(json.dumps(failures,indent=2))
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--library',required=True)
    p.add_argument('--baselines',default='/data/user4/tongleiwen/moe-open-baselines')
    p.add_argument('--baseline',default='/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/results/R04_baseline')
    p.add_argument('--checkpoint',default='/data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth')
    p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--benchmark-only',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--cutlass-selection')
    p.add_argument('--default-policy',action='store_true')
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'benchmark_source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    lib=bind_ours(Path(args.library))
    with torch.inference_mode():
        if not args.benchmark_only:
            synthetic=validate_synthetic(lib,out,args.smoke)
            live=validate_live_counts(lib,out)
            replays=[] if args.smoke else validate_replays(lib,out,Path(args.baseline),args.checkpoint)
        if not args.smoke and not args.validate_only: benchmark(lib,Path(args.baselines),out,args.cutlass_selection,args.default_policy)
    environment=dict(python=os.sys.version,executable=os.sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(),SM=torch.cuda.get_device_properties(0).multi_processor_count,
                     physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),tf32=False,
                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
                     timing='CUDA Graph 64 nodes, seven rotated rounds, ten replays; preallocated tensors and workspace; hot cache',
                     full_timing='kernel9 metadata+compute; CUTLASS prebuilt descriptor compute')
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    print('ALL CHECKS PASSED',flush=True)


if __name__=='__main__':main()
