import argparse,json,sys,time,statistics
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--source',type=Path,default=Path('source'));p.add_argument('--output',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args()
sys.path.insert(0,str(a.source/'dev/pytorch'))
from minimind_ep_routing import Workspace

torch.ops.load_library(a.library);torch.set_num_threads(1);torch.manual_seed(20261009);ws=Workspace();rows=[];timings=[]
def measure(fn):
    for _ in range(10):fn()
    torch.cuda.synchronize();v=[]
    for _ in range(5):
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):fn()
        e.record();e.synchronize();v.append(s.elapsed_time(e)*1000/50)
    return statistics.median(v)
with torch.inference_mode():
    for n in ([0,1,33,257,513] if a.quick else [0,1,2,31,32,33,255,256,257,511,512,513,21760,65537]):
        for divisor in [1,2]:
            ids=torch.randint(divisor*2,(n,),device='cuda');perm,inv,counts=ws.sort(ids,divisor,'test')
            expected=torch.argsort(ids//divisor,stable=True)
            assert torch.equal(perm,expected),(n,divisor)
            assert torch.equal(counts,torch.bincount(ids//divisor,minlength=2))
            assert torch.equal(perm[inv],torch.arange(n,device='cuda'))
            rows.append(dict(test='stable_sort',n=n,divisor=divisor,exact=True))
    for n in [0,33,257,21760]:
        ids=torch.randint(3,(n,),device='cuda');perm,inv,counts=ws.sort(ids,1,'three',3)
        assert torch.equal(perm,torch.argsort(ids,stable=True)) and torch.equal(counts,torch.bincount(ids,minlength=3))
        rows.append(dict(test='three_expert_partition',n=n,exact=True))
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        for topk in [1,2]:
            for n,h in [(0,768),(1,7),(33,768),(257,769),(1025,768)]:
                x=torch.randn(n,h,device='cuda',dtype=dtype);ids=torch.randint(4,(n*topk,),device='cuda')
                perm,inv,counts=ws.sort(ids,2,'test');out=torch.empty(n*topk,h,device='cuda',dtype=dtype);local=torch.empty_like(ids)
                torch.ops.my_cuda_ep.pack_out(x,ids,perm,topk,2,out,local)
                assert torch.equal(out,x[perm//topk]) and torch.equal(local,ids[perm]%2)
                restored=torch.empty_like(out);torch.ops.my_cuda_ep.gather_out(out,inv,restored)
                assert torch.equal(restored,x.repeat_interleave(topk,dim=0))
                rows.append(dict(test='pack_gather',dtype=str(dtype),n=n,h=h,topk=topk,exact=True))
        for wtype in [torch.float32,torch.float16,torch.bfloat16]:
            for otype in [torch.float32,torch.float16,torch.bfloat16]:
                for topk in [1,2]:
                    n,h=257,769
                    ids=torch.randint(4,(n*topk,),device='cuda');perm,inv,_=ws.sort(ids,2,'test')
                    returned=(torch.randn(n*topk,h,device='cuda')*3).to(dtype)
                    weights=torch.rand(n*topk,device='cuda').to(wtype)
                    out=torch.empty(n,h,device='cuda',dtype=otype)
                    torch.ops.my_cuda_ep.combine_out(returned,weights,inv,topk,out)
                    weighted=(returned[inv]*weights[:,None]).to(otype)
                    expected=torch.zeros_like(out).index_add_(0,torch.arange(n,device='cuda').repeat_interleave(topk),weighted)
                    assert torch.equal(out,expected),(dtype,wtype,otype,topk,int((out!=expected).sum()),float((out-expected).abs().max()))
                    rows.append(dict(test='combine',dtype=str(dtype),weight=str(wtype),out=str(otype),topk=topk,exact=True))
    for experts in [2,3]:
        for dtype in [torch.float32,torch.bfloat16]:
            for n in [0,1,16,32]:
                for topk in [1,2]:
                    capacity=32*topk;h=768;x=torch.randn(n,h,device='cuda',dtype=dtype)
                    ids=torch.randint(2*experts,(n*topk,),device='cuda');perm,_,counts=ws.sort(ids,experts,'fixed_send')
                    wire=torch.empty(2*capacity,h,device='cuda',dtype=dtype);meta=torch.empty(2*capacity,device='cuda',dtype=torch.int64);inverse=torch.empty_like(ids)
                    torch.ops.my_cuda_ep.pack_fixed_out(x,ids,perm,counts,topk,experts,wire,meta,inverse)
                    pmap=torch.empty_like(meta);imap=torch.empty_like(meta);ecount=torch.empty(experts,device='cuda',dtype=torch.int64)
                    torch.ops.my_cuda_ep.masked_sort_out(meta,pmap,imap,ecount)
                    valid=torch.nonzero(meta>=0,as_tuple=True)[0];expected=valid[torch.argsort(meta[valid],stable=True)];r=len(valid)
                    assert torch.equal(pmap[:r],expected) and torch.equal(ecount,torch.bincount(meta[valid],minlength=experts))
                    packed=torch.empty(r,h,device='cuda',dtype=dtype);torch.ops.my_cuda_ep.gather_out(wire,pmap[:r],packed)
                    back=torch.empty_like(wire);torch.ops.my_cuda_ep.scatter_fixed_out(packed,imap,back)
                    assert torch.equal(back,wire)
                    out=torch.empty_like(x);weights=torch.rand(n*topk,device='cuda',dtype=dtype)
                    torch.ops.my_cuda_ep.combine_out(back,weights,inverse,topk,out)
                    reference=torch.zeros_like(x).index_add_(0,torch.arange(n,device='cuda').repeat_interleave(topk),(x.repeat_interleave(topk,dim=0)*weights[:,None]).to(dtype))
                    assert torch.equal(out,reference)
                    rows.append(dict(test='fixed_capacity',experts=2*experts,dtype=str(dtype),n=n,topk=topk,exact=True))
    # Changing assignments with identical capacity must rebuild counts/maps.
    n=137;ids=torch.zeros(n*2,device='cuda',dtype=torch.int64);perm,inv,counts=ws.sort(ids,2,'dynamic')
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):ws.sort(ids,2,'dynamic')
    for value in [3,0,2,1]:
        ids.fill_(value);graph.replay();assert counts.cpu().tolist()==([n*2,0] if value<2 else [0,n*2])
    rows.append(dict(test='graph_changed_routing',exact=True))
    before=ws.growth
    for _ in range(10):ws.sort(ids,2,'dynamic')
    assert before==ws.growth
    rows.append(dict(test='buffer_reuse',exact=True))
    # Shape/stride/type checks reject unsafe public API usage.
    for kind in ['noncontiguous','wrong_dtype','wrong_size']:
        ids=torch.randint(4,(34,),device='cuda');perm,inv,counts=ws.sort(ids,2,'invalid')
        x=torch.randn(17,768,device='cuda');out=torch.empty(34,768,device='cuda');local=torch.empty_like(ids)
        bad=x[:,::2] if kind=='noncontiguous' else x
        badids=ids.int() if kind=='wrong_dtype' else ids[:-1] if kind=='wrong_size' else ids
        try:torch.ops.my_cuda_ep.pack_out(bad,badids,perm,2,2,out,local)
        except RuntimeError:pass
        else:raise AssertionError(kind)
        rows.append(dict(test=kind,rejected=True))
    if not a.quick:
        for n in [1,16,137,128,2720,10880]:
            for topk in [1,2]:
                x=torch.randn(n,768,device='cuda',dtype=torch.bfloat16);ids=torch.randint(4,(n*topk,),device='cuda');weights=torch.rand(n*topk,device='cuda',dtype=torch.bfloat16)
                def original():
                    rank=ids//2;perm=torch.argsort(rank,stable=True);send=x[torch.arange(n,device='cuda').repeat_interleave(topk)][perm]
                    local=ids[perm]%2;counts=torch.bincount(rank,minlength=2)
                    inv=torch.empty_like(perm);inv[perm]=torch.arange(len(perm),device='cuda')
                    y=torch.zeros_like(x);y.index_add_(0,torch.arange(n,device='cuda').repeat_interleave(topk),(send[inv]*weights[:,None]).to(x.dtype));return y
                perm,inv,counts=ws.sort(ids,2,'bench');send=torch.empty(n*topk,768,device='cuda',dtype=x.dtype);local=torch.empty_like(ids);out=torch.empty_like(x)
                def fused():
                    ws.sort(ids,2,'bench');torch.ops.my_cuda_ep.pack_out(x,ids,perm,topk,2,send,local);torch.ops.my_cuda_ep.combine_out(send,weights,inv,topk,out);return out
                assert torch.equal(original(),fused())
                timings.append(dict(n=n,topk=topk,original_us=measure(original),fused_us=measure(fused)))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(checks=rows,timing=timings),indent=2)+'\n');print('OPS PASSED',len(rows),timings,flush=True)
