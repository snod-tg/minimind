"""Pair ranks before summarizing latency; correlate diagnostic trace without adding stages."""
import argparse,json,statistics
from collections import defaultdict
from bisect import bisect_right
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--profile',type=Path);a=p.parse_args()

def stats(values):
    values=sorted(values)
    return dict(n=len(values),median_ms=statistics.median(values),p95_ms=values[int(.95*(len(values)-1))],
                min_ms=min(values),max_ms=max(values))

def paired(file):
    if not (a.input/f'{file}_rank0.json').exists():return []
    ranks=[json.loads((a.input/f'{file}_rank{i}.json').read_text()) for i in [0,1]];rows=[]
    assert len(ranks[0])==len(ranks[1])
    for left,right in zip(*ranks):
        assert left.get('case')==right.get('case')
        result={k:v for k,v in left.items() if k not in ['rank','paths']};result['paths']={}
        for backend in left['paths']:
            paths=[left['paths'][backend],right['paths'][backend]]
            assert len(paths[0]['wall'])==len(paths[1]['wall'])
            values=[max(x,y) for x,y in zip(paths[0]['wall'],paths[1]['wall'])]
            event=[max(x,y) for x,y in zip(paths[0]['event'],paths[1]['event'])]
            result['paths'][backend]=dict(wall=stats(values),event=stats(event),samples_ms=values,
                                          rank_wall=[stats(r['wall']) for r in paths],
                                          prepare_events=[r['prepare_events'] for r in paths])
        original=result['paths']['original']['wall']['median_ms']
        result['latency_reduction_percent']={k:100*(1-v['wall']['median_ms']/original) for k,v in result['paths'].items()}
        # Compare the rotated-block effects; distribution overlap does not imply significance.
        block_size=len(values)//5
        result['block_reduction_percent']={}
        for k,v in result['paths'].items():
            new=v['samples_ms'];old=result['paths']['original']['samples_ms']
            result['block_reduction_percent'][k]=[100*(1-statistics.median(new[s:s+block_size])/statistics.median(old[s:s+block_size]))
                                                    for s in range(0,len(new),block_size)]
        rows.append(result)
    return rows

def trace(path):
    events=json.loads(path.read_text())['traceEvents']
    annotations=[e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('ep/')]
    thread=defaultdict(list)
    for e in annotations:thread[e['pid'],e['tid']].append(e)
    indexes={}
    for key,items in thread.items():
        items.sort(key=lambda e:e['ts']);indexes[key]=([e['ts'] for e in items],items)
    def owner(event):
        if event is None:return None
        index=indexes.get((event.get('pid'),event.get('tid')))
        if not index:return None
        starts,items=index;i=bisect_right(starts,event['ts'])-1
        return items[i]['name'].split('/')[-1] if i>=0 and event['ts']<=items[i]['ts']+items[i]['dur'] else None
    external={};correlation={};stages=defaultdict(lambda:dict(cpu_ms=0.,gpu_ms=0.,kernels=0,sync_cpu_ms=0.))
    for e in events:
        args=e.get('args',{})
        if e.get('cat') in ['cpu_op','user_annotation'] and 'External id' in args:external[args['External id']]=e
        if e.get('cat') in ['cuda_runtime','cuda_driver'] and 'correlation' in args:correlation[args['correlation']]=e
    for e in annotations:stages[e['name'].split('/')[-1]]['cpu_ms']+=e['dur']/3000
    for e in events:
        if e.get('cat') in ['cuda_runtime','cuda_driver'] and 'Synchronize' in e.get('name',''):
            name=owner(e)
            if name:stages[name]['sync_cpu_ms']+=e['dur']/3000
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
        args=e.get('args',{});name=owner(external.get(args.get('External id'))) or owner(correlation.get(args.get('correlation')))
        if name:
            stages[name]['gpu_ms']+=e['dur']/3000;stages[name]['kernels']+=e.get('cat')=='kernel'
    return dict(stages=dict(stages),note='3 profiled forwards; CPU ranges and GPU sums are diagnostic, non-additive; timing measured separately')

document=dict(model=paired('model'),ep=paired('ep'),growing_decode=paired('growing_decode'))
if a.profile:document['profiles']={f.stem:trace(f) for f in a.profile.glob('*_rank*.json') if f.name.startswith(('decode_batch_','prefill_medium_','prefill_small_'))}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(document,indent=2)+'\n')
for r in document['model']:print(r['case'],{k:round(v['wall']['median_ms'],3) for k,v in r['paths'].items()},r['latency_reduction_percent'])
