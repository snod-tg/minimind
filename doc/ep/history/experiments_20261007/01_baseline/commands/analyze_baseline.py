"""Analyze raw Chrome events; exclude GPU copies of user annotations."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import statistics


OUTER = {'router', 'rank_pack', 'count_exchange', 'split_to_cpu', 'dispatch',
         'expert_sort', 'expert_counts_to_cpu', 'experts', 'inverse_expert',
         'combine', 'restore'}


def stats(values):
    ordered = sorted(values)
    return dict(n=len(values), median_ms=statistics.median(values),
                mean_ms=statistics.mean(values), min_ms=min(values), max_ms=max(values),
                p95_ms=ordered[int(.95*(len(values)-1))])


def trace_summary(path, repeats):
    events = json.loads(path.read_text())['traceEvents']
    annotations = [e for e in events if e.get('cat') == 'user_annotation'
                   and e.get('name', '').startswith('baseline/L')]
    outer = [e for e in annotations if e['name'].split('/')[-1] in OUTER]
    by_thread = defaultdict(list)
    for e in outer:
        by_thread[e['pid'], e['tid']].append(e)
    indexes = {}
    for key, rows in by_thread.items():
        rows.sort(key=lambda e: e['ts'])
        indexes[key] = ([e['ts'] for e in rows], rows)

    def owner(event):
        index = indexes.get((event.get('pid'), event.get('tid')))
        if not index:
            return None
        starts, rows = index
        i = bisect_right(starts, event['ts'])-1
        if i >= 0 and event['ts'] <= rows[i]['ts'] + rows[i]['dur']:
            return rows[i]['name'].split('/')[-1]
        return None

    cpu_by_external, runtime_by_correlation = {}, {}
    for e in events:
        args = e.get('args', {})
        if e.get('cat') in ('cpu_op', 'user_annotation') and 'External id' in args:
            cpu_by_external[args['External id']] = e
        if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in args:
            runtime_by_correlation[args['correlation']] = e
    stages = {name: dict(cpu_ms=0., gpu_ms=0., cpu_calls=0, kernel_calls=0,
                         sync_cpu_ms=0., sync_calls=0) for name in OUTER}
    for e in outer:
        row = stages[e['name'].split('/')[-1]]
        row['cpu_ms'] += e['dur']/1000/repeats
        row['cpu_calls'] += 1
    for e in events:
        if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'Synchronize' in e.get('name', ''):
            name = owner(e)
            if name:
                stages[name]['sync_cpu_ms'] += e['dur']/1000/repeats
                stages[name]['sync_calls'] += 1
    kernels = defaultdict(lambda: dict(calls=0, duration_ms=0.))
    matched, unmatched = 0, 0
    for e in events:
        if e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        args = e.get('args', {})
        origin = cpu_by_external.get(args.get('External id'))
        name = owner(origin) if origin else None
        if name is None:
            origin = runtime_by_correlation.get(args.get('correlation'))
            name = owner(origin) if origin else None
        if name:
            stages[name]['gpu_ms'] += e['dur']/1000/repeats
            stages[name]['kernel_calls'] += e.get('cat') == 'kernel'
            matched += 1
        else:
            unmatched += 1  # Attention, embeddings and other non-EP work are expected.
        if e.get('cat') == 'kernel':
            kernels[e['name']]['calls'] += 1
            kernels[e['name']]['duration_ms'] += e['dur']/1000/repeats
    gemm = {k: v for k, v in kernels.items()
            if any(token in k.lower() for token in ('gemm', 'gemv', 'mma'))}
    return dict(stages=stages,
                kernel_top=sorted([dict(name=k, **v) for k,v in kernels.items()],
                                  key=lambda x:x['duration_ms'], reverse=True)[:20],
                gemm_kernel_ms=sum(v['duration_ms'] for v in gemm.values()),
                cpu_annotations=len(annotations), matched_gpu_events=matched,
                unmatched_non_ep_gpu_events=unmatched,
                note='CPU inclusive ranges and GPU duration sums are diagnostic; neither is an additive critical-path breakdown.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rank_results = [json.loads((source/f'summary_rank{rank}.json').read_text()) for rank in (0,1)]
    assert len(rank_results[0]) == len(rank_results[1]) == 10
    cases = []
    for a,b in zip(*rank_results):
        assert a['case'] == b['case']
        assert a['annotation_equal'] and b['annotation_equal']
        assert all(m['exact_replay'] for x in (a,b) for m in x['micro_mlp'])
        global_assignments = a['local_batch']*a['seq_len']*a['topk']*2
        for ar,br in zip(a['counts'],b['counts']):
            assert ar['layer'] == br['layer']
            assert ar['total_m']+br['total_m'] == global_assignments
        paired = [max(x,y) for x,y in zip(a['e2e']['wall_samples_ms'],b['e2e']['wall_samples_ms'])]
        diagnoses = [trace_summary(source/a['case']/f'trace_rank{rank}.json',3) for rank in (0,1)]
        micro=[]
        for ar,br in zip(a['micro_mlp'],b['micro_mlp']):
            assert ar['layer']==br['layer']
            mlp=stats([max(x,y) for x,y in zip(ar['timing']['wall_samples_ms'],br['timing']['wall_samples_ms'])])
            ep=stats([max(x,y) for x,y in zip(ar['ep_layer_timing']['wall_samples_ms'],br['ep_layer_timing']['wall_samples_ms'])])
            micro.append(dict(layer=ar['layer'], counts=[ar['counts'],br['counts']], mlp=mlp,ep=ep))
        cases.append(dict(case=a['case'], topk=a['topk'], batch_per_rank=a['local_batch'],
                          seq_len=a['seq_len'], context_len=a['context_len'],
                          e2e=stats(paired), rank_e2e=[a['e2e']['wall'],b['e2e']['wall']],
                          micro=micro, counts=[a['counts'],b['counts']], diagnoses=diagnoses))
    document=dict(source_run='R04_baseline', correctness='20 annotation comparisons; all MLP/EP replays exact; assignment counts conserved in all layers',
                  cases=cases,
                  caution='Raw profiler_stages/profiler_kernels fields double-count CPU/GPU annotations; use this raw-trace-derived analysis for diagnostic data.')
    (output/'analysis.json').write_text(json.dumps(document,indent=2,ensure_ascii=False))
    for case in cases:
        print(case['case'], round(case['e2e']['median_ms'],3),
              'MLP',[(m['layer'],round(m['mlp']['median_ms'],3),round(m['ep']['median_ms'],3)) for m in case['micro']])
    print(document['correctness'])


if __name__=='__main__':
    main()
