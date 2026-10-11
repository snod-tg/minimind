"""Only the final optimized EP policy versus the original PyTorch EP."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rank0,rank1=[json.loads((a.input/f'rank{i}.json').read_text()) for i in range(2)]
rows=[]
for x,y in zip(rank0['checks'],rank1['checks']):
    medians={key:statistics.median(map(max,zip(x['samples'][key],y['samples'][key])))/1000 for key in x['samples']}
    backend='fixed' if x['case'].startswith('decode') else 'compact'
    rows.append(dict(case=x['case'],topk=x['topk'],original_ms=medians['original'],optimized_ms=medians[backend],backend=backend))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
labels=['Decode\nB1','Decode\nB16','Prefill\nM128','Prefill\nM2720','Prefill\nM10880']
for topk,ax in enumerate(axes,1):
    selected=[r for r in rows if r['topk']==topk]
    original=np.array([r['original_ms'] for r in selected]);optimized=np.array([r['optimized_ms'] for r in selected]);x=np.arange(len(selected))
    ax.bar(x-.18,original,.36,color='#4e79a7',label='Original PyTorch EP')
    bars=ax.bar(x+.18,optimized,.36,color='#f28e2b',label='Optimized EP')
    for bar,o,v in zip(bars,original,optimized):ax.text(bar.get_x()+bar.get_width()/2,max(o,v)+1,f'-{100*(1-v/o):.1f}%',ha='center',fontsize=9)
    ax.set_xticks(x,labels);ax.set_title(f'Top-{topk}');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Full-model median latency (ms)')
axes[1].legend(frameon=False,loc='upper left')
fig.suptitle('Native BF16 experts: stable routing and fused output merge')
fig.text(.5,.015,'RTX 4090D x 2 | 4-expert checkpoint | 8 layers | 21 paired rounds | last-position logits\nDecode: fixed-capacity dispatch; Prefill: compact dispatch; both use native PyTorch expert GEMMs.',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.11,1,.93))
a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output,bbox_inches='tight')
(a.output.parent/'ep_routing_optimization.json').write_text(json.dumps(rows,indent=2)+'\n')
print(a.output,flush=True)
