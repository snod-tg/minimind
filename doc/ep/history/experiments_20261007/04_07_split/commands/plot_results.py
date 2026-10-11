"""Latest standalone scientific figure; numerical eligibility stays explicit."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model-summary',type=Path);a=p.parse_args()
mlp=json.loads((a.results/'R11_mlp_benchmark/benchmark.json').read_text());models=json.loads((a.model_summary or (a.results/'R15_clean_model/summary_paired.json')).read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
fig,axes=plt.subplots(1,3,figsize=(17,4.8));colors=['#475569','#0284c7'];x=np.arange(len(mlp))
for i,key in enumerate(['original','cuda_split']):axes[0].bar(x+(i-.5)*.34,[r['graph_us'][key] for r in mlp],.34,color=colors[i],label=key)
axes[0].set_xticks(x,[str(r['m']) for r in mlp]);axes[0].set_yscale('log');axes[0].set_xlabel('Local M');axes[0].set_ylabel('Latency (microseconds, log scale)');axes[0].set_title('Complete expert MLP: CUDA Graph');axes[0].legend(frameon=False);axes[0].text(.02,.96,'Real weights / synthetic input\nCUDA: FP32 accumulation',transform=axes[0].transAxes,va='top',fontsize=9)
for axis,cat,title in [(axes[1],'model','Full-model forward'),(axes[2],'growing_decode','Growing KV: 8 Decode steps')]:
 rows=models[cat];labels=[r.get('case',f"B{r['batch']}/top{r['topk']}").replace('prefill_','P-').replace('decode_','D-').replace('_top','/T') for r in rows]
 speed=[100*(r['paths']['original']['wall']['median_ms']/r['paths']['cuda_split_last_layer']['wall']['median_ms']-1) for r in rows]
 y=np.arange(len(rows));axis.barh(y,speed,color='#0284c7');axis.set_yticks(y,labels);axis.invert_yaxis();axis.axvline(0,color='#475569',lw=1);axis.set_xlabel('Throughput change vs original (%)');axis.set_title(title+'\nLast-layer small-M CUDA only');axis.grid(axis='x',alpha=.2);axis.set_axisbelow(True)
fig.suptitle('RTX 4090D | BF16 I/O | Full CUDA model fails numerical admission',fontsize=14,y=1.02)
fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180,bbox_inches='tight');plt.close(fig)
