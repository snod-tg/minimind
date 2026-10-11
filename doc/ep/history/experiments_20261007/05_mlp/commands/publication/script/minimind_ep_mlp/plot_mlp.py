"""True complete-MLP latency, with eager and CUDA Graph scopes kept separate."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=sorted(json.loads(a.results.read_text()),key=lambda r:r['m'])
labels=[f'{r["case"].replace("_top1"," / top1").replace("_top2"," / top2").replace("_"," ")}\nM={r["m"]}, counts={r["counts"]}' for r in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                    'axes.facecolor':'#f8fafc','grid.color':'#cbd5e1','grid.alpha':.45})
fig,axes=plt.subplots(1,2,figsize=(15,9),sharey=True);colors=plt.get_cmap('tab10').colors;y=np.arange(len(rows));width=.23
for ax,metric,title in zip(axes,['graph_us','eager_wall_us'],['CUDA Graph: full MLP execution','Eager: full MLP enqueue + final synchronize']):
    for i,(key,label) in enumerate([('original_fp32_autocast','Original per-expert PyTorch'),
                                   ('grouped_preallocated','Kernel9 grouped + reused buffers'),
                                   ('cutlass_static_preallocated','Tuned CUTLASS + static descriptors')]):
        ax.barh(y+(i-1)*width,[r[metric][key] for r in rows],height=width,color=colors[i],label=label)
    ax.set_xscale('log');ax.set_xlabel('Latency per MLP (us, log scale)');ax.set_title(title);ax.grid(axis='x');ax.set_axisbelow(True)
axes[0].set_yticks(y,labels);axes[0].invert_yaxis()
fig.suptitle('MiniMind-EP: real local expert MLP replay',x=.02,ha='left',fontweight='bold')
fig.text(.02,.925,'RTX 4090D GPU1 | BF16, 768 -> 2432 -> 768 | checkpoint weights and true inputs | 7 rotated rounds | 2026-10-07',fontsize=9,color='#475569')
handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.055))
fig.text(.02,.025,'Includes 3 GEMMs + SiLU + multiply. Counts/weights/workspaces prepared outside timing. CUTLASS descriptors are static.',fontsize=9,color='#475569')
fig.text(.02,.005,'Replay excludes routing, communication and whole-model forward. Eager C ABI and dispatcher costs differ.',fontsize=9,color='#475569')
fig.tight_layout(rect=[0,.12,1,.9]);a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180,bbox_inches='tight')
