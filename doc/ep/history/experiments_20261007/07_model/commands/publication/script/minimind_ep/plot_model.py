"""Model latency and conservative-policy variability; rejected variants labeled."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=json.loads(a.results.read_text())['model'];y=np.arange(len(rows));width=.18
labels=[r['case'].replace('_top',' / top').replace('_',' ') for r in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                    'axes.facecolor':'#f8fafc','grid.color':'#cbd5e1','grid.alpha':.45})
fig,axes=plt.subplots(1,2,figsize=(15,8),sharey=True);colors=plt.get_cmap('tab10').colors
for i,(key,name) in enumerate([('original','Original'),('kernel9','Kernel9 all layers (accuracy failed)'),
                              ('hybrid','Hybrid M<=256 (accuracy failed)'),('last_layer','Last layer M<=256 (tested pass)')]):
    axes[0].barh(y+(i-1.5)*width,[r['paths'][key]['wall']['median_ms'] for r in rows],height=width,color=colors[i],label=name)
axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_title('Whole model: slower-rank wall median');axes[0].set_xlabel('Latency per forward (ms)')
for i,r in enumerate(rows):
    blocks=r['block_reduction_percent']['last_layer'];point=r['latency_reduction_percent']['last_layer']
    axes[1].plot([min(blocks),max(blocks)],[i,i],color=colors[3],linewidth=2,alpha=.6)
    axes[1].scatter([point],[i],color=colors[3],s=40,zorder=3)
    if max(blocks)>2:
        axes[1].scatter([1.93],[i],marker='>',color=colors[3],s=45)
        axes[1].text(1.1,i+.25,f'{max(blocks):.1f}% (1 block)',fontsize=8,color='#64748b')
axes[1].set_xlim(-1.25,2)
axes[1].axvline(0,color='#64748b',linestyle='--');axes[1].set_title('Conservative policy: reduction and 5-block range')
axes[1].set_xlabel('Latency reduction vs original (%) | positive = faster')
for ax in axes:ax.grid(axis='x');ax.set_axisbelow(True)
fig.suptitle('MiniMind-EP: from local GEMM to whole-model evidence',x=.02,ha='left',fontweight='bold')
fig.text(.02,.925,'2 x RTX 4090D GPU4/5 | EP=2 | BF16 autocast | CPU 28/29 | 60 calls/backend, 5 rotated blocks | 2026-10-07',fontsize=9,color='#475569')
handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.045))
fig.text(.02,.025,'Includes routing, communication, workspace prepare, attention, all-position logits and KV output. Hot cache, unlocked clocks.',fontsize=9,color='#475569')
fig.text(.02,.005,'Block ranges indicate variability, not confidence intervals. Right panel zooms to -1.25%..2%; outlier block is annotated.',fontsize=9,color='#475569')
fig.tight_layout(rect=[0,.16,1,.9]);a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180,bbox_inches='tight')
