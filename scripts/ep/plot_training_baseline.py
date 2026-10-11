"""Plot accepted short-training baselines and per-update CUDA memory."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stem',default='ep6_top2_fair_training')
    args=parser.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    assert summary['passed'] and summary['occupancy_clean']
    rows=summary['results']
    names={'ddp':'DDP (no EP)','ep_original':'Original EP','ep_legacy':'Previous compact','ep_compact':'Pooled compact','ep_dedup':'Rank-dedup EP'}
    palette={'ddp':'#637b94','ep_original':'#d89b49','ep_legacy':'#d89b49','ep_compact':'#347fa1','ep_dedup':'#398969'}
    labels=[names[row['variant']] for row in rows]
    colors=[palette[row['variant']] for row in rows]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13.4,4.8))
    metrics=[('valid_tokens_per_second',1/1000,'Valid tokens/s (thousands)'),
             ('optimizer_update_seconds',1,'Seconds / optimizer update'),
             ('peak_allocated',1/(1024**3),'Peak allocated / rank (GiB)')]
    for ax,(field,factor,title) in zip(axes,metrics):
        values=[row[field]*factor for row in rows]
        ax.barh(labels,values,color=colors,height=.56)
        ax.invert_yaxis();ax.set_title(title,pad=12)
        ax.set_xlim(0,max(values)*1.23);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
        for i,value in enumerate(values):ax.text(value+max(values)*.025,i,f'{value:.2f}',va='center')
    fig.suptitle('MiniMind 6 experts / Top-2: matched training baseline',fontsize=15,y=.98)
    fig.text(.5,.045,'2 x RTX 4090D | BF16 autocast | eager | batch 32/rank, seq 340, accumulation 8\n'
        f'Same initialization and data; median of {summary["repeats"]} windows, {summary["measured_updates"]} updates/window; memory is maximum across ranks and trials.',
        ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=[0,.15,1,.91]);args.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output/(args.stem+'.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.6))
    for label,row,color in zip(labels,rows,colors):
        curves=[]
        for i in range(summary['repeats']):
            payload=json.loads((args.run/f'performance_{i}_{row["variant"]}/summary.json').read_text())
            curves.append([max(r['updates'][u]['peak_allocated'] for r in payload['ranks'])/(1024**3)
                for u in range(len(payload['ranks'][0]['updates']))])
        y=[statistics.median(v) for v in zip(*curves)]
        ax.plot(range(1,len(y)+1),y,label=label,color=color,marker='o',markersize=4)
    warmup=summary['warmup_updates']
    ax.axvspan(.5,warmup+.5,color='#e7edf2',alpha=.7,zorder=-1)
    ax.axvline(warmup+.5,color='#87949d',linestyle='--',linewidth=1)
    ax.set_xlabel('Optimizer update');ax.set_ylabel('Running peak allocated / rank (GiB)')
    ax.set_title('Memory during warmup and measured training')
    ax.set_xticks(range(1,len(y)+1));ax.grid(alpha=.18);ax.legend(loc='best')
    fig.text(.5,.015,f'Shaded: warmup. CUDA peak counters reset after warmup. Median of {summary["repeats"]} trials; running peaks, max across ranks.',
        ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(args.output/(args.stem+'_memory.png'),dpi=180);plt.close(fig)


if __name__=='__main__':main()
