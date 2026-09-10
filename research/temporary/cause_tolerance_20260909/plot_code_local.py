"""Static, independently exportable plot of the full factory comparison."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.summary.read_bytes());a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
      'svg.fonttype':'none','svg.hashsalt':'code-local-v1','figure.dpi':160})
    fig,ax=plt.subplots(figsize=(9.2,5.4));xs=list(range(4));width=.32
    cells=[x for x in d['cells'] if x['run']=='qwen35_code_local_production']
    for shift,mode,label,color in [(-width/2,'retained','Retained capture','#8998a7'),(width/2,'local','Code-local capture','#147c78')]:
        rows=[next(x for x in cells if x['target_input_tokens']==n and x['mode']==mode) for n in [128,256,512,1024]]
        bars=ax.bar([x+shift for x in xs],[1000*r['seconds_mean'] for r in rows],width,
          yerr=[1000*r['seconds_std'] for r in rows],capsize=3,label=label,color=color)
        ax.bar_label(bars,labels=[f"{1000*r['seconds_mean']:.0f}" for r in rows],padding=5,fontsize=10)
    ax.set_ylim(0,960);ax.set_ylabel('Complete attribution latency (ms)')
    ax.set_xticks(xs,[str(x['actual_total_tokens']) for x in d['production_comparisons']])
    ax.set_xlabel('Actual model sequence length, including target + EOS (tokens)')
    ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(),frameon=False,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2)
    for i,c in enumerate(d['production_comparisons']):
        ax.text(i,900,f"{c['latency_change_percent']:+.1f}%",ha='center',fontweight='bold',color='#147c78' if i<3 else '#666666')
    fig.suptitle('Qwen3.5: lower passive-capture overhead on short inputs',fontweight='bold',y=.97)
    fig.text(.5,.025,'One sample · 3 interleaved warmed calls per version (mean ± SD) · MetaX C550\nFull vectors match on 4 synthetic and 11 original short examples. Largest bin shows no established gain.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.11,1,.85))
    fig.savefig(a.output/'code_local_latency.png');fig.savefig(a.output/'code_local_latency.svg',metadata={'Date':None});plt.close(fig)
    svg=a.output/'code_local_latency.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
