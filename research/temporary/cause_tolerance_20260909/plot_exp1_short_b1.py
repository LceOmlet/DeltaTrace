"""Publication-style static plots of the retained exp1 timing measurements."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.summary.read_bytes());a.output.mkdir(parents=True,exist_ok=True)
    assert len(d['warm_comparisons'])==16 and all(not c['DT_faster'] for c in d['warm_comparisons'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
      'svg.fonttype':'none','svg.hashsalt':'exp1-short-b1-v1','figure.dpi':160})
    families=[('qwen3','Qwen3-8B / FP16'),('qwen35','Qwen3.5-9B / BF16')]
    core=[('deltatrace_retained','DeltaTrace (retained)','#b64132','o','-'),
          ('ifr_multi_hop_both','FT Both','#157a78','s','-'),
          ('ifr_multi_hop','FT multi-hop','#4d6fa9','^','--')]
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.8),sharey=True)
    warm_top=1100*max(c['seconds_mean']+c['seconds_std'] for c in d['cells'] if c['phase']=='warm3' and c['valid_timing'])
    for ax,(family,title) in zip(axes,families):
        for method,label,color,marker,style in core:
            cells=sorted((c for c in d['cells'] if c['family']==family and c['method']==method and c['phase']=='warm3' and c['valid_timing']),key=lambda x:x['actual_total_tokens'])
            ax.errorbar([c['actual_total_tokens'] for c in cells],[1000*c['seconds_mean'] for c in cells],
                yerr=[1000*c['seconds_std'] for c in cells],color=color,marker=marker,linestyle=style,
                linewidth=1.8,markersize=5,capsize=3,label=label)
        ax.set_title(title,fontweight='bold',pad=10)
        ax.set_xlabel('Actual model sequence length (tokens)')
        ax.set_xlim(100,960);ax.set_ylim(0,warm_top);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Full attribution latency (ms)')
    axes[0].legend(frameon=False,loc='upper left')
    fig.suptitle('Before capture optimization: retained DT versus FT',fontweight='bold',y=.98)
    fig.text(.5,.025,'3 warmed full calls (mean ± SD) · fixed 32-token target + EOS · MetaX C550\nSequence length includes the formatted prompt, target and EOS; model loading is separate.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.11,1,.91))
    fig.savefig(a.output/'exp1_short_b1_warm.png')
    fig.savefig(a.output/'exp1_short_b1_warm.svg',metadata={'Date':None})
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.4),sharey=True)
    for ax,(family,title) in zip(axes,families):
        for method,label,color,marker,style in core:
            cells=sorted((c for c in d['cells'] if c['family']==family and c['method']==method and c['phase']=='warm3' and c['valid_timing']),key=lambda x:x['actual_total_tokens'])
            ax.plot([c['actual_total_tokens'] for c in cells],[c['peak_allocated_gb'] for c in cells],color=color,marker=marker,linestyle=style,label=label,linewidth=1.8)
        ax.set_title(title,fontweight='bold');ax.set_xlabel('Actual model sequence length (tokens)');ax.grid(axis='y',alpha=.18);ax.set_xlim(100,960)
    axes[0].set_ylabel('Peak allocated GPU memory (GB)');axes[0].legend(frameon=False)
    fig.text(.5,.025,'Peak allocated memory includes model weights; GB = 10^9 bytes. Reserved memory is retained separately in the CSV.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(a.output/'exp1_short_b1_memory.png');fig.savefig(a.output/'exp1_short_b1_memory.svg',metadata={'Date':None});plt.close(fig)
    methods=[*core,
      ('ifr_all_positions','IFR all positions','#8c6bb1','D','-'),
      ('attnlrp','AttnLRP','#8b6d46','v','-'),
      ('IG','IG (20 steps)','#dd8452','P','-'),
      ('attention_I_G','Attention × IG','#c56da8','X','--'),
      ('perturbation_all','Perturbation (log loss)','#717875','o',':'),
      ('perturbation_CLP','CLP (KL)','#8a983d','s',':'),
      ('perturbation_REAGENT','REAGENT (MLM)','#567eae','^',':')]
    fig,axes=plt.subplots(1,2,figsize=(11.8,6.2),sharey=True)
    for ax,(family,title) in zip(axes,families):
        for method,label,color,marker,style in methods:
            cells=sorted((c for c in d['cells'] if c['family']==family and c['method']==method and c['phase']=='original3'),key=lambda x:x['actual_total_tokens'])
            # NaN gaps preserve unavailable/OOM cells; never interpolate them.
            ax.plot([c['actual_total_tokens'] for c in cells],[c['seconds_mean'] if c['valid_timing'] else float('nan') for c in cells],
                color=color,marker=marker,linestyle=style,label=label,linewidth=1.5,markersize=5)
        ax.set_title(title,fontweight='bold');ax.set_xlabel('Actual model sequence length (tokens)')
        ax.set_yscale('log');ax.yaxis.set_minor_formatter(NullFormatter());ax.grid(axis='y',alpha=.18,which='both');ax.set_xlim(100,960)
    axes[0].set_ylabel('Original 3-call mean latency (seconds, log scale)')
    fig.suptitle('Author exp1: completed cells before the efficiency-only redirect',fontweight='bold',y=.99)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.035),fontsize=8.5)
    fig.text(.5,.01,'First-call compilation is included. Missing points: OOM, user-cancelled or unstarted cells; see the report.',ha='center',fontsize=8.5,color='#555555')
    fig.tight_layout(rect=(0,.19,1,.94))
    fig.savefig(a.output/'exp1_short_b1_all_methods.png');fig.savefig(a.output/'exp1_short_b1_all_methods.svg',metadata={'Date':None});plt.close(fig)
    for name in ['exp1_short_b1_warm.svg','exp1_short_b1_memory.svg','exp1_short_b1_all_methods.svg']:
        svg=a.output/name
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
