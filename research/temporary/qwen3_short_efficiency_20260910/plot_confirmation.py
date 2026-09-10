"""Standalone scientific figure from the verified complete-call timing summary."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(study):
    data=json.loads((study/'confirmation_v4_summary.json').read_bytes());assert data['acceptance_passed']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#172033','xtick.color':'#475569','ytick.color':'#475569'})
    methods=['deltatrace_graphed','ifr_multi_hop_both','ifr_multi_hop']
    labels={'deltatrace_graphed':'DeltaTrace (native graph)','ifr_multi_hop_both':'FT Both','ifr_multi_hop':'FT multi-hop'}
    colors={'deltatrace_graphed':'#087f8c','ifr_multi_hop_both':'#b86c25','ifr_multi_hop':'#66748b'}
    sizes=[128,256,512,1024];xs=np.arange(4);fig,(left,right)=plt.subplots(1,2,figsize=(12.4,4.9),gridspec_kw={'width_ratios':[1.1,1]})
    fig.suptitle('Qwen3-8B: complete warm attribution',x=.065,y=.975,ha='left',fontsize=17,fontweight='bold')
    fig.text(.065,.908,'B1 / two DT endpoints · fixed 32-token output + EOS · MetaX C550 · FP16',fontsize=10.5,color='#526174',va='top')
    for mi,method in enumerate(methods):
        cells=[next(c for c in data['pooled_cells'] if c['method']==method and c['target_input_tokens']==n) for n in sizes]
        means=np.asarray([c['mean_seconds']*1000 for c in cells]);left.plot(xs,means,'o-',color=colors[method],label=labels[method],lw=1.8,ms=5,zorder=3)
        for x,c in zip(xs,cells):
            left.scatter(x+np.linspace(-.05,.05,6),np.asarray(c['times_seconds'])*1000,s=16,color=colors[method],alpha=.48,zorder=2,edgecolors='none')
    left.set_title('Full-call latency',loc='left',pad=13);left.set_ylabel('Milliseconds');left.set_ylim(0,635)
    left.set_xticks(xs,['128\n158 actual','256\n265 actual','512\n478 actual','1024\n905 actual'])
    left.set_xlabel('Requested input tokens / complete sequence',labelpad=9)
    left.grid(axis='y',color='#e2e8f0',lw=.8);left.set_axisbelow(True);left.legend(loc='upper left',frameon=False,fontsize=9)
    for mi,method in enumerate(methods[1:]):
        points=[next(c for c in data['comparisons'] if c['FT_method']==method and c['target_input_tokens']==n) for n in sizes]
        ratios=np.asarray([c['DT_to_FT_mean_ratio'] for c in points]);bounds=np.asarray([c['bootstrap_ratio_95_percentile_interval'] for c in points])
        right.errorbar(xs+(-.085 if mi==0 else .085),ratios,yerr=[ratios-bounds[:,0],bounds[:,1]-ratios],fmt='o',capsize=4,lw=1.6,ms=5,color=colors[method],label='DT / '+labels[method])
    right.axhline(1,color='#7c8798',lw=1.2,linestyle='--');right.text(3.25,1.009,'Equal latency',ha='right',fontsize=9,color='#667085')
    right.set_title('DT / FT mean latency ratio',loc='left',pad=13);right.set_ylabel('Lower means faster DT')
    right.set_xticks(xs,[str(x) for x in sizes]);right.set_xlabel('Requested input tokens',labelpad=9);right.set_xlim(-.45,3.4);right.set_ylim(.62,1.04)
    right.grid(axis='y',color='#e2e8f0',lw=.8);right.set_axisbelow(True);right.legend(loc='lower right',frameon=False,fontsize=9)
    fig.text(.065,.037,'All six measured repeats per method and length retained. Error bars: prespecified 95% bootstrap intervals (20,000 resamples).',fontsize=9,color='#526174')
    fig.subplots_adjust(left=.065,right=.978,bottom=.24,top=.81,wspace=.29)
    output=study/'warm_latency_and_ratio.png';fig.savefig(output,dpi=180,facecolor='white',metadata={'Software':'Matplotlib '+matplotlib.__version__});plt.close(fig)
    (study/'figure_environment.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'numpy':np.__version__,'input':'confirmation_v4_summary.json','output':output.name},indent=2)+'\n',newline='\n')
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();plot(a.study)
