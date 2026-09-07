"""Standalone scientific figure; observed budgets, no interpolated quality claims."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'plot_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
A=Path(__file__).resolve().parent;s=json.loads((A/'finite_FA_FT_cost16_summary_20260907.json').read_text())
assert s['status']=='verified_complete'
fig,axs=plt.subplots(1,3,figsize=(13.5,4.5),layout='constrained')
panels=[('niah_mq_q2','recovery','NIAH needle recovery (higher is better)'),('morehopqa','rise','MoreHopQA RISE (lower is better)'),('morehopqa','mas','MoreHopQA MAS (lower is better)')]
points=[]
for ax,(ds,metric,title) in zip(axs,panels):
    group=s['dataset_summary'][ds]['methods'];reference=group['both_1']['mean_seconds']
    for name,row in group.items():
        quality=row['quality'][metric]
        if quality['count']!=8 or quality['mean'] is None:continue
        x=row['mean_seconds']/reference;y=quality['mean']
        color='#c4423e' if name=='finite' else ('#176b9c' if name.startswith('both') else '#777d86')
        marker='*' if name=='finite' else ('o' if name.startswith('both') else '^')
        ax.scatter(x,y,color=color,marker=marker,s=155 if name=='finite' else 48,zorder=4)
        label='P1' if name=='finite' else ('B' if name.startswith('both') else 'L')+name[-1]
        ax.annotate(label,(x,y),xytext=(5,5 if name!='both_0' else -12),textcoords='offset points',fontsize=9,color=color)
        points.append({'dataset':ds,'metric':metric,'method':name,'normalized_mean_seconds':x,'quality':y})
    ax.axvline(1,color='#cccccc',ls='--',lw=1)
    ax.set_title(title,fontsize=11,pad=12);ax.set_xlabel('Mean latency / FT full-entry both-1')
    ax.grid(alpha=.2);ax.margins(x=.14,y=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Original development 16: fresh full costs + exact-vector verified historical quality',fontsize=13)
fig.supxlabel('B0–B3: original FlashTrace both family   |   L0–L3: original legacy family   |   P1: finite FA propagation\nBoth-family full entry includes unused row/rec views: see the seq-only fairness correction.\nSingle-example latency; warmup excluded. Independent confirmation remains incomplete.',fontsize=9)
folder=A/'figures';folder.mkdir(exist_ok=True)
for suffix in ['png','svg']:fig.savefig(folder/('finite_FA_FT_cost16_quality_cost_20260907.'+suffix),dpi=180)
svg=folder/'finite_FA_FT_cost16_quality_cost_20260907.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
plt.close(fig)
(folder/'finite_FA_FT_cost16_figure_manifest_20260907.json').write_text(json.dumps({'summary_sha256':hashlib.sha256((A/'finite_FA_FT_cost16_summary_20260907.json').read_bytes()).hexdigest(),'points':points},indent=2))
print('Generated observed-quality / fresh-cost figure from verified complete results.')
