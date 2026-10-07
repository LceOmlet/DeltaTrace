"""Plot saved original-author arrays and raw native scores; do not smooth."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'analysis.json').read_bytes())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes = plt.subplots(1,3,figsize=(13.5,3.8),layout='constrained')
colors = {'signed_RISE':'#1b6ca8','positive_MAS':'#e07a1f'}
for name,view in data['views'].items():
    steps = [g['step'] for g in view['groups']]
    label = 'Signed ranking' if name == 'signed_RISE' else 'Positive-only evaluation ranking'
    axes[0].plot(steps,[g['native_logp'] for g in view['groups']],marker='o',markersize=3,lw=1.6,color=colors[name],label=label)
    ax = axes[1 if name == 'signed_RISE' else 2]
    ax.plot(steps,view['author_arrays']['normalized_model_response'],marker='o',markersize=3,lw=1.5,label='Author normalized response')
    ax.plot(steps,view['author_arrays']['density'],ls='--',lw=1.5,label='Author attribution density')
    ax.axhline(0,color='#777777',lw=.6)
    ax.set_title('Signed input: RISE %.4f' % view['author_return']['rise'] if name == 'signed_RISE' else 'Positive-only input: MAS %.4f' % view['author_return']['mas'])
    ax.set_ylabel('Original author arrays')
    ax.legend(fontsize=8,frameon=False,loc='upper right')
axes[0].set_title('Raw joint action target score')
axes[0].set_ylabel('Native joint log-probability')
axes[0].legend(fontsize=8,frameon=False,loc='upper right')
axes[0].annotate('Final negative-ranked group:\nlog-prob drops 15.73',xy=(20,-153.912),xytext=(8,-85),
    fontsize=9,arrowprops=dict(arrowstyle='->',color='#555555'))
for ax in axes:
    ax.set_xlabel('Author cumulative deletion step (k = 20)')
    ax.set_xticks([0,5,10,15,20]);ax.grid(axis='y',alpha=.15)
fig.suptitle('One real AppWorld trajectory | 1,266 prior-source tokens | same native scores on both ranks',fontsize=12)
fig.savefig(HERE/'action-deletion-curve.png',dpi=170)
plt.close(fig)
print(HERE/'action-deletion-curve.png')
