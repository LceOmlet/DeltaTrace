"""Plot actual original-author arrays and unsmoothed native joint scores."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'analysis.json').read_bytes())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(13.5,3.8),layout='constrained')
for name,color,column,label in [('signed_RISE','#1b6ca8',1,'Signed ranking'),('positive_MAS','#e07a1f',2,'Positive-only evaluation ranking')]:
    view=data['views'][name]
    steps=[g['step'] for g in view['groups']]
    axes[0].plot(steps,[g['native_joint_logp'] for g in view['groups']],marker='o',markersize=3,color=color,label=label)
    axes[column].plot(steps,view['author_arrays']['normalized_model_response'],marker='o',markersize=3,label='Author normalized response')
    axes[column].plot(steps,view['author_arrays']['density'],ls='--',label='Author attribution density')
    metric='rise' if name=='signed_RISE' else 'mas'
    axes[column].set_title(metric.upper()+' %.4f'%view['author_return'][metric])
    axes[column].set_ylabel('Original author arrays')
axes[0].set_title('Raw joint action target score')
axes[0].set_ylabel('Native joint log-probability')
for ax in axes:
    ax.set_xlabel('Author cumulative deletion step (k = 20)')
    ax.set_xticks([0,5,10,15,20])
    ax.grid(axis='y',alpha=.15)
    ax.legend(fontsize=8,frameon=False)
fig.suptitle('Actual TextCraft Format trajectory | %s source tokens | same native scores on both ranks'%data['source_count'],fontsize=12)
fig.savefig(HERE/'textcraft-action-deletion-curve.png',dpi=170)
plt.close(fig)
print(HERE/'textcraft-action-deletion-curve.png')
