"""Plot original author arrays and native scores without smoothing or correction."""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--memory',action='store_true');args=parser.parse_args()
data = json.loads((HERE/('memory-curve-analysis.json' if args.memory else 'curve-analysis.json')).read_bytes())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(13.2,7.0),layout='constrained')
styles={'original_content1':('#1b6ca8','Original content1'),
        'existing_content0':('#d46b2a','Existing content0')}
if args.memory:styles={'original_symmetric_memory':('#1b6ca8','Original averaged memory'),
                      'existing_forward_memory':('#d46b2a','Existing forward memory')}
for row,name in enumerate(('signed_RISE','positive_MAS')):
    for mode,profile in data['profiles'].items():
        view=profile['views'][name]
        color,label=styles[mode]
        steps=[point['step'] for point in view['groups']]
        arrays=view['author_arrays']
        axes[row,0].plot(steps,[point['native_logp'] for point in view['groups']],color=color,marker='o',markersize=3,label=label)
        axes[row,1].plot(steps,arrays['normalized_model_response'],color=color,marker='o',markersize=3,label=label+' | RISE %.4f'%view['author_return']['rise'])
        axes[row,2].plot(steps,arrays['density'],color=color,ls='--',label=label+' density')
        axes[row,2].plot(steps,arrays['normalized_model_response'],color=color,lw=1.0,label=label+' response')
    axes[row,0].set_title(('Signed ranking' if row==0 else 'Positive-only evaluation ranking')+' | raw score')
    axes[row,0].set_ylabel('Native joint action log-probability')
    axes[row,1].set_title('Original author normalized response')
    axes[row,2].set_title('Original author attribution density')
    for ax in axes[row]:
        ax.set_xlabel('Original cumulative deletion step (k = 20)')
        ax.set_xticks([0,5,10,15,20]);ax.grid(axis='y',alpha=.15)
        ax.legend(fontsize=8,frameon=False)
count=next(iter(data['profiles'].values()))['source_count']
fig.suptitle('One real AppWorld trajectory | %d prior-source tokens | unchanged author metric\nEvaluation only: no credit clipping, rule deployment or optimizer update'%count,fontsize=12)
target=HERE/('existing-memory-author-curves.png' if args.memory else 'existing-pv-author-curves.png')
fig.savefig(target,dpi=170)
plt.close(fig)
print(target.resolve())
