"""One actual single-deletion finite trace; not an official tolerance plot."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(__file__).with_name('textcraft-degradation-20261005')
rank,=[x for x in json.loads((folder/'model-probe-v4/result.json').read_text(encoding='utf-8')) if x['rank']==1]
observed=rank['passes'][0]
rows=observed['passive_finite_effects']['records']
seed,=[x for x in rows if x['phase']=='seed_effect']
xs=[32];values=[seed['per_sample_sums'][0]]
for layer in range(31,-1,-1):
    row,=[x for x in rows if x['phase']=='layer_input_effect' and x['current_layer_index']==layer]
    xs.append(layer);values.append(row['per_sample_sums'][0])
native=observed['values'][0]['root_effect']
fig,ax=plt.subplots(figsize=(9,4.6))
fig.subplots_adjust(left=.1,right=.97,bottom=.24,top=.85)
ax.plot(xs,values,'o-',ms=3,lw=1.6,color='#475da7',label='Original finite effect at each boundary')
ax.axhline(native,color='#18795b',lw=1.2,label=f'Native single-EOS endpoint effect {native:.5f}')
ax.axhline(0,color='#555555',lw=.7)
ax.axvspan(3.4,2.6,color='#b45e15',alpha=.15)
ax.set_xlim(32.5,-.5);ax.set_xticks([32,28,24,20,16,12,8,4,3,2,1,0])
ax.set_xlabel('Propagation from output seed (32) to decoder input (31 ... 0)')
ax.set_ylabel('Signed event log-prob effect')
ax.set_title('TextCraft checkpoint25: one real Thought-token deletion')
ax.grid(alpha=.18);ax.legend(loc='lower left',fontsize=8)
fig.text(.1,.055,'Rank1 / original UID1744... / B4, pairedB8, prefix320. No optimizer step.\n'
    'Actual coefficients FP32, captured endpoints BF16, contraction FP64. Selected case, not a full-batch error rate.\n'
    'Shaded decoder3 contains attention, projections, MLP and norms; no operator blame or tolerance verdict implied.',fontsize=8,color='#555555')
fig.savefig(folder/'single-effect-propagation.png',dpi=170)
print(folder/'single-effect-propagation.png')
