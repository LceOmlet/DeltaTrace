"""Plot original stopped training records; missing data remain missing."""
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(__file__).with_name('textcraft-degradation-20261005')
records=json.loads((folder/'records.json').read_text(encoding='utf-8'))
xs,success,entropy,ppo_kl,nonfinite=[],[],[],[],[]
for row in records['metrics']:
    text=row['text'];xs.append(row['step'])
    def value(key):
        match=re.search(re.escape(key)+r':([^ ]+)',text)
        return float(match.group(1)) if match else float('nan')
    success.append(value('episode/reward/mean')*100)
    entropy.append(value('actor/entropy_loss'))
    ppo_kl.append(value('actor/ppo_kl'))
    if 'actor/grad_norm:nan' in text: nonfinite.append(row['step'])
fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
fig.subplots_adjust(left=.095,right=.985,bottom=.145,top=.945,hspace=.13)
for ax,y,label,color in zip(axes,[success,entropy,ppo_kl],
        ['Training success (%)','Logged old-policy entropy','Logged PPO KL'],
        ['#16795c','#b45e15','#5268a7']):
    ax.plot(xs,y,lw=1.6,color=color);ax.set_ylabel(label);ax.grid(alpha=.18)
    ax.axvspan(35,40,color='#777777',alpha=.12)
    for step in nonfinite: ax.axvline(step,color='#bd2449',alpha=.35,linestyle='--')
axes[0].set_title('TextCraft: original training records, iterations 26–117')
axes[-1].set_xlabel('Completed iteration (4 native optimizer attempts per iteration)')
fig.text(.095,.021,'Gray: earliest decline window to inspect. Red: nonfinite gradient warnings (83, 85, 109).\n'
    'Console values rounded to 3 decimals. Entropy uses the deployed policy loss_mask; PPO KL is logged actor aggregation.',
    fontsize=8,color='#555555')
fig.savefig(folder/'degradation.png',dpi=170)
print(folder/'degradation.png')
