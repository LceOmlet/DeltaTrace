"""Plot original joint coefficient/single-hidden-difference contractions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'analysis.json').read_bytes());r=d['ranks'][0]
rows=r['boundaries'];x=[v['boundary'] for v in rows];y=[v['joint_coefficient_times_single_deletion_delta'] for v in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9.2,4.1),layout='constrained')
ax.plot(x,y,'o-',markersize=3,lw=1.6,color='#1b6ca8',label='Original joint coefficient × actual single-deletion hidden difference')
ax.axhline(r['single_source_focused_root'],color='#888888',ls='--',lw=1.1,label='Native cached single-deletion target difference (20.719)')
ax.axhline(0,color='#333333',lw=.8)
ax.annotate('After output head + final norm: +7.614',xy=(32,y[0]),xytext=(30,12),fontsize=9,arrowprops=dict(arrowstyle='->',color='#777777'))
ax.annotate('Decoder 27: +0.996 → −1.587',xy=(27,-1.587),xytext=(22,5),fontsize=9,arrowprops=dict(arrowstyle='->',color='#777777'))
ax.annotate('Input: −3.128\n= original joint token credit',xy=(0,y[-1]),xytext=(8,-.5),fontsize=9,arrowprops=dict(arrowstyle='->',color='#777777'))
ax.invert_xaxis();ax.set_xticks([32,28,24,20,16,12,8,4,0]);ax.grid(axis='y',alpha=.15)
ax.set_xlabel('Reverse propagation boundary (32 = output, 0 = input)')
ax.set_ylabel('Diagnostic contraction (not a new advantage)')
ax.set_title('Actual AppWorld newline token | original B4 | factual hidden states exactly equal across both calls')
ax.legend(fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.01))
fig.savefig(HERE/'layer-effects.png',dpi=170);plt.close(fig)
print(HERE/'layer-effects.png')
