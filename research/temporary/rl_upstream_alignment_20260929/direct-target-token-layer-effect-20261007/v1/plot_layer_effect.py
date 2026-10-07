"""Plot original joint coefficient/single-hidden-difference contractions."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path,default=HERE/'analysis.json');parser.add_argument('--output',type=Path,default=HERE/'layer-effects.png');args=parser.parse_args()
d=json.loads(args.analysis.read_bytes());r=d['ranks'][0]
rows=r['boundaries'];x=[v['boundary'] for v in rows];y=[v['joint_coefficient_times_single_deletion_delta'] for v in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9.2,4.1),layout='constrained')
ax.plot(x,y,'o-',markersize=3,lw=1.6,color='#1b6ca8',label='Original joint coefficient × actual single-deletion hidden difference')
ax.axhline(r['single_source_focused_root'],color='#888888',ls='--',lw=1.1,label='Native cached single-deletion target difference (%.3f)'%r['single_source_focused_root'])
ax.axhline(0,color='#333333',lw=.8)
negative=r['first_negative_during_reverse_traversal']
if negative is not None:
    i=next(i for i,v in enumerate(rows) if v['boundary']==negative['boundary'])
    before=rows[i-1]['joint_coefficient_times_single_deletion_delta']
    ax.annotate('Decoder %d: %+.3f to %+.3f'%(negative['boundary'],before,negative['joint_coefficient_times_single_deletion_delta']),xy=(negative['boundary'],negative['joint_coefficient_times_single_deletion_delta']),xytext=(.35,.8),textcoords='axes fraction',fontsize=9,arrowprops=dict(arrowstyle='->',color='#777777'))
ax.annotate('Input: %+.3f\n= original joint token credit'%y[-1],xy=(0,y[-1]),xytext=(.65,.5),textcoords='axes fraction',fontsize=9,arrowprops=dict(arrowstyle='->',color='#777777'))
ax.invert_xaxis();ax.set_xticks([32,28,24,20,16,12,8,4,0]);ax.grid(axis='y',alpha=.15)
ax.set_xlabel('Reverse propagation boundary (32 = output, 0 = input)')
ax.set_ylabel('Diagnostic contraction (not a new advantage)')
ax.set_title('Actual AppWorld token | original B4 | factual endpoint equality: %s'%r['factual_endpoint_all_equal'])
ax.legend(fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.01))
fig.savefig(args.output,dpi=170);plt.close(fig)
print(args.output.resolve())
