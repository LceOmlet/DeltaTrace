"""Pointwise mathematical convolution check using actual saved operands.

This computes no replacement convolution or backward for training. The original
native VJP is already saved, and the subsequent independent forward check uses
the installed public convolution and FLA functions.
"""
import json,sys
from pathlib import Path
import torch

torch.set_num_threads(2)
d=Path(sys.argv[1])
for rank in [0,1]:
    data=torch.load(d/f'native-transport-v3-rank{rank}.pt',map_location='cpu',weights_only=True)
    p=data['metadata']['layer30_payload'];c=p['conv_silu']
    dx=p['projected_qkv_pair'][1].double()-p['projected_qkv_pair'][0].double()
    # Native causal-conv reference uses cross-correlation with rightmost tap
    # at the current token: x_i reaches y_(i+lag) through weight[width-1-lag].
    w=c['weight'].double().flip(-1)
    native_seed=c['result'].to(c['pre'].dtype).double()
    mathematical=dx[:,None]*w*c['result'].double()
    seed_cast=dx[:,None]*w*native_seed
    saved=next(x for x in p['linear_branches'] if x['name']=='in_proj_qkv')['upstream'].double()
    kd=p['num_k_heads']*p['head_k_dim'];vd=p['num_v_heads']*p['head_v_dim']
    result=dict(rank=rank,checkpoint=25,incident_exact_replay=False,
        source_position=1870,conv_width=p['conv_width'],pre_dtype=str(c['pre'].dtype),
        per_qkv_component={},observed_conv_pre_collisions=int(((c['pre'][0]==c['pre'][1])&(c['output'][0]!=c['output'][1])).sum()))
    for name,start,length in [('q',0,kd),('k',kd,kd),('v',2*kd,vd)]:
        s=slice(start,start+length)
        result['per_qkv_component'][name]=dict(
            observed_native_VJP_dot=float((saved[s]*dx[s]).sum()),
            predicted_per_lag_before_seed_cast=mathematical[s].sum(0).tolist(),
            predicted_per_lag_after_original_seed_cast=seed_cast[s].sum(0).tolist(),
            predicted_total_after_seed_cast=float(seed_cast[s].sum()))
    v=seed_cast[2*kd:].reshape(p['num_v_heads'],p['head_v_dim'],p['conv_width']).sum(1)
    result['value_heads_per_lag']=v.tolist()
    top=[]
    pre=c['pre'].float();out=c['output'].float();dp=pre[1]-pre[0];dy=out[1]-out[0]
    for _,flat in sorted([(float(z),int(i)) for i,z in enumerate(seed_cast[2*kd:].abs().flatten())],reverse=True)[:12]:
        channel=flat//p['conv_width']+2*kd;lag=flat%p['conv_width']
        top.append(dict(value_channel=channel-2*kd,head=(channel-2*kd)//p['head_v_dim'],lag=lag,
            contribution=float(seed_cast[channel,lag]),projected_source_delta=float(dx[channel]),
            tap=float(w[channel,lag]),pre0=float(pre[0,channel,lag]),pre1=float(pre[1,channel,lag]),
            output0=float(out[0,channel,lag]),output1=float(out[1,channel,lag]),
            secant=None if dp[channel,lag]==0 else float(dy[channel,lag]/dp[channel,lag]),
            incoming_value_coefficient=float(c['upstream'][channel,lag])))
    result['largest_value_channel_terms']=top
    (d/f'conv-point-prediction-rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['value_heads_per_lag','largest_value_channel_terms']}))
