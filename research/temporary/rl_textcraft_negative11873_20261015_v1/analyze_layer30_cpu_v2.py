"""Summarize observed layer30 branches; no new model or finite rule calls."""
import json,sys
from pathlib import Path
import torch

torch.set_num_threads(2)
dest=Path(sys.argv[1])
def dot(a,b):return float((a.double()*b.double()).sum())
def vector(a):return dict(norm=float(a.double().norm()),absmax=float(a.abs().max()))
for rank in [0,1]:
    data=torch.load(dest/f'native-transport-v2r1-rank{rank}.pt',map_location='cpu',weights_only=True)
    p=data['metadata']['layer30_payload'];row=data['focus_row'];pay=data['payload']
    native_delta=p['native_input_norm_pair'][1].double()-p['native_input_norm_pair'][0].double()
    branches=[]
    for branch in p['linear_branches'][1:]:
        branches.append(dict(name=branch['name'],effect_at_native_norm_input=dot(branch['result'],native_delta),
            coefficient=vector(branch['result'])))
    qkv=next(x for x in p['linear_branches'] if x['name']=='in_proj_qkv')['upstream']
    projected=p['projected_qkv_pair'][1].double()-p['projected_qkv_pair'][0].double()
    key_dim=p['num_k_heads']*p['head_k_dim'];value_dim=p['num_v_heads']*p['head_v_dim']
    assert qkv.numel()==projected.numel()==2*key_dim+value_dim
    qkv_parts={name:dot(x,y) for name,x,y in zip(['q','k','v'],qkv.split([key_dim,key_dim,value_dim]),projected.split([key_dim,key_dim,value_dim]))}
    norm_steps=[]
    for i,b in enumerate(p['norm_boundaries']):
        dx=b['input1'].double()-b['input0'].double()
        norm_steps.append(dict(name='post_attention_norm_and_residual' if i==0 else 'input_norm_and_residual',
            combined=dot(b['result'],dx),residual=dot(b['residual'],dx),
            finite_norm_part=dot(b['result']-b['residual'],dx)))
    groups=[]
    for group in p['fla_groups']:
        exponent=group['seed_exponent']
        effects={}
        for name in ['q','k','v','beta','g']:
            coeff=group['coefficients_focus'][name].double()
            if exponent is not None:
                power=exponent.reshape(-1).double()
                coeff=coeff*torch.exp2(power[:,None] if coeff.ndim==2 else power)
            endpoints=group['endpoints_focus']['raw_g' if name=='g' else name].double()
            delta=endpoints[1]-endpoints[0]
            product=coeff*delta
            effects[name]=product.sum(-1).tolist() if product.ndim==2 else product.tolist()
        entry=dict(head_start=group['head_start'],head_count=group['head_count'],effects_per_head=effects)
        if 'native_endpoints' in group:
            ep=group['native_endpoints'];cs=group['native_coefficients'];k=group['focus_index']
            positions=[]
            for t in range(max(0,k-2),min(k+p['conv_width']+2,ep['q'].shape[1])):
                effects={}
                for name in ['q','k','v','beta','g']:
                    coeff=cs[name][0,t].double()
                    if exponent is not None:
                        power=exponent.reshape(-1).double()
                        coeff=coeff*torch.exp2(power[:,None] if coeff.ndim==2 else power)
                    val=ep['raw_g' if name=='g' else name][:,t].double()
                    z=coeff*(val[1]-val[0])
                    effects[name]=z.sum(-1).tolist() if z.ndim==2 else z.tolist()
                positions.append(dict(original_position=t+group['original_time_start'],effects_per_head=effects,
                    native_do_norm_per_head=group['native_do'][0,t].double().norm(dim=-1).tolist()))
            entry['conv_window_positions']=positions
            entry['saved_native_shapes']={k:list(v.shape) for k,v in ep.items()}
        groups.append(entry)
    layer_records=[(s,t) for s,t in zip(data['metadata']['records'],pay) if s['current_layer_index']==30]
    layer_effects=[float(t['token_effects'][row,t['focus_local_index']]) for s,t in layer_records]
    result=dict(rank=rank,checkpoint=25,incident_exact_replay=False,
        final_DT=float(pay[-1]['token_effects'][row,pay[-1]['focus_local_index']]),
        layer30_root_replay_input=layer_effects,
        projection_branches=branches,qkv_preprojection_components=qkv_parts,norm_steps=norm_steps,
        fla_groups=groups,
        warnings=['Component localization is not an independent root-cause proof.',
            'No single-deletion estimand or arbitrary numerical tolerance introduced.'],
        new_model_or_rule_calls=0)
    output=dest/f'layer30-components-rank{rank}.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fla_groups'}))
