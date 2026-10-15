"""Read the saved upstream FA branches; original boundary helpers own Q/K norms."""
import json,os,sys
from pathlib import Path
import torch

torch.set_num_threads(2)
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
launch=json.loads((root/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3/launch.json').read_bytes())
env=json.loads(Path(launch['effective_environment']['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
dt=Path(launch['effective_environment']['DT_ROOT'])
sys.path[:0]=[str(dt),launch['effective_environment'].get('DT_OFFICIAL_ROOT') or env['official_root'],str(dt/'clean/qwen35')]
from qwen35_decoder_finite import _partial_rotation_transpose,rmsnorm_secant_pullback

dest=Path(sys.argv[1])
for rank in [0,1]:
    data=torch.load(dest/f'native-transport-v6-rank{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
    p=data['metadata']['layer31_payload'];a=p['attention_input_arguments'];pay=data['payload'];row=data['focus_row']
    dq=a['dq'][None].float();dk=a['dk'][None].float();dv=a['dv'][None].float()
    b,h,t,dim=dq.shape;groups=p['groups'];kh=h//groups
    cos=a['cos'][None];sin=a['sin'][None]
    mq=_partial_rotation_transpose(dq,cos,sin).transpose(1,2)
    mk=_partial_rotation_transpose(dk.reshape(b,kh,groups,t,dim).sum(2),cos,sin).transpose(1,2)
    mq=rmsnorm_secant_pullback(a['q0'][None].float(),a['q1'][None].float(),1+p['q_norm_weight'].float(),mq,p['eps'])
    mk=rmsnorm_secant_pullback(a['k0'][None].float(),a['k1'][None].float(),1+p['k_norm_weight'].float(),mk,p['eps'])
    mv=dv.reshape(b,kh,groups,t,dim).sum(2).transpose(1,2)
    mgate=a['mgate'][None]
    qproj=p['q_projection_pair'].reshape(2,t,h,2*dim)
    delta_gate=(qproj[1,:,:,dim:].float()-qproj[0,:,:,dim:].float())[None]
    delta_v=(p['v_pair'][1].float()-p['v_pair'][0].float()).transpose(0,1)[None]
    terms={'q':(mq.double()*(a['q1'].double()-a['q0'].double())[None]).sum((-1,-2))[0],
           'k':(mk.double()*(a['k1'].double()-a['k0'].double())[None]).sum((-1,-2))[0],
           'v':(mv*delta_v).double().sum((-1,-2))[0],
           'gate':(mgate*delta_gate).double().sum((-1,-2))[0]}
    # Actual native input coefficient, before input-norm pullback.
    x=p['native_input_norm_pair'];delta=x[1].double()-x[0].double()
    native=(a['result'].double()*delta).sum(-1)
    result=dict(rank=rank,checkpoint=25,incident_exact_replay=False,pv_rule=p['pv_rule'],
        positions=p['positions'],projection_space_effects={k:v.tolist() for k,v in terms.items()},
        observed_attention_input_effect=native.tolist(),
        original_layer31_input_effect=[float(pay[3]['token_effects'][row,pay[3]['focus_local_index']+i]) for i in [0,1]],
        upstream_at_positions_norm=p['upstream_at_positions'].double().norm(dim=-1).tolist(),
        new_model_calls=0,new_finite_FA_calls=0,tolerance=None)
    if 'native_FA_operands' in p:
        ops=p['native_FA_operands'];out=p['native_FA_outputs'];start=p['native_row_start']
        head_terms=[]
        for position in p['positions']:
            qi=position-start
            delta_k=ops['k1'][0,:,position].float()-ops['k0'][0,:,position].float()
            k_effect=(out['dk'][0,:,qi].float()*delta_k.repeat_interleave(groups,dim=0)).double().sum(-1)
            v_effect=(out['dv'][0,:,qi].float()*delta_v[0,position-p['positions'][0]].repeat_interleave(groups,dim=0)).double().sum(-1)
            head_terms.append(dict(position=position,k_effect_per_head=k_effect.tolist(),v_effect_per_head=v_effect.tolist()))
        result['native_FA_head_terms']=head_terms
        u=ops['u'][0];query_norm=u.double().norm(dim=(0,2));indices=query_norm.topk(12).indices
        result['largest_existing_query_seeds']=[dict(query_position=int(i)+start,norm=float(query_norm[i]),
            tau=out['tau'][0,:,i].tolist(),center=out['center'][0,:,i].tolist()) for i in indices]
        result['query_support_count']=int(query_norm.count_nonzero())
        result['native_FA_shapes']={k:list(v.shape) for k,v in ops.items()}
    (dest/f'layer31-components-rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['largest_existing_query_seeds','native_FA_shapes']}))
