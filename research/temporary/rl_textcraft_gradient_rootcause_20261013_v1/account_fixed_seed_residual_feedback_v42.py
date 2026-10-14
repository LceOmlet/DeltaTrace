"""Account signed native residual feedback on the already saved fixed seed.

At layer31, all three checkpoints receive the exact same saved BF16 output
adjoint. Native module grad_input/grad_output distinguish the MLP branch from
the identity path. A norm identity estimates their signed projection; it is
not a directly stored inner product, operator norm, or attribution of updates.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
paths=[HERE/'native-fixed-seed-propagation-v33-status-final.json',
       HERE/'native-midpoint-fixed-seed-v35-status-final.json']
wanted={5:'late5_seed80',40:'late40_seed80',80:'late80_native'}
raw=[]
for p in paths:
    value=json.loads(p.read_text(encoding='utf-8'))['report']
    assert not value['alive']
    raw += [r['value'] for r in value['chain_reports'] if wanted[r['value']['checkpoint']]==r['value']['iteration']]
cases={}
for cp in (5,40,80):
    records=[r for r in raw if r['checkpoint']==cp]
    assert sorted(r['rank'] for r in records)==[0,1]
    norms=defaultdict(float);events=[];checks=[]
    for r in records:
        boundaries={b['micro']:b for b in r['records'] if b['boundary'].endswith('.layers.31')}
        seeds={s['micro']:s for s in r['diagnostic_boundary_seeds']}
        counters=Counter((b['name'],b['micro']) for b in r['native_decoder_branch_backward'])
        assert len(boundaries)==len(seeds)==8
        assert all(n==1 for n in counters.values())
        assert r['trainable_SHA256_before']==r['trainable_SHA256_after']
        for micro,b in sorted(boundaries.items()):
            checks.append(dict(rank=r['rank'],micro=micro,
                native_output_norm_minus_applied_seed=b['output_adjoint_norm']-seeds[micro]['applied_norm']))
            norms['g_out']+=b['output_adjoint_norm']**2
            norms['block_input']+=b['input_adjoint_norm']**2
            for suffix,side,label in [('post_attention_layernorm','input','MLP_branch'),
                                      ('mlp','output','MLP_received_seed'),
                                      ('self_attn','output','middle_total'),
                                      ('input_layernorm','input','attention_branch')]:
                selected=[v for v in r['native_decoder_branch_backward']
                          if v['micro']==micro and v['name'].endswith('.layers.31.'+suffix)]
                assert len(selected)==1
                v=selected[0]['native_module_grad_'+side]
                assert len(v)==1 and v[0]['index']==0 and v[0]['nonfinite_positions']==0
                norms[label]+=v[0]['norm']**2
                if label=='MLP_received_seed':
                    checks[-1]['native_MLP_received_norm_minus_applied_seed']=v[0]['norm']-seeds[micro]['applied_norm']
            events.append(dict(rank=r['rank'],micro=micro))
    norms={k:math.sqrt(v)/2 for k,v in norms.items()}
    g=norms['g_out'];m=norms['MLP_branch'];mid=norms['middle_total']
    a=norms['attention_branch'];total=norms['block_input']
    parallel=(mid*mid-g*g-m*m)/(2*g*g)
    attention_parallel=(total*total-mid*mid-a*a)/(2*mid*mid)
    cases[cp]=dict(checkpoint=cp,norms=norms,events=events,seed_hook_checks=checks,
        MLP_branch_norm_over_incoming=m/g,
        MLP_signed_projection_on_same_incoming=parallel,
        MLP_residual_direction_cosine=parallel/(m/g),
        MLP_post_add_norm_gain=mid/g,
        attention_branch_norm_over_its_incoming=a/mid,
        attention_signed_projection_on_its_incoming=attention_parallel,
        attention_post_add_norm_gain=total/mid,
        block_total_directional_gain=total/g)
old,new=cases[40],cases[80]
report=dict(scope=__doc__,input_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    cases={str(k):v for k,v in cases.items()},
    last_MLP_40_to_80=dict(
        incoming_norm_ratio=new['norms']['g_out']/old['norms']['g_out'],
        signed_feedback_change=new['MLP_signed_projection_on_same_incoming']-old['MLP_signed_projection_on_same_incoming'],
        MLP_branch_size_ratio=new['MLP_branch_norm_over_incoming']/old['MLP_branch_norm_over_incoming'],
        post_add_gain_ratio=new['MLP_post_add_norm_gain']/old['MLP_post_add_norm_gain']),
    model_calls=0,new_backward_calls=0,optimizer_updates=0,rootcause_resolved=False,
    limits=['Projection uses norms of native BF16 vectors. Finite rounding in residual addition and norm readouts remains; no exact-vector inner-product or new tolerance claim.',
            'The incoming tensor is held fixed only at layer31 output. Other layers and the attention sub-branch receive their actual changed propagated directions.',
            'Changing checkpoint changes both weights and induced forward states. This identifies the local feedback change, not the historical update component that caused it.',
            'The same-input/seed parameter VJP evidence is directional and cannot replace the original whole-policy gradient trend or establish universal amplification.'])
dest=HERE/'fixed-seed-residual-feedback-v42.json'
dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('MLP fixed-seed feedback',[(cp,c['MLP_signed_projection_on_same_incoming'],c['MLP_branch_norm_over_incoming'],c['MLP_post_add_norm_gain']) for cp,c in cases.items()])
print('40to80',report['last_MLP_40_to_80'])
print('hook check max',max(abs(v) for c in cases.values() for r in c['seed_hook_checks'] for k,v in r.items() if k.endswith('minus_applied_seed')))
print('receipt',dest)
