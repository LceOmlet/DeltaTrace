"""Account completed native v47 updates and fixed-seed feedback from saved files."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import torch

torch.set_num_threads(4)
OUT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
DEST=OUT/'native-common-update-feedback-v47'
state=json.loads((DEST/'result.json').read_bytes())
assert state['phase'].startswith('complete_') and len(state['observations'])==2
report=dict(unix=time.time(),scope=__doc__,cases={},arms={},model_calls=0,new_backward_calls=0,
    optimizer_calls=0,source=str(DEST/'result.json'),source_SHA256=hashlib.sha256((DEST/'result.json').read_bytes()).hexdigest(),
    limits=['This is conditional on the original old checkpoint80, saved iteration81 inputs and optimizer history; not a complete85-iteration counterfactual or a from-scratch training proof.',
        'Residual projections use norms of native BF16 vectors, not stored exact dot products. Native residual rounding remains.',
        'Before-probe VJP variation is measured and retained despite identical parameter digests; no bitwise reproducibility or invented tolerance claim.',
        'The two native update arms include original clipping, adaptive optimizer state, entropy and KL. Their difference cannot be treated as a linear sum of optimizer component updates.'])
all_displacements={}
for observation in state['observations']:
    name=observation['arm'];before=observation['before'];after=observation['after']
    assert all(x['scheduler']==80 and x['optimizer_steps']==[320.] for x in before)
    assert all(x['scheduler']==81 and x['optimizer_steps']==[324.] for x in after)
    by_side=defaultdict(float);by_layer=defaultdict(float);deltas={}
    for b,a in zip(sorted(before,key=lambda x:x['rank']),sorted(after,key=lambda x:x['rank'])):
        assert b['rank']==a['rank']
        tensors=[]
        for item in (b,a):
            path=Path(item['trainable_snapshot']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['snapshot_SHA256']
            tensors.append(torch.load(path,map_location='cpu',weights_only=False))
        x,y=tensors;assert x.keys()==y.keys()
        for key,v in x.items():
            assert v.shape==y[key].shape
            d=y[key].double()-v.double();deltas[(b['rank'],key)]=d
            side='A' if 'lora_A' in key else 'B' if 'lora_B' in key else 'other'
            layer=key.split('.layers.',1)[1].split('.',1)[0] if '.layers.' in key else 'other'
            ss=float(d.square().sum());by_side[side]+=ss;by_layer[layer]+=ss
    all_displacements[name]=deltas
    native_gradients=observation['original_native_update_metrics']['actor/grad_norm']
    report['arms'][name]=dict(native_grad_norms=native_gradients,native_grad_norm_mean=statistics.mean(native_gradients),
        native_actual_net_parameter_displacement_norm_by_side={k:math.sqrt(v) for k,v in by_side.items()},
        native_actual_net_parameter_displacement_norm_by_layer={k:math.sqrt(v) for k,v in by_layer.items()},
        before_parameter_digests=[v['trainable_SHA256'] for v in sorted(before,key=lambda x:x['rank'])],
        after_parameter_digests=[v['trainable_SHA256'] for v in sorted(after,key=lambda x:x['rank'])],
        fixed_seed_parameter_VJP_before=observation['before_probe']['fixed_seed_gradient_statistics']['norms']['native_target'],
        fixed_seed_parameter_VJP_after=observation['after_probe']['fixed_seed_gradient_statistics']['norms']['native_target'])
    a=report['arms'][name]
    a['fixed_seed_parameter_VJP_ratio']=a['fixed_seed_parameter_VJP_after']/a['fixed_seed_parameter_VJP_before']
    for suffix in ('before','after'):
        case=name+'_'+suffix;sums=defaultdict(float);seed_checks=[]
        for rank in (0,1):
            r=json.loads((DEST/f'iteration{case}-rank{rank}-chain.json').read_bytes())
            assert r['trainable_SHA256_before']==r['trainable_SHA256_after']
            seeds={v['micro']:v for v in r['diagnostic_boundary_seeds']}
            boundaries={v['micro']:v for v in r['records'] if v['boundary'].endswith('.layers.31')}
            assert len(seeds)==len(boundaries)==8
            for micro,b in boundaries.items():
                seed_checks.append(b['output_adjoint_norm']-seeds[micro]['applied_norm'])
                sums['incoming']+=b['output_adjoint_norm']**2
                sums['block_input']+=b['input_adjoint_norm']**2
                for module,side,label in [('post_attention_layernorm','input','MLP_branch'),('mlp','output','MLP_seed'),('self_attn','output','post_MLP_add')]:
                    events=[e for e in r['native_decoder_branch_backward'] if e['micro']==micro and e['name'].endswith('.layers.31.'+module)]
                    assert len(events)==1
                    row=events[0]['native_module_grad_'+side]
                    assert len(row)==1 and row[0]['nonfinite_positions']==0
                    sums[label]+=row[0]['norm']**2
        g2=sums['incoming'];m2=sums['MLP_branch'];mid2=sums['post_MLP_add']
        report['cases'][case]=dict(squared_norms=sums,seed_hook_difference_max=max(abs(x) for x in seed_checks),
            MLP_signed_projection_on_fixed_incoming=(mid2-g2-m2)/(2*g2),
            MLP_branch_norm_over_incoming=math.sqrt(m2/g2),MLP_post_add_norm_gain=math.sqrt(mid2/g2),
            block_input_norm_gain=math.sqrt(sums['block_input']/g2))
full,ablated=(report['arms'][k] for k in ('full_original','without_constant_positive_component'))
assert full['before_parameter_digests']==ablated['before_parameter_digests']
report['identical_before_parameter_digests']=True
report['before_VJP_relative_difference']=ablated['fixed_seed_parameter_VJP_before']/full['fixed_seed_parameter_VJP_before']-1
dot=left=right=0.
for key,x in all_displacements['full_original'].items():
    y=all_displacements['without_constant_positive_component'][key]
    dot+=float((x*y).sum());left+=float(x.square().sum());right+=float(y.square().sum())
report['native_four_step_net_displacement_cosine_between_arms']=dot/math.sqrt(left*right)
report['native_four_step_net_displacement_norms']=[math.sqrt(left),math.sqrt(right)]
report['observed_local_feedback_signs']=dict(full_growth=full['fixed_seed_parameter_VJP_ratio']-1,
    ablated_growth=ablated['fixed_seed_parameter_VJP_ratio']-1)
report['historical_fixed_seed_40_to_80_geometric_mean_gain_per_iteration']=(1.1039044780780574/0.08634081593)**(1/40)
report['historical_rate_limit']='A descriptive magnitude comparison only, not extrapolation of the local gain or evidence of a global exponential law.'
assert not torch.cuda.is_initialized()
dest=DEST/'completed-feedback-accounting-v57.json';assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
