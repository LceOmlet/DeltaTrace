"""Read native submodule VJP gains on the existing fixed-input/seed cases."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
paths=[HERE/'native-fixed-seed-propagation-v33-status-final.json',HERE/'native-midpoint-fixed-seed-v35-status-final.json']
wanted={5:'late5_seed80',40:'late40_seed80',80:'late80_native'}
raw=[]
for path in paths:
    raw += [r['value'] for r in json.loads(path.read_text(encoding='utf-8'))['report']['chain_reports']
            if r['value']['iteration']==wanted[r['value']['checkpoint']]]
report=dict(scope=__doc__,source_snapshots={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},cases={},
    new_model_calls=0,new_backward_calls=0,optimizer_updates=0,
    limits=['These are actual directional input/output adjoint-norm ratios, not operator norms.',
            'Forward quantiles are preserved per existing B4 event; extrema/median ranges here are not a pooled-token distribution.',
            'The exact same incoming tensor is fixed at block31 output only; submodules receive the directions propagated by their actual native graph.',
            'This localizes the propagation change, not which historical optimizer component caused it.'])
for cp in (5,40,80):
    modules=defaultdict(lambda:dict(input_sumsq=0.,output_sumsq=0.,active_input_RMS_quantiles=[],forward_input_RMS_quantiles=[]))
    for r in [v for v in raw if v['checkpoint']==cp]:
        for event in r['native_decoder_branch_backward']:
            if '.layers.31.' not in event['name']:continue
            name=event['name'].split('.layers.31.')[1];row=modules[name]
            for side in ('input','output'):
                values=[v for v in event['native_module_grad_'+side] if v['index']==0]
                if values:
                    row[side+'_sumsq']+=values[0]['norm']**2
                    q=values[0].get('forward_RMS_at_nonzero_branch_adjoint_quantiles')
                    if side=='input' and q is not None:row['active_input_RMS_quantiles'].append(q)
        for event in r['native_decoder_branch_forward']:
            if '.layers.31.' not in event['name']:continue
            q=event.get('input_RMS_quantiles_all')
            if q is not None:modules[event['name'].split('.layers.31.')[1]]['forward_input_RMS_quantiles'].append(q)
    for row in modules.values():
        row.update(input_adjoint_norm=math.sqrt(row['input_sumsq'])/2,output_adjoint_norm=math.sqrt(row['output_sumsq'])/2)
        row['directional_input_over_output']=math.sqrt(row['input_sumsq']/row['output_sumsq']) if row['input_sumsq'] and row['output_sumsq'] else None
        q=row['active_input_RMS_quantiles']
        row['range_of_active_B4_medians']=None if not q else [min(a[2] for a in q),max(a[2] for a in q)]
        row['minimum_observed_active_RMS']=None if not q else min(a[0] for a in q)
    report['cases'][str(cp)]=dict(modules)
dest=HERE/'saved-native-branch-gain-v46.json'
dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
for name in ('mlp','post_attention_layernorm','input_layernorm','self_attn.q_norm','self_attn.k_norm'):
    print(name,[(cp,report['cases'][str(cp)][name]['directional_input_over_output'],report['cases'][str(cp)][name]['range_of_active_B4_medians']) for cp in (5,40,80)])
print('receipt',dest)
