"""Locate completed v47 gain changes in actual saved adjoints/gradient shards."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
source=HERE/'read_native_gradient_chain_v4-dt.json'
data=json.loads(source.read_text())['report']
assert data['state']['phase'].startswith('complete_')
assert len(data['chain_reports'])==8
cases={}
for row in data['chain_reports']:
    r=row['value'];name=r['iteration']
    c=cases.setdefault(name,dict(ranks=[],layers=defaultdict(lambda:defaultdict(float)),parameter_layers=defaultdict(float),parameter_factors=defaultdict(float),final_ss=0.,increment_diag_ss=0.))
    c['ranks'].append(r['rank'])
    for b in r['records']:
        match=re.search(r'\.layers\.(\d+)$',b['boundary'])
        if not match:continue
        layer=int(match[1])
        c['layers'][layer]['input_ss']+=b['input_adjoint_norm']**2
        c['layers'][layer]['output_ss']+=b['output_adjoint_norm']**2
    a=r['native_accumulation_readout']
    c['final_ss']+=a['local_final_preclip_sumsq']
    c['increment_diag_ss']+=sum(a['local_increment_Gram'][i][i] for i in range(len(a['local_increment_Gram'])))
    for p in a['parameters']:
        assert p['placements']==['S(0)']
        m=re.search(r'\.layers\.(\d+)\.(.+)\.lora_([AB])\.',p['name']);assert m,p['name']
        c['parameter_layers'][int(m[1])]+=p['local_preclip_sumsq']
        c['parameter_factors'][m[3]]+=p['local_preclip_sumsq']
for name,c in cases.items():
    assert sorted(c['ranks'])==[0,1]
    c['layers']={str(k):{key[:-3]+'_norm':math.sqrt(value) for key,value in v.items()} for k,v in sorted(c['layers'].items())}
    c['parameter_layers']={str(k):dict(norm=math.sqrt(v),energy_fraction=v/c['final_ss']) for k,v in sorted(c['parameter_layers'].items())}
    c['parameter_factors']={k:math.sqrt(v) for k,v in c['parameter_factors'].items()}
    c['parameter_VJP_norm']=math.sqrt(c['final_ss'])
    c['B4_native_increment_RSS_norm']=math.sqrt(c['increment_diag_ss'])
    c['B4_accumulation_direction_factor']=math.sqrt(c['final_ss']/c['increment_diag_ss'])
arms={}
for name in ('full_original','without_constant_positive_component'):
    b,a=(cases[name+'_'+s] for s in ('before','after'))
    arms[name]=dict(parameter_VJP_ratio=a['parameter_VJP_norm']/b['parameter_VJP_norm'],
        B4_increment_RSS_ratio=a['B4_native_increment_RSS_norm']/b['B4_native_increment_RSS_norm'],
        B4_accumulation_direction_factor_ratio=a['B4_accumulation_direction_factor']/b['B4_accumulation_direction_factor'],
        layer_input_adjoint_norm_ratio={k:a['layers'][k]['input_norm']/v['input_norm'] for k,v in b['layers'].items()},
        layer_parameter_VJP_norm_ratio={k:a['parameter_layers'][k]['norm']/v['norm'] for k,v in b['parameter_layers'].items()})
report=dict(source=str(source),source_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=cases,arms=arms,
    model_calls=0,new_backward_calls=0,optimizer_calls=0,
    limits=['Layer and LoRA partitions are coordinate accounting, not independent causal contributions.',
        'B4 direction factor is for actual accumulated native minibatch gradients; it is not a token-noise estimate.',
        'Fixed-adjoint comparisons hold the entire incoming tensor, not every induced internal state. Lower-layer incoming directions can change.',
        'The conditional update result does not prove the entire old temporal mechanism; observed before-probe variation remains.'])
path=HERE/'completed-common-update-chain-v58.json';path.write_text(json.dumps(report,indent=2)+'\n')
for name,a in arms.items():
    print(name,'VJP',a['parameter_VJP_ratio'],'RSS',a['B4_increment_RSS_ratio'],'direction',a['B4_accumulation_direction_factor_ratio'])
    print('layer input ratios',[(k,a['layer_input_adjoint_norm_ratio'][str(k)]) for k in (0,15,24,29,30,31)])
    print('layer parameter ratios',[(k,a['layer_parameter_VJP_norm_ratio'][str(k)]) for k in (0,15,24,29,30,31)])
print('receipt',path)
