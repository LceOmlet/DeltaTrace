"""Localize the measured fixed-input/fixed-seed gain in existing gradient records.

Only saved native pre-clip parameter shard norms are summed. This is neither
a new model evaluation nor an independent causal attribution of the updates.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
inputs=[HERE/'native-fixed-seed-propagation-v33-status-final.json',
        HERE/'native-midpoint-fixed-seed-v35-status-final.json']
wanted={5:'late5_seed80',40:'late40_seed80',80:'late80_native'}
cases={}
for path in inputs:
    for row in json.loads(path.read_text(encoding='utf-8'))['report']['chain_reports']:
        r=row['value'];cp=r['checkpoint']
        if wanted[cp]!=r['iteration']:continue
        state=cases.setdefault(cp,dict(ranks=[],parameters=defaultdict(float)))
        state['ranks'].append(r['rank'])
        for p in r['native_accumulation_readout']['parameters']:
            assert p['placements']==['S(0)']
            state['parameters'][p['name']]+=p['local_preclip_sumsq']
for cp,c in cases.items():
    assert sorted(c['ranks'])==[0,1]
    layer=defaultdict(float);family=defaultdict(float);factor=defaultdict(float)
    for name,ss in c['parameters'].items():
        match=re.search(r'\.layers\.(\d+)\.(.+)\.lora_([AB])\.',name)
        assert match,name
        layer[int(match[1])]+=ss;family[match[2]]+=ss;factor[match[3]]+=ss
    total=sum(c['parameters'].values())
    c.update(norm=math.sqrt(total),total_sumsq=total,
             layers={str(k):dict(norm=math.sqrt(v),energy_fraction=v/total) for k,v in sorted(layer.items())},
             families={k:dict(norm=math.sqrt(v),energy_fraction=v/total) for k,v in sorted(family.items())},
             LoRA_factors={k:dict(norm=math.sqrt(v),energy_fraction=v/total) for k,v in sorted(factor.items())})
    c['parameters']=dict(sorted(c['parameters'].items()))
growth={}
for a,b in [(5,40),(40,80),(5,80)]:
    old,new=cases[a],cases[b]
    growth[f'{a}_to_{b}']=dict(total_norm_ratio=new['norm']/old['norm'],
        layers={k:dict(norm_ratio=new['layers'][k]['norm']/old['layers'][k]['norm'],
                       old_energy_fraction=old['layers'][k]['energy_fraction'],
                       new_energy_fraction=new['layers'][k]['energy_fraction']) for k in old['layers']},
        families={k:dict(norm_ratio=new['families'][k]['norm']/old['families'][k]['norm'],
                         old_energy_fraction=old['families'][k]['energy_fraction'],
                         new_energy_fraction=new['families'][k]['energy_fraction']) for k in old['families']})
report=dict(scope=__doc__,source_snapshots={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    cases={str(k):v for k,v in sorted(cases.items())},growth=growth,
    model_calls=0,optimizer_updates=0,DT_calls=0,rootcause_repair_verified=False,
    limits=['These parameter groups partition gradient coordinates, not causes or independent token contributions.',
            'Same actual output adjoint and same input at checkpoints5/40/80 isolate weight/induced-state directional sensitivity.',
            'This does not identify the original update component that produced those weights, and does not imply every actual full policy gradient grows by the same ratio.'])
dest=HERE/'saved-fixed-seed-parameter-growth-v41.json'
dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('norms',[(k,v['norm']) for k,v in sorted(cases.items())])
for k in ('5_to_40','40_to_80','5_to_80'):
    r=growth[k]
    print(k,r['total_norm_ratio'])
    print('layer0,15,24,29,30,31',[(j,r['layers'][str(j)]) for j in (0,15,24,29,30,31)])
print('largest families80',sorted(cases[80]['families'].items(),key=lambda item:item[1]['energy_fraction'],reverse=True)[:5])
print('receipt',dest)
