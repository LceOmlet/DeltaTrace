"""Account the completed native28.274x contrast using observed adjoints/Gram."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
status=json.loads((HERE/'read_native_gradient_chain_v4-dt.json').read_text(encoding='utf-8'))['report']
assert status['job']['pid']==3892843 and status['job']['birth']==1791960916.13
assert not status['alive'] and status['state']['phase']=='complete_native_gradient_chain_localization_not_repair'
cases=[]
for case in ('early','late'):
    ranks=[r['value'] for r in status['chain_reports'] if r['value']['iteration']==case]
    assert len(ranks)==2
    sums=defaultdict(lambda:defaultdict(float));events=[]
    gram=[[0.0]*8 for _ in range(8)];layer_sums=defaultdict(float)
    digests=[]
    for rank in ranks:
        assert rank['trainable_SHA256_before']==rank['trainable_SHA256_after']
        assert rank['scheduler_before']==rank['scheduler_after']==5
        digests.append(rank['trainable_SHA256_before'])
        counts=Counter((r['boundary'],r['micro']) for r in rank['records'])
        assert set(counts.values())=={1},counts
        events.append(dict(rank=rank['rank'],forward_events=len(rank['records']),unique_boundary_micro_events=len(counts)))
        for r in rank['records']:
            name=r['boundary'].split('base_model.model.')[-1]
            for which in ('input','output'):
                assert r.get(which+'_adjoint_nonfinite_positions',0)==0
                if which+'_adjoint_norm' in r:sums[name][which]+=r[which+'_adjoint_norm']**2
        readout=rank['native_accumulation_readout'];assert readout['original_backward_calls']==8
        for i in range(8):
            for j in range(8):gram[i][j]+=readout['local_increment_Gram'][i][j]
        for p in readout['parameters']:
            assert p['placements']==['S(0)'],p
            match=re.search(r'layers\.(\d+)\.',p['name'])
            key=('layer'+match.group(1) if match else 'other')+('_A' if '.lora_A.' in p['name'] else '_B')
            layer_sums[key]+=p['local_preclip_sumsq']
    # Native rank-local scalar losses are averaged by the original two-rank
    # FSDP parameter reducer. Thus concatenated output adjoints use /2 for
    # comparison with the logical parameter gradient. Ratios are unchanged.
    boundaries={name:{k:math.sqrt(v)/2 for k,v in row.items()} for name,row in sums.items()}
    observation=next(r for r in status['state']['observations'] if r['iteration']==case)
    G=observation['gradient_statistics']['norms']['native_target']
    Z=boundaries.get('lm_head',{}).get('output');H=boundaries['model.norm']['output']
    diagonal=math.fsum(gram[i][i] for i in range(8));assembled=math.fsum(v for row in gram for v in row)
    cases.append(dict(case=case,events=events,digests=digests,boundaries=boundaries,
        native_global_sharded_parameter_norm=G,native_global_logit_adjoint_norm=Z,
        native_global_final_hidden_adjoint_norm=H,H_over_Z=H/Z if Z else None,G_over_H=G/H,
        missing_logit_readout='The registered lm_head module forward was not called on this native path; no logit norm was measured.' if Z is None else None,
        B4_increment_Gram=gram,B4_increment_energy_sqrt=math.sqrt(diagonal),
        B4_directional_accumulation_factor=math.sqrt(assembled/diagonal),
        gradient_norm_from_increment_Gram=math.sqrt(assembled),
        gram_reconstruction_difference=math.sqrt(assembled)-G,
        parameter_sumsq_by_layer_and_adapter_side=dict(layer_sums),
        parameter_norm_from_final_shards=math.sqrt(sum(layer_sums.values()))))
assert cases[0]['digests']==cases[1]['digests']
a,b=cases
fields=['native_global_sharded_parameter_norm','native_global_logit_adjoint_norm','native_global_final_hidden_adjoint_norm',
        'H_over_Z','G_over_H','B4_increment_energy_sqrt','B4_directional_accumulation_factor']
ratios={k:b[k]/a[k] for k in fields if a[k] is not None and b[k] is not None}
boundary_ratios={name:{k:b['boundaries'][name][k]/v if v else None for k,v in row.items()} for name,row in a['boundaries'].items()}
report=dict(scope=__doc__,job=status['job'],cases=cases,late_over_early_ratios=ratios,boundary_ratios=boundary_ratios,
    limits=['All quantities are observed on the same15 matched facts/common coefficients/checkpoint5, with original native owner arithmetic.',
        'G=Z*(H/Z)*(G/H) is a finite norm identity. Its factors are dependent descriptions, not independent causal fractions or singular values.',
        'The B4 Gram uses differences of accumulated FP32 parameter gradients after native backward. Its reconstruction residual is reported rather than hidden.',
        'This locates the finite28.274x ratio. It does not by itself prove the entire historical85-step root cause or accept a repair.'],
    new_model_calls=0,optimizer_updates=0,rootcause_resolved=False,accepted_repair=None,deployed_repair=None)
dest=HERE/'native-gradient-ratio-chain-summary-v23.json'
dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(path=str(dest.resolve()),ratios=ratios,boundary_ratios=boundary_ratios)))
