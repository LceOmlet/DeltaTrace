"""Report fixed-input gradient observations without claiming historical causality."""
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'read_fixed_history_status_v1.json').read_bytes())['report']
assert data['result']['phase']=='completed_fixed_batch_gradient_observations' and not data['alive']
out={'scope':__doc__,'job':data['job'],'rows':[],
     'interpretation':'Fixed late input/advantages give large gradients already at checkpoint5; checkpoint change5->80 increases this fixed-input norm by only about35%. This does not reproduce the much larger temporal growth in native on-policy logs, and does not establish a unique root cause.',
     'limits':['The late iteration81 coefficients are held fixed; they are not historical checkpoint5 or40 on-policy coefficients.',
               'Each checkpoint uses its own native recomputed old-logprob point; no optimizer/scheduler update or validation occurs.',
               'Existing diagnostic sums FP64 squared DTensor local-shard gradients over their original mesh; the native clip returns are retained separately, with their semantics not inferred from scalar values.',
               'Do not divide historical growth by this contrast to assign causal percentages.'],
     'source':str(HERE/'read_fixed_history_status_v1.json')}
for obs in data['result']['observations']:
    step=obs['checkpoint'];params=[data['parameter_statistics'][f'checkpoint{step}-rank{r}-parameter-statistics.json'] for r in (0,1)]
    assert all(p['trainable_SHA256_before']==p['trainable_SHA256_after'] for p in params)
    assert set(params[0]['parameters'])==set(params[1]['parameters'])
    totals={}
    for n,p in params[0]['parameters'].items():
        entries=[rank['parameters'][n] for rank in params]
        assert entries[0]['bucket']==entries[1]['bucket']
        if p['bucket'][0]=='sharded':sq=sum(x['gradient_sumsq'] or 0 for x in entries)
        else:sq=p['gradient_sumsq'] or 0
        totals[n]=sq
    layers={};kinds={}
    for n,sq in totals.items():
        layer=re.search(r'layers\.(\d+)\.',n)[1];layers[layer]=layers.get(layer,0)+sq
        kind='A' if '.lora_A.' in n else 'B';kinds[kind]=kinds.get(kind,0)+sq
    squared=sum(totals.values())
    measured=obs['gradient_statistics_by_rank'][0]['norms']['fixed_DT_PG']
    out['rows'].append({'checkpoint':step,'FP64_shard_gradient_norm':measured,'parameter_sums_reconstruct_norm':squared**.5,
       'native_clip_returns_by_rank':obs['native_clip_return_by_rank'],'parameters_unchanged':True,
       'first_microbatch_clip_fraction_by_rank':[x[0]['pg_clipfrac'] for x in obs['microbatch_PPO_diagnostics']],
       'A_squared_norm_fraction':kinds['A']/squared,
       'top_layers_squared_norm_fraction':sorted([(k,v/squared) for k,v in layers.items()],key=lambda x:x[1],reverse=True)[:5],
       'top_parameters_squared_norm_fraction':sorted([(k,v/squared) for k,v in totals.items()],key=lambda x:x[1],reverse=True)[:5]})
out['checkpoint80_over_checkpoint5_fixed_input_norm_ratio']=out['rows'][-1]['FP64_shard_gradient_norm']/out['rows'][0]['FP64_shard_gradient_norm']
(HERE/'fixed-history-analysis-v1.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
