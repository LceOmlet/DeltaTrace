"""Summarize native PG localization and audit the earlier unmatched pair."""
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime

HERE=Path(__file__).resolve().parent
new=[json.loads((HERE/f'native-PG-groups-v1/rank{r}-groups.json').read_bytes()) for r in (0,1)]
old=[json.loads((HERE/f'dt-components-rank{r}.json').read_bytes()) for r in (0,1)]
candidate=[json.loads((HERE/f'dt-global-token-mean-v2-components-rank{r}.json').read_bytes()) for r in (0,1)]
assert new[0]['gradient_statistics']==new[1]['gradient_statistics']
stats=new[0]['gradient_statistics']
labels=new[0]['labels']
gram=stats['inner_products']
def dot(a,b):return gram.get(a+':'+b,gram.get(b+':'+a))
square=sum(dot(a,b) for a in labels for b in labels)
comparisons=[]
for rank in (0,1):
    reference=new[rank]['passes']['successful_target']['microbatch_losses']
    groups_identical=all(all(p[k]==q[k] for k in ['weighted_entropy','weighted_kl','ppo_kl','pg_clipfrac','pg_clipfrac_lower'])
        for label in labels for p,q in zip(reference,new[rank]['passes'][label]['microbatch_losses']))
    legacy=old[rank]['passes']['dt_pg']['microbatch_losses']
    revised=candidate[rank]['passes']['dt_pg']['microbatch_losses']
    comparisons.append({'rank':rank,'new_group_forward_statistics_identical':groups_identical,
        'old_baseline_entropy_minus_new':[a['weighted_entropy']['value']-b['weighted_entropy']['value'] for a,b in zip(legacy,reference)],
        'old_candidate_entropy_minus_new':[a['weighted_entropy']['value']-b['weighted_entropy']['value'] for a,b in zip(revised,reference)],
        'old_baseline_ppo_kl':[a['ppo_kl'] for a in legacy],
        'old_baseline_clip_fraction':[a['pg_clipfrac'] for a in legacy],
        'new_ppo_kl':[a['ppo_kl'] for a in reference],
        'new_clip_fraction':[a['pg_clipfrac'] for a in reference]})
    assert groups_identical
report={'observed_local':datetime.now().astimezone().isoformat(),'scope':__doc__,
    'diagnostic_source_commit':'db3bf28ef2c1dad3a89f0153063cd2b065769a54',
    'diagnostic_source_SHA256':hashlib.sha256((HERE/'observe_native_PG_groups_v1.py').read_bytes()).hexdigest(),
    'native_actor':new[0]['sources']['actor'],'native_core':new[0]['sources']['core'],
    'group_gradient_statistics':stats,'reconstructed_sum_PG_norm':math.sqrt(square),
    'signed_projection_on_sum':{a:sum(dot(a,b) for b in labels)/square for a in labels},
    'factual_point':{'input':'original iteration81 first global64','checkpoint':80,
        'whole_policy_mask_and_B4_denominators_preserved':True,'group_advantages_sum_to_original':True,
        'optimizer_steps':0,'scheduler_steps':0,'DT_calls':0,'rollout_calls':0,'GRPO_calls':0,'validation_calls':0,
        'parameters_unchanged':all(json.loads((HERE/f'native-PG-groups-v1/rank{r}-state.json').read_bytes())['trainable_SHA256_before']==json.loads((HERE/f'native-PG-groups-v1/rank{r}-state.json').read_bytes())['trainable_SHA256_after'] for r in (0,1))},
    'forward_point_audit':comparisons,
    'withdrawn_evidence':'The old legacy PG0.9583064 versus candidate PG0.1620889 ratio was not a controlled same-forward-point comparison: rank1 entropy/KL/clipping differ. Do not use its5.9 ratio as a causal or repair effect. The independent scalar/dL/dlogp global-token-mean reference remains valid.',
    'limits':['This localizes a late-point native PG, not the85-iteration temporal cause.',
        'FP64 global sharded gradient norms and the logged native mean of per-optimizer preclip norm returns have distinct aggregation scopes.',
        'Signed projection shares include cancellation and are not independent percentages of gradient norm.',
        'The older point mismatch is an evidence defect, not evidence of the original training growth mechanism.'],
    'rootcause_resolved':False,'accepted_repair':None,'deployed_repair':None}
path=HERE/'native-PG-localization-audit-v2.json'
path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report':str(path.resolve()),'norms':stats['norms'],'sum':math.sqrt(square),'projection':report['signed_projection_on_sum'],'old_pair_withdrawn':True}))
