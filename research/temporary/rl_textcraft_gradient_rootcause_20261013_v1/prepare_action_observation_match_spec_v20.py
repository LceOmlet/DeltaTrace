"""Bind all15 exact action/observation matches to an existing native derivative carrier.

This does not manufacture a deletion reference. Both inputs are unchanged
observed histories. The same saved late target derivative coefficients are
transported to both histories through the existing original-loss carrier.
"""
import copy
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'matched-action-observation-target-derivatives-v20'
inventory=json.loads((OUT/'matched-action-observation-history-inventory-v19.json').read_bytes())
assert inventory['common_early_late_action_observation_sequences']==15
assert inventory['distinct_matched_initial_states']==15
selected=[]
for match in inventory['matched']:
    a=match['early'][0];b=match['late'][0]
    assert a['target_ID_SHA256']==b['target_ID_SHA256']
    assert a['observation_ID_SHA256']==b['observation_ID_SHA256']
    assert a['non_source_ID_SHA256']==b['non_source_ID_SHA256']
    pair={}
    for name,entry in [('early',a),('late',b)]:
        row=copy.deepcopy(entry)
        row.update(initial_prefix_ID_SHA256=entry['initial_native_state_SHA256'],representative_UID=entry['UID'],
            policy_tokens=entry['original_full_policy_tokens'],
            original_B4_policy_count=entry['original_B4_policy_count_approx'],
            diagnostic_common_target_alpha_rounded=b['original_target_alpha_rounded'],
            diagnostic_common_original_B4_policy_count=b['original_B4_policy_count_approx'])
        pair[name]=[row]
    pair['non_source_ID_SHA256']=match['non_source_ID_SHA256']
    selected.append(pair)
report=dict(unix=time.time(),scope=__doc__,destination=str(DEST),selected_tasks=selected,
    selection='All15 exact native action/observation ID sequence matches from fixed early1-10/late81-85 windows; first chronological original row in each. No gradient/advantage magnitude selects a match.',
    question='With initial state, executed target/action IDs, observation IDs/order, model checkpoint5 and every per-target derivative coefficient held the same, can the two actual reasoning histories alone produce the observed target-gradient amplification?',
    why_needed='The completed matched-task68x contrast still changed target/action and observation content. CPU inspection found15 genuine early/late matches with all non-source native IDs identical and shorter late reasoning. This separates that specific remaining input ambiguity without a counterfactual value reference or training experiment.',
    expected_discrimination='If amplification remains, actual reasoning-history differences are sufficient for it in these15 matched facts at fixed weights/target coefficients. If it disappears, the older68x result cannot be attributed to reasoning shortening alone; target/action or observation differences remain material. Neither result explains by itself why the policy adopted those histories.',
    limits=['Only15 matched successful factual pairs; not a population sample or a complete historical optimizer batch.',
        '49 unchanged original native rows have zero diagnostic target derivative in both cases to retain the original global64/B4 owner interface. They are not fabricated failure trajectories or independent matched evidence.',
        'Each pair retains the same saved late per-target derivative coefficient in both cases; this is a derivative comparison, not new training advantages or a repaired objective.',
        'Historical printed alpha and B4 denominator uncertainty applies to the common coefficient, rather than different early/late coefficients.',
        'Source-token values, lengths and placements differ naturally between the factual inputs. No input is intervened on or re-generated.',
        'This does not measure causal-value estimation error, prove full85-iteration root cause, or accept a repair.'],
    GPU_model_calls=0,optimizer=0,DT=0,rollout=0,validation=0,GRPO=0,
    accepted_repair=None,deployed_repair=None)
DEST.mkdir(exist_ok=True)
dest=DEST/'match-spec.json';assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
