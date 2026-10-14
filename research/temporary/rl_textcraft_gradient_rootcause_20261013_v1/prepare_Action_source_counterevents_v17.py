"""Check whether the two observed iteration81 Action-source counterevents can compensate.

These are the opposite-direction events already identified in the85-batch
inventory, not new extrema or a different outcome definition. The finite
cancellation calculation is diagnostic, not a population-baseline assertion.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import torch
from verl import DataProto
from reward_readout import DirectActionTargetReadout

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-source-delete-values-v17';DEST.mkdir(exist_ok=True)
previous=DEST/'query-manifest.json'
if previous.exists():
    assert not (DEST/'job.json').exists() and not (DEST/'result.json').exists()
    preserved=DEST/'prepared-only-query-manifest-v17-initial.json'
    assert not preserved.exists();preserved.write_bytes(previous.read_bytes())
role_inventory=json.loads((OUT/'first-Thought-baseline-roles-v13.json').read_bytes())['rows']
counterevents=role_inventory[80]['observed_Action_source_counterevents']
assert len(counterevents)==2
paths=json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())['rows'][80]['raw_files']
raw={}
for path in paths:
    for index,row in enumerate(torch.load(path,map_location='cpu',weights_only=False)['rows']):
        raw.setdefault(row['traj_uid'],(row,path,index))
sources=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(r['input']) for r in sources])
forks={f['prefix_SHA256']:f for f in json.loads((OUT/'same-prefix-DT-values-v5.json').read_bytes())['forks']}
queries=[];inventory=[]
for point in counterevents:
    uid=point['UID'];row,path,index=raw[uid];slot=point['position']
    prompt=int(row['prompt_length']);ids=row['input_ids'];response=ids[prompt:]
    policy=row['policy_mask'][:len(response)].bool();target=row['target_mask'][:len(response)].bool()
    assert policy[slot] and not target[slot] and int(response[slot])==point['token_id']
    prepared=DirectActionTargetReadout._prepare_row(dict(input_ids=ids,attention_mask=torch.ones_like(ids),
        responses=response,policy_mask=policy,target_mask=target,dt_direct_reward=row['reward'],traj_uid=uid),index)
    assert prepared['target_offsets']==row['target_offsets']
    assert prompt+slot<min(prompt+j for j in row['target_offsets'])
    prefix=hashlib.sha256(b''.join(struct.pack('<I',int(i)) for i in ids[:prompt+slot])).hexdigest()
    fork=forks.get(prefix)
    occurrences=[] if fork is None else [p for b in fork['branches'] for p in b['occurrences']]
    native_rows=[i for i,v in enumerate(data.non_tensor_batch['traj_uid']) if str(v)==uid]
    assert native_rows
    i=native_rows[0];positions=torch.as_tensor(data.non_tensor_batch['dt_direct_target_artifact'][i]['retained_response_positions'])
    retained=positions.eq(slot).nonzero().flatten();assert len(retained)==1
    factual=float(row['factual_target_logp'])
    bound=math.exp(-factual)
    action=[p for p in occurrences if p['token_id']==point['token_id']]
    other=[p for p in occurrences if p['token_id']!=point['token_id']]
    balance=None if not other else len(action)*sum(p['V'] for p in other)/len(other)
    inventory.append(dict(UID=uid,prefix_SHA256=prefix,original_factual_joint_logp=factual,
        native_probability_upper_bound_for_V=bound,saved_V=point['raw_V'],
        action_choice_observations=len(action),other_choice_observations=len(other),
        empirical_Action_V_sum=sum(p['V'] for p in action),
        empirical_other_V_sum=sum(p['V'] for p in other),
        Action_V_sum_needed_to_equal_observed_other_mean=balance,
        scope='Finite observed conditional-V means, not a claim of equal true Q or a distribution-free population test. The probability upper bound uses the recorded native factual endpoint, subject to its numerical precision.'))
    queries.append(dict(UID=uid,original_native_row=i,native_response_slot=int(retained[0]),
        original_response_slot=slot,packed_slot=prompt+slot,token_id=point['token_id'],
        saved_DT_d=float(row['source_log_ratios'][slot]),saved_V=point['raw_V'],
        saved_Q=point['raw_Q'],saved_raw_A=point['raw_A'],original_raw_file=path,original_raw_row=index,
        selected=prepared['selected'],case=prepared['case'],target_offsets=prepared['target_offsets']))
payload=DEST/'queries.pt';torch.save(queries,payload)
report=dict(unix=time.time(),scope=__doc__,destination=str(DEST),iteration=81,checkpoint=80,
    query_file=str(payload),query_SHA256=hashlib.sha256(payload.read_bytes()).hexdigest(),
    distinct_source_points=len(queries),lengths=[len(q['selected']) for q in queries],inventory=inventory,
    question='Do the two already observed Action_source counterevents at the same original checkpoint80 carry a large native reference V that was underestimated by DT, potentially offsetting the observed Action-target/source preference?',
    why_needed='All21 measured non-Action sources leave a positive baseline contribution favoring Action. This does not exclude rare opposite Action_source events with large inverse-likelihood weights. Their two actual iteration81 values must be checked before blaming the method construction.',
    decision='Retain or reject practical missing-counterweight evidence from the actual values, while preserving finite sampling, unequal B4 weights and native numerical drift limits. No result is a population unbiasedness proof or repair acceptance.',
    selection='All two Action_source choices in the original iteration81 first-Thought inventory; no extremum selection or additional rollout.',
    operations=dict(model=0,DT=0,backward=0,optimizer=0,rollout=0),
    accepted_repair=None,deployed_repair=None,formal_training_changed=False)
matched_prefixes={g['prefix_SHA256'] for g in json.loads((OUT/'native-QV-Action-direction-v12.json').read_bytes())['groups']}
all_opposite=[]
for step in role_inventory:
    opposite=step['observed_Action_source_counterevents']
    if not opposite:continue
    originals={}
    for path in step['raw_sources']:
        for row in torch.load(path,map_location='cpu',weights_only=False)['rows']:
            originals.setdefault(row['traj_uid'],row)
    for p in opposite:
        row=originals[p['UID']];end=int(row['prompt_length'])+p['position']
        prefix=hashlib.sha256(b''.join(struct.pack('<I',int(i)) for i in row['input_ids'][:end])).hexdigest()
        all_opposite.append(dict(iteration=step['iteration'],UID=p['UID'],prefix_SHA256=prefix,
            matches_any_native81_target_source_fork=prefix in forks,
            matches_any_of_measured14_groups=prefix in matched_prefixes,
            saved_raw_V=p['raw_V'],factual_joint_logp=row['factual_target_logp']))
report['all85_observed_opposite_prefix_audit']=all_opposite
report['status']='prepared_only_not_submitted_CPU_prefix_and_probability_evidence_retained'
report['GPU_submission_decision']='Do not launch the prepared two-point probe now: their exact prefixes do not match the measured14 fork groups and their recorded native probability bound is close to1. Their mere presence does not demonstrate a missing large compensating weight at those states. This does not exclude unseen opposite outcomes or prove a population bias.'
assert not torch.cuda.is_initialized()
dest=DEST/'query-manifest.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
