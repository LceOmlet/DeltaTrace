"""Find actual saved histories with identical actions/observations and different reasoning.

This only fingerprints existing native IDs. Filtered sequences are never fed
to a model. No token is intervened on, no causal value is read, and no training
or numerical-model implementation is introduced.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
account=json.loads((OUT/'historical-B4-weight-accounting-v1.json').read_bytes())['rows']
white=json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())['rows']
windows={'early':range(1,11),'middle':range(31,41),'late':range(81,86)}
groups={name:defaultdict(list) for name in windows}
skipped=[];loaded=0
def sha_ids(ids):return hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()
for name,steps in windows.items():
    for step in steps:
        a=account[step-1];w=white[step-1]
        if a['truncated_successes']:
            skipped.append(dict(iteration=step,reason='Original retained response truncates successful target histories.'))
            continue
        saved={}
        for path in w['raw_files']:
            for index,row in enumerate(torch.load(path,map_location='cpu',weights_only=False)['rows']):
                saved.setdefault(row['traj_uid'],(row,path,index))
        used=set()
        for original in a['positions']:
            if original['reward']!=1:continue
            uid=original['representative_UID']
            if uid in used:continue
            used.add(uid)
            row,path,index=saved[uid];loaded+=1
            ids=row['input_ids'];prompt=int(row['prompt_length']);response=ids[prompt:]
            policy=row['policy_mask'][:len(response)].bool()
            target=row['target_mask'][:len(response)].bool()
            assert not (target&~policy).any()
            assert target.nonzero().flatten().tolist()==row['target_offsets']
            source=policy&~target
            first=int(policy.nonzero()[0]);first_target=int(target.nonzero()[0])
            initial=sha_ids(ids[:prompt+first])
            # A fingerprint only: preserve the complete original order of
            # prompt, observations and targets while omitting source tokens.
            # It is not a generated or counterfactual input sequence.
            non_source=torch.cat((ids[:prompt],response[~source]))
            key=sha_ids(non_source)
            entry=dict(iteration=step,UID=uid,raw_source=dict(path=path,row_index=index),
                initial_native_state_SHA256=initial,non_source_ID_SHA256=key,
                target_ID_SHA256=sha_ids(response[target]),observation_ID_SHA256=sha_ids(response[~policy]),
                source_ID_SHA256=sha_ids(response[source]),causal_context_tokens=len(ids),
                causal_policy_tokens=int(policy.sum()),target_tokens=int(target.sum()),
                source_tokens=int(source.sum()),source_tokens_before_first_target=int(source[:first_target].sum()),
                original_full_policy_tokens=original['native_success_policy_count'],
                original_target_alpha_rounded=w['native_whitened_maximum_rounded'],
                original_B4_policy_count_approx=a['B4'][original['position']//4]['policy_count'],
                original_native_dump_position=original['position'])
            groups[name][key].append(entry)
common=sorted(set(groups['early'])&set(groups['late']))
triples=sorted(set(common)&set(groups['middle']))
matched=[]
for key in common:
    a=groups['early'][key];b=groups['late'][key]
    assert len({r['initial_native_state_SHA256'] for r in a+b})==1
    assert len({r['target_ID_SHA256'] for r in a+b})==1
    assert len({r['observation_ID_SHA256'] for r in a+b})==1
    matched.append(dict(non_source_ID_SHA256=key,initial_native_state_SHA256=a[0]['initial_native_state_SHA256'],
        early=a,late=b,middle=groups['middle'].get(key,[]),
        early_source_median=statistics.median(r['source_tokens'] for r in a),
        late_source_median=statistics.median(r['source_tokens'] for r in b)))
report=dict(unix=time.time(),scope=__doc__,windows={k:[min(v),max(v)] for k,v in windows.items()},
    original_unique_positive_rows_loaded=loaded,skipped_iterations=skipped,
    common_early_late_action_observation_sequences=len(common),common_early_middle_late_sequences=len(triples),
    distinct_matched_initial_states=len({r['initial_native_state_SHA256'] for r in matched}),
    available_early_late_original_pairs=sum(min(len(r['early']),len(r['late'])) for r in matched),
    available_three_window_original_triples=sum(min(len(r['early']),len(r['middle']),len(r['late'])) for r in matched),
    same_action_observation_reasoning_shorter=sum(r['late_source_median']<r['early_source_median'] for r in matched),
    matched=matched,
    question='Does the saved corpus contain an early/late comparison where native initial state, complete target/action IDs and observation IDs/order are identical, leaving only actual non-target policy history differences?',
    evidence_gap='The completed same-task gradient contrast retained changes in actual target/actions and observations as well as reasoning. Before another native derivative measurement, inspect whether the existing corpus can separate these input changes without fabricating a counterfactual.',
    selection='Exact byte equality of native non-source ID sequences; fixed windows, all eligible saved positive rows. No norm, probability or advantage value selects a match.',
    status='CPU_inventory_only_no_GPU_probe_or_repair',model_calls=0,DT_calls=0,backward=0,optimizer=0,
    limits=['Positive saved trajectories only; no missing zero-reward trajectory is reconstructed.',
            'The saved input ends at the last causal target; later zero-target suffix tokens are not reconstructed.',
            'These matches are existing factual trajectories, not a world intervention, a value-error reference, or evidence of a population causal effect.',
            'Matched observations/actions do not themselves show which advantage caused histories to change or establish a repair.',
            'Native original B4 denominator estimates and printed alpha rounding remain approximate metadata, not exact gradient factors.'],
    script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
assert not torch.cuda.is_initialized()
dest=OUT/'matched-action-observation-history-inventory-v19.json'
assert not dest.exists(),'Preserve a completed inventory.'
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
