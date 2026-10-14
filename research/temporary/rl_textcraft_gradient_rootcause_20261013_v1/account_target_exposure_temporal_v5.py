"""Account for target placement using the original 85 saved rollouts only.

This is an observational summary, not a new credit implementation or a test
of a fresh training counterfactual. No model, backward or update is called.
"""
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import torch
from transformers import AutoTokenizer

torch.set_num_threads(4)
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN = ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch = json.loads((RUN/'launch.json').read_bytes())
tokenizer = AutoTokenizer.from_pretrained(launch['options']['actor_rollout_ref.model.path'],
    trust_remote_code=True, local_files_only=True)
inference = json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())
account = json.loads((OUT/'historical-B4-weight-accounting-v1.json').read_bytes())
native = json.loads((OUT/'temporal-native-reports-v1.json').read_bytes())

def median(xs):
    return statistics.median(xs) if xs else None

report = {'unix':time.time(), 'scope':__doc__, 'model_GPU_calls':0, 'rows':[],
    'limits':[
        'Positive native IDs and masks are deduplicated by original UID; zero-reward decoded text is used only for the explicitly labelled text feature.',
        'Targets and source coefficients remain those of the fixed DT/VERL run; this summary does not modify a denominator or any owner.',
        'Observed shortening, placement and force can support a mechanism but cannot prove the counterfactual outcome of another training trajectory.',
        'The factual DT endpoint score is not the native actor score; these readouts are labelled separately.',
        'First-target prefix length counts generated policy tokens, not observations; full trajectories beyond the native response cap are flagged.',
    ]}
for step, prior in enumerate(inference['rows'],1):
    saved = {}
    for path in prior['raw_files']:
        for row in torch.load(path,map_location='cpu',weights_only=False)['rows']:
            saved.setdefault(row['traj_uid'],row)
    before=[]; total=[]; target_counts=[]; first_ids=[]; immediate=[]
    for row in saved.values():
        policy=row['policy_mask'].bool()[:10240]
        target=row['target_mask'].bool()[:10240]
        locations=target.nonzero().flatten()
        if not len(locations):
            continue
        first=int(locations[0])
        before.append(int(policy[:first].sum()))
        target_counts.append(int(target.sum()))
        total.append(int(policy.sum()))
        response=row['input_ids'][row['prompt_length']:][:10240]
        first_ids.append(int(response[first]))
        prefix=tokenizer.decode(response[:first],skip_special_tokens=True)
        # This is an explicit text feature, not a reconstruction of policy IDs.
        immediate.append(bool(re.search(r'Thought:\s*$',prefix)))
    dump_path=Path(launch['options']['trainer.rollout_data_dir'])/f'{step}.jsonl'
    dump_bytes=dump_path.read_bytes()
    dumps=[json.loads(line) for line in dump_bytes.splitlines()]
    text_immediate=[bool(re.search(r'Thought:\s*Action:',r['output'])) for r in dumps]
    score=native['metrics'][step-1]['values']
    report['rows'].append({
        'iteration':step,'original_dump_path':str(dump_path),
        'original_dump_SHA256':hashlib.sha256(dump_bytes).hexdigest(),
        'positive_native_UIDs':len(saved),'native_grad_norm':score['actor/grad_norm'],
        'native_reward':score.get('critic/rewards/mean'),
        'native_entropy':score.get('actor/entropy_loss'),
        'raw_credit_std_inferred':prior.get('inferred_raw_std'),
        'native_target_actor_coefficient_rounded':prior['native_whitened_maximum_rounded'],
        'positive_first_target_preceding_policy_tokens_median':median(before),
        'positive_first_target_preceding_policy_tokens_mean':statistics.mean(before),
        'positive_first_target_preceding_policy_tokens_le16_fraction':sum(n<=16 for n in before)/len(before),
        'positive_empty_Thought_before_first_target_fraction':sum(immediate)/len(immediate),
        'all_rollout_text_has_empty_Thought_fraction':sum(text_immediate)/len(text_immediate),
        'positive_policy_tokens_median':median(total),
        'positive_target_tokens_median':median(target_counts),
        'positive_target_fraction_median':median([t/n for t,n in zip(target_counts,total)]),
        'positive_first_target_token_counts':{str(t):first_ids.count(t) for t in sorted(set(first_ids))},
        'truncated_positive_UIDs':prior['truncated_successes'],
    })
report['windows']=[]
for lo,hi in [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,85)]:
    rows=report['rows'][lo-1:hi]
    keys=[k for k,v in rows[0].items() if isinstance(v,(float,int)) and k not in ('iteration','positive_native_UIDs')]
    report['windows'].append({'iterations':[lo,hi], 'n':len(rows),
        'medians':{k:median([r[k] for r in rows if r.get(k) is not None]) for k in keys}})
assert not torch.cuda.is_initialized()
path=OUT/'target-exposure-temporal-v5.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
