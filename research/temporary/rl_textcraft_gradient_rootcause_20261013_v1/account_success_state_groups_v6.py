"""Group saved successes by the exact state before the first policy token.

The decoded dump input is the common system prompt and cannot recover task
groups. Use original IDs including the first observation; calibrate against
native UIDs on the complete iteration81 snapshot. No GRPO execution occurs.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch
from verl import DataProto

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
inference=json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())

def state_key(ids):
    return hashlib.sha256(ids.to(torch.int32).contiguous().numpy().tobytes()).hexdigest()

report={'unix':time.time(),'scope':__doc__,'model_GPU_calls':0,'rows':[],
    'withdrawn_diagnostic':'Historical decoded-input groups in native-target-force-v5 are unavailable: all256 input strings are the common system prompt. Their zero all-success counts must not be used.',
    'limits':[
        'Positive-only state group counts describe the observed batch, not conditional population success probabilities.',
        'A count of8 is a fully successful native group only where task states map one-to-one to the native eight-row UIDs. Iteration81 is checked directly; earlier repeated task-state collisions cannot be universally excluded.',
        'This is a credit-signal comparison with the original GRPO formula, not a rerun or a claim that removing this component repairs training.',
    ]}
for step,prior in enumerate(inference['rows'],1):
    saved={}
    for file in prior['raw_files']:
        for row in torch.load(file,map_location='cpu',weights_only=False)['rows']:
            saved.setdefault(row['traj_uid'],row)
    groups=defaultdict(list)
    for row in saved.values():
        first=int(row['policy_mask'].bool().nonzero()[0])
        key=state_key(row['input_ids'][:row['prompt_length']+first])
        groups[key].append(row)
    total=sum(int(r['target_mask'].bool()[:10240].sum()) for r in saved.values())
    full=[v for v in groups.values() if len(v)==8]
    saturated=sum(int(r['target_mask'].bool()[:10240].sum()) for v in full for r in v)
    report['rows'].append({'iteration':step,'positive_native_UIDs':len(saved),
        'initial_state_success_counts':sorted(len(v) for v in groups.values()),
        'states_with_eight_successes':len(full),
        'retained_positive_target_tokens':total,
        'retained_positive_target_tokens_in_eight_success_states':saturated,
        'retained_target_fraction_in_eight_success_states':saturated/total,
        'state_collisions_over_eight':sum(len(v)>8 for v in groups.values())})
sources=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(r['input']) for r in sources])
state_to_uid=defaultdict(set);uid_to_state=defaultdict(set);counts=defaultdict(int)
for i in range(len(data)):
    p=int(data.batch['input_ids'].shape[-1]-data.batch['responses'].shape[-1])
    first=int(data.batch['response_mask'][i].bool().nonzero()[0])
    ids=data.batch['input_ids'][i,:p+first]
    att=data.batch['attention_mask'][i,:p+first].bool()
    key=state_key(ids[att]);uid=str(data.non_tensor_batch['uid'][i])
    state_to_uid[key].add(uid);uid_to_state[uid].add(key);counts[key]+=1
report['native81_group_calibration']={
    'state_count':len(state_to_uid),'uid_count':len(uid_to_state),
    'state_row_counts':sorted(counts.values()),
    'state_to_uid_one_to_one':all(len(v)==1 for v in state_to_uid.values()),
    'uid_to_state_one_to_one':all(len(v)==1 for v in uid_to_state.values())}
assert report['native81_group_calibration']['state_to_uid_one_to_one']
assert report['native81_group_calibration']['uid_to_state_one_to_one']
assert not torch.cuda.is_initialized()
path=OUT/'success-state-groups-v6.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
