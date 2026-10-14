"""Select a same-task early/late saved-input contrast before any native replay."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
account=json.loads((OUT/'historical-B4-weight-accounting-v1.json').read_bytes())['rows']
white=json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())['rows']
by_window={'early':defaultdict(list),'late':defaultdict(list)}
for step in list(range(1,11))+list(range(81,86)):
    a=account[step-1];w=white[step-1]
    if a['truncated_successes']:continue
    saved={};origin={}
    for path in w['raw_files']:
        for index,row in enumerate(torch.load(path,map_location='cpu',weights_only=False)['rows']):
            if row['traj_uid'] not in saved:
                saved[row['traj_uid']]=row;origin[row['traj_uid']]={'path':path,'row_index':index}
    tasks=defaultdict(list)
    for position in a['positions']:
        if position['reward']!=1:continue
        uid=position['representative_UID'];row=saved[uid]
        first=int(row['policy_mask'].bool().nonzero()[0])
        prefix=row['input_ids'][:row['prompt_length']+first]
        key=hashlib.sha256(prefix.numpy().tobytes()).hexdigest()
        entry={'iteration':step,'native_dump_position':position['position'],
               'representative_UID':uid,'raw_source':origin[uid],
               'initial_prefix_ID_SHA256':key,'initial_prefix_tokens':len(prefix),
               'policy_tokens':position['native_success_policy_count'],
               'target_tokens':position['target_tokens'],
               'original_B4_policy_count':a['B4'][position['position']//4]['policy_count'],
               'original_target_alpha_rounded':w['native_whitened_maximum_rounded']}
        tasks[key].append(entry)
    window='early' if step<=10 else 'late'
    for key,rows in tasks.items():
        if len(rows)>=4:by_window[window][key].append(rows[:4])
common=sorted(set(by_window['early'])&set(by_window['late']))
selected=[{'task_prefix_SHA256':key,'early':by_window['early'][key][0],
           'late':by_window['late'][key][0]} for key in common[:16]]
report={'unix':time.time(),'scope':__doc__,'GPU_model_calls':0,
        'eligible_same_native_initial_states':len(common),'selected_tasks':selected,
        'selection':'First16 sorted native initial-prefix hashes; first chronological iteration with at least4 positive saved rows in each fixed window; first4 original dump positions. No gradient value is used to select tasks or rows.',
        'question':'Does the late-input target derivative amplification remain after matching the exact initial task states, at the same original checkpoint5, with the saved historical target coefficients?',
        'why_needed':'Earlier same-checkpoint iteration6/81 comparisons changed task composition as well as generated histories and coefficients. This contrast removes that specific ambiguity rather than repeating a capacity or validation-quality check.',
        'planned_scope':'Two positive-target native PG backward passes, global64/B4, same checkpoint5 and16 matched task states. No optimizer/scheduler/DT/generation/validation/GRPO calls. It is a historical input comparison, not fresh-training repair acceptance.',
        'training_source_changes':0,'historical_B4_failure_count_uncertainty_retained':True,
        'status':'CPU_inventory_only_no_new_GPU_replay_submitted'}
assert not torch.cuda.is_initialized()
dest=OUT/'matched-native-task-inventory-v11.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
