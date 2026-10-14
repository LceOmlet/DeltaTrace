"""Read existing target force and reward groups, without any model execution."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from verl import DataProto

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
original=json.loads((OUT/'historical-B4-weight-accounting-v1.json').read_bytes())
report={'unix':time.time(),'scope':__doc__,'model_GPU_calls':0,'reward_group_rows':[]}
for iteration in range(1,86):
    path=Path(launch['options']['trainer.rollout_data_dir'])/f'{iteration}.jsonl'
    rows=[json.loads(line) for line in path.read_bytes().splitlines()]
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups[r['input']].append(i)
    # Decoded identical inputs are a bounded grouping diagnostic, not UID recovery.
    complete=[v for v in groups.values() if len(v)==8]
    all_success={i for ids in complete if all(rows[i]['score']==1 for i in ids) for i in ids}
    positions=original['rows'][iteration-1]['positions']
    total_target=sum(r.get('target_tokens',0) for r in positions)
    saturated_target=sum(r.get('target_tokens',0) for r in positions if r['position'] in all_success)
    report['reward_group_rows'].append({'iteration':iteration,'decoded_input_group_sizes':sorted(len(v) for v in groups.values()),
        'exact_eight_decoded_input_groups':len(complete),
        'all_success_exact_eight_groups':len(all_success)//8,
        'positive_target_tokens_in_all_success_exact_eight_groups':saturated_target,
        'positive_target_tokens':total_target,
        'positive_target_fraction_in_all_success_exact_eight_groups':saturated_target/total_target if total_target else None})

sources=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(s['input']) for s in sources])
policy=data.batch['response_mask'].bool()
force=defaultdict(lambda:defaultdict(lambda:{'tokens':0,'L1_deficit':0.,'L2_lower_squared':0.,'coefficient_squared':0.}))
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(launch['options']['actor_rollout_ref.model.path'],local_files_only=True,trust_remote_code=True)
for row,artifact in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
    if float(data.batch['token_level_rewards'][row].sum())==0:
        continue
    p=torch.as_tensor(artifact['retained_response_positions'])
    keep=p>=0
    target=torch.zeros_like(policy[row]);target[keep]=torch.as_tensor(artifact['target_mask'],dtype=torch.bool)[p[keep]]
    n=int(policy[(row//4)*4:(row//4)*4+4].sum())
    mini=(row%128)//32
    ids=data.batch['responses'][row]
    for position in target.nonzero().flatten().tolist():
        token=int(ids[position]); prefix=tok.decode(ids[max(0,position-35):position].tolist(),skip_special_tokens=False)
        label='Action_after_empty_Thought' if token==2432 and re.search(r'Thought:\s*$',prefix) else 'other_target_tokens'
        coefficient=float(data.batch['advantages'][row,position])/(16*n)
        deficit=-math.expm1(float(data.batch['old_log_probs'][row,position]))
        for group in (label,'all_target_tokens'):
            v=force[mini][group];v['tokens']+=1
            v['L1_deficit']+=abs(coefficient*deficit)
            v['L2_lower_squared']+=(coefficient*deficit)**2
            v['coefficient_squared']+=coefficient**2
report['native81_output_force_by_mini']=dict(force)
report['native81_UID_reward_groups']={}
uid_groups=defaultdict(list)
for i,uid in enumerate(data.non_tensor_batch['uid']):
    uid_groups[str(uid)].append(float(data.batch['token_level_rewards'][i].sum()))
report['native81_UID_reward_groups']={'sizes':sorted(len(r) for r in uid_groups.values()),
    'all_success':sum(all(v==1 for v in r) for r in uid_groups.values()),
    'mixed_reward':sum(len(set(r))>1 for r in uid_groups.values()),
    'all_zero':sum(all(v==0 for v in r) for r in uid_groups.values())}
report['limits']=[
    'The output-logit squared norm lies between L2_lower_squared and twice that value; full softmax vectors are not retained.',
    'Token output-force shares are not parameter-gradient shares or independent causal percentages.',
    'Historical grouping by decoded input is not claimed to recover native UIDs. Native UID grouping is separately checked at iteration81.',
    'Original fixed VERL/PPO and all saved DT coefficients remain untouched.',
    'All-success GRPO groups have zero native outcome advantages mathematically; no GRPO rerun is performed.',
]
assert not torch.cuda.is_initialized()
dest=OUT/'native-target-force-v5.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
