"""Control task identity in the already completed 85-rollout observations.

Use the initial task observation before the first assistant turn, rather than
the common system prompt. Validate that grouping against native iteration81
UIDs; all data are read-only, and no model or training call occurs.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
original=json.loads((OUT/'historical-B4-weight-accounting-v1.json').read_bytes())
report={'unix':time.time(),'scope':__doc__,'model_GPU_calls':0,'iterations':[],
    'limits':[
        'Task identity uses the decoded initial observation, not native IDs. Native UID one-to-one calibration is checked on the saved full iteration81 batch.',
        'First Action text position is a decoded-text feature and can include an unexecuted earlier Action marker. The exact positive policy-token placement is separately measured in target-exposure-temporal-v5.',
        'Within-task changes remain observational. They do not identify a fresh-training counterfactual or a unique root cause.',
    ]}
tasks=defaultdict(list)
for previous in original['rows']:
    step=previous['step']
    dumps=[json.loads(s) for s in Path(previous['dump_path']).read_bytes().splitlines()]
    groups=defaultdict(list)
    for i,r in enumerate(dumps):
        parts=r['output'].split('\nassistant\n',1)
        assert len(parts)==2,(step,i,r['output'][:100])
        prefix=parts[0]
        key=hashlib.sha256(prefix.encode()).hexdigest()
        generated=parts[1].split('\nuser\n',1)[0]
        match=re.search(r'Action:\s*',generated)
        pos=previous['positions'][i]
        groups[key].append({'row':i,'reward':r['score'],
            'native_success_policy_count':pos.get('native_success_policy_count'),
            'native_success_target_count':pos.get('target_tokens'),
            'first_response_chars_before_first_Action':match.start() if match else None,
            'empty_Thought_before_first_Action':bool(re.search(r'Thought:\s*Action:',generated)),
            'initial_observation':prefix})
    assert len(groups)==32 and all(len(v)==8 for v in groups.values()),(step,len(groups),[len(v) for v in groups.values()])
    features=[]
    for key,rows in groups.items():
        good=[r for r in rows if r['reward']==1]
        entry={'task_key':key,'iteration':step,'success_count':len(good),
            'first_response_empty_Thought_fraction':sum(r['empty_Thought_before_first_Action'] for r in rows)/8,
            'positive_policy_count_median':statistics.median([r['native_success_policy_count'] for r in good]) if good else None,
            'positive_target_count_median':statistics.median([r['native_success_target_count'] for r in good]) if good else None,
            'positive_target_fraction_median':statistics.median([r['native_success_target_count']/r['native_success_policy_count'] for r in good]) if good else None,
            'first_response_chars_before_first_Action_median':statistics.median([r['first_response_chars_before_first_Action'] for r in rows if r['first_response_chars_before_first_Action'] is not None]) if any(r['first_response_chars_before_first_Action'] is not None for r in rows) else None}
        tasks[key].append(entry);features.append(entry)
    report['iterations'].append({'iteration':step,'task_features':features})

import torch
from verl import DataProto
sources=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(r['input']) for r in sources])
uid_map={}
for group in report['iterations'][80]['task_features']:
    uid_map[group['task_key']]=set()
for i,r in enumerate([json.loads(s) for s in Path(original['rows'][80]['dump_path']).read_bytes().splitlines()]):
    key=hashlib.sha256(r['output'].split('\nassistant\n',1)[0].encode()).hexdigest()
    uid_map[key].add(str(data.non_tensor_batch['uid'][i]))
report['native81_UID_calibration']={'decoded_task_states':len(uid_map),'one_uid_per_decoded_state':all(len(v)==1 for v in uid_map.values()),
    'distinct_UIDs':len(set().union(*uid_map.values()))}
assert report['native81_UID_calibration']['one_uid_per_decoded_state']
assert report['native81_UID_calibration']['distinct_UIDs']==32
report['unique_task_states']=len(tasks)
report['matched_tasks']=[]
keys=['success_count','first_response_empty_Thought_fraction','positive_policy_count_median',
      'positive_target_count_median','positive_target_fraction_median','first_response_chars_before_first_Action_median']
for key,observations in tasks.items():
    early=[r for r in observations if r['iteration']<=30]
    late=[r for r in observations if r['iteration']>=61]
    if not early or not late:continue
    values={}
    for feature in keys:
        a=[r[feature] for r in early if r[feature] is not None]
        b=[r[feature] for r in late if r[feature] is not None]
        if a and b:values[feature]={'early':statistics.median(a),'late':statistics.median(b),'difference':statistics.median(b)-statistics.median(a)}
    report['matched_tasks'].append({'task_key':key,'early_iterations':[r['iteration'] for r in early],
        'late_iterations':[r['iteration'] for r in late],'features':values})
report['matched_task_summary']={}
for feature in keys:
    values=[r['features'][feature] for r in report['matched_tasks'] if feature in r['features']]
    report['matched_task_summary'][feature]={'n':len(values),
        'early_median_across_tasks':statistics.median(r['early'] for r in values),
        'late_median_across_tasks':statistics.median(r['late'] for r in values),
        'paired_difference_median':statistics.median(r['difference'] for r in values),
        'increased':sum(r['difference']>0 for r in values),'decreased':sum(r['difference']<0 for r in values)}
assert not torch.cuda.is_initialized()
path=OUT/'matched-task-changes-v7.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
