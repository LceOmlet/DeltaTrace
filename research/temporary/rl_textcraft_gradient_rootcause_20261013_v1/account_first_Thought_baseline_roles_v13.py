"""Inventory observed first-Thought baseline counterevents in the saved85 batches.

Uses original owner Q/V composition and existing decoder span mapping; no
model query, new rollout, new parser, PPO implementation or training change.
"""
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import statistics
import sys
import time

import torch
from transformers import AutoTokenizer
from executed_target_spans import decoded_payload_mask

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
source=json.loads((RUN/'source.json').read_bytes())
owner_path=Path(source['entry'])/'counterfactual.py'
spec=importlib.util.spec_from_file_location('recorded_QV_owner_v13',owner_path)
owner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner)
tokenizer=AutoTokenizer.from_pretrained(launch['options']['actor_rollout_ref.model.path'],local_files_only=True,trust_remote_code=True)
inference=json.loads((OUT/'whitening-temporal-inference-v1.json').read_bytes())['rows']
report={'unix':time.time(),'scope':__doc__,'new_GPU_model_calls':0,'new_training_calls':0,'rows':[],
        'QV_owner':{'path':str(owner_path),'SHA256':hashlib.sha256(owner_path.read_bytes()).hexdigest()},
        'span_mapping_owner':{'path':inspect.getsourcefile(decoded_payload_mask),
            'SHA256':hashlib.sha256(Path(inspect.getsourcefile(decoded_payload_mask)).read_bytes()).hexdigest()},
        'limits':[
            'First Thought is an explicitly labelled decoded-text diagnostic feature, not an environment parser or a training target definition.',
            'Only observed finite batches are counted; absence of a sampled counterevent does not prove zero population probability or an invalid idealized estimator.',
            'Raw V counts/sums do not include state probabilities, native B4 weights or model Jacobians and are not gradient proportions.',
            'Source records end at the final causal target. Later zero-target suffix tokens are not reconstructed.',
            'Zero-reward rows have raw V=0 under the unchanged PLAN. Coverage of positive rows is checked against each original rollout dump.']}
for step,previous in enumerate(inference,1):
    saved={}
    for path in previous['raw_files']:
        for row in torch.load(path,map_location='cpu',weights_only=False)['rows']:
            saved.setdefault(row['traj_uid'],row)
    dump_path=Path(launch['options']['trainer.rollout_data_dir'])/f'{step}.jsonl'
    dumps=[json.loads(line) for line in dump_path.read_bytes().splitlines()]
    positive=sum(row['score']!=0 for row in dumps)
    assert len(saved)==positive,(step,len(saved),positive)
    choices=[];missing=[]
    for uid,row in saved.items():
        policy=row['policy_mask'].bool();target=row['target_mask'].bool()
        locations=policy.nonzero().flatten();start=int(locations[0])
        zeros=(~policy[start:]).nonzero().flatten()
        end=start+int(zeros[0]) if len(zeros) else len(policy)
        response=row['input_ids'][row['prompt_length']:]
        end=min(end,len(response))
        ids=response[start:end].tolist();text=tokenizer.decode(ids)
        match=re.search(r'Thought:\s*(?=\S)',text)
        if match is None:
            missing.append({'UID':uid,'reason':'No first Thought marker in the saved first native policy run.'});continue
        position=match.end()
        mapped,_=decoded_payload_mask(tokenizer,text,ids,[(position,position+1)])
        selected=[i for i,v in enumerate(mapped) if v]
        assert len(selected)==1,(step,uid,selected)
        j=start+selected[0]
        if j>=10240:
            missing.append({'UID':uid,'reason':'Choice lies outside the original retained response cap.'});continue
        d=row['source_log_ratios']
        credit=owner.reward_event_token_credit(d[None,None,:],torch.tensor([[row['reward']]],dtype=torch.float32),
            torch.ones((1,1,len(d)),dtype=torch.bool),policy[None,:],self_target_mask=target[None,None,:])
        token=int(response[j]);piece=tokenizer.decode([token]).strip()
        role=('Action_target' if target[j] else 'Action_source') if piece=='Action' else ('other_target' if target[j] else 'other_source')
        choices.append({'UID':uid,'position':j,'token_id':token,'decoded_piece':piece,'role':role,
                        'raw_Q':float(credit.q_estimates[0,j]),'raw_V':float(credit.v_estimates[0,j]),
                        'raw_A':float(credit.advantages[0,j])})
    roles={}
    for role in ['Action_target','Action_source','other_target','other_source']:
        members=[c for c in choices if c['role']==role]
        roles[role]={'count':len(members),'raw_V_sum':sum(c['raw_V'] for c in members),
                     'raw_V_range':([min(c['raw_V'] for c in members),max(c['raw_V'] for c in members)] if members else None)}
    report['rows'].append({'iteration':step,'original_dump':str(dump_path),'positive_original_rows':positive,
        'positive_saved_UIDs':len(saved),'first_Thought_choice_rows':len(choices),'missing_choices':missing,
        'roles':roles,'choice_token_counts':dict(Counter(c['decoded_piece'] for c in choices)),
        'observed_Action_source_counterevents':[c for c in choices if c['role']=='Action_source'],
        'choice_records':choices,'raw_sources':previous['raw_files']})
report['windows']=[]
for lo,hi in [(1,10),(11,20),(31,40),(51,60),(61,70),(71,80),(81,85)]:
    rows=report['rows'][lo-1:hi]
    report['windows'].append({'iterations':[lo,hi],
        'positive_rows':sum(r['positive_original_rows'] for r in rows),
        'available_first_Thought_choices':sum(r['first_Thought_choice_rows'] for r in rows),
        'role_counts':{role:sum(r['roles'][role]['count'] for r in rows) for role in ['Action_target','Action_source','other_target','other_source']},
        'raw_V_sums_by_role':{role:sum(r['roles'][role]['raw_V_sum'] for r in rows) for role in ['Action_target','Action_source','other_target','other_source']}})
assert not torch.cuda.is_initialized()
report['finished_unix']=time.time()
dest=OUT/'first-Thought-baseline-roles-v13.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
