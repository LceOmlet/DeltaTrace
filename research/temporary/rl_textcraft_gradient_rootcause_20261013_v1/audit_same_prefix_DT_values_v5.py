"""Read native DT Q/V/A at exact shared prefixes; no model or new credit values."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct
import time
import torch
from transformers import AutoTokenizer
from verl import DataProto

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
tokenizer=AutoTokenizer.from_pretrained(launch['options']['actor_rollout_ref.model.path'],local_files_only=True,trust_remote_code=True)
sources=[json.loads((OUT/f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(r['input']) for r in sources])
width=data.batch['responses'].shape[1]
target=torch.zeros_like(data.batch['responses'],dtype=torch.bool)
for i,artifact in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
    p=torch.as_tensor(artifact['retained_response_positions']);keep=p>=0
    target[i,keep]=torch.as_tensor(artifact['target_mask'],dtype=torch.bool)[p[keep]]
groups=defaultdict(list)
for row in range(len(data)):
    ids=data.batch['input_ids'][row].tolist()
    attention=data.batch['attention_mask'][row].bool().tolist()
    policy=data.batch['loss_mask'][row].bool().tolist()
    prompt=len(ids)-width
    h=hashlib.sha256();valid_prefix=[]
    reward=float(data.batch['token_level_rewards'][row].sum())
    for position,(token,active,is_action) in enumerate(zip(ids,attention,policy)):
        if not active:continue
        if is_action:
            j=position-prompt
            assert 0<=j<width
            groups[h.digest()].append({'row':row,'position':j,'token_id':token,
                'target':bool(target[row,j]),'reward':reward,
                'Q':float(data.batch['dt_q_estimates'][row,j]),
                'V':float(data.batch['dt_v_estimates'][row,j]),
                'raw_A':float(data.batch['dt_token_advantages'][row,j]),
                'actor_coefficient':float(data.batch['advantages'][row,j]),
                'native_old_logp':float(data.batch['old_log_probs'][row,j]),
                'prefix_length':len(valid_prefix),'prefix_tail_ids':valid_prefix[-50:]})
        h.update(struct.pack('<I',token));valid_prefix.append(token)
forks=[]
for key,records in groups.items():
    if len({r['token_id'] for r in records})<2 or {r['target'] for r in records}!={False,True}:
        continue
    by_token=defaultdict(list)
    for r in records:by_token[(r['token_id'],r['target'])].append(r)
    branches=[]
    for (token,is_target),a in by_token.items():
        branches.append({'token_id':token,'token':tokenizer.decode([token]),'target':is_target,'n':len(a),
            **{k+'_mean':sum(r[k] for r in a)/len(a) for k in ('reward','Q','V','raw_A','actor_coefficient')},
            'occurrences':[{k:v for k,v in r.items() if k!='prefix_tail_ids'} for r in a]})
    forks.append({'prefix_SHA256':key.hex(),'prefix_length':records[0]['prefix_length'],
        'prefix_tail':tokenizer.decode(records[0]['prefix_tail_ids'],skip_special_tokens=False),
        'observations':len(records),'branches':branches})
forks.sort(key=lambda f:(-f['observations'],f['prefix_length']))
report={'unix':time.time(),'scope':__doc__,'model_GPU_calls':0,'source_iteration':81,
        'input_sources':sources,'exact_prefixes_with_target_source_branches':len(forks),'forks':forks,
        'limits':[
            'These are actual sampled Q/V/A, not conditional population expectations or exact counterfactual values.',
            'Equal rewards in a finite branch sample do not prove equal underlying Q values.',
            'A target/source coefficient difference does not itself prove bias: the conditional expectation of the sampled V and reference support must be analyzed.',
            'This diagnoses DT construction while leaving the fixed VERL owner and all training parameters untouched.',
        ]}
assert not torch.cuda.is_initialized()
path=OUT/'same-prefix-DT-values-v5.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
