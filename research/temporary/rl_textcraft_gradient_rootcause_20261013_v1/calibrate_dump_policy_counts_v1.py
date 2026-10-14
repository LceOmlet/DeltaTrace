"""Check whether existing decoded dumps retain enough information for B4 counts.

Offline analysis only. No training inputs are replaced or generated, and no
model/rollout/DT/update is called. Native retained iteration81 IDs/masks and
saved positive trajectory IDs are the independent references.
"""
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from transformers import AutoTokenizer
from verl import DataProto

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
tok=AutoTokenizer.from_pretrained(launch['options']['actor_rollout_ref.model.path'],trust_remote_code=True,local_files_only=True)
report={'started_unix':time.time(),'scope':__doc__,'model_GPU_calls':0,
    'rollout_directory':launch['options'].get('trainer.rollout_data_dir'),
    'tokenizer_class':type(tok).__name__,'special_tokens':tok.all_special_tokens}
directory=Path(report['rollout_directory'])
dump=[json.loads(s) for s in (directory/'81.jsonl').read_text().splitlines()]
assert len(dump)==256
receipts=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
full=DataProto.concat([DataProto.load_from_disk(s['input']) for s in receipts])
report['dump_keys']=list(dump[0])
pattern=re.compile(r'\n(user|assistant)\n')
suffix_n=len(tok.encode('<|im_end|>',add_special_tokens=False))

def count_from_decoded(text):
    # Role delimiters have known fixed native strings, but the text dump drops
    # special tokens. Ambiguity and truncation are measured below, not assumed
    # absent. This count is never supplied to an actor.
    parts=pattern.split(text)
    policy=0
    if parts[0]:
        policy+=len(tok.encode(parts[0],add_special_tokens=False))+suffix_n
    turns=[]
    for role,content in zip(parts[1::2],parts[2::2]):
        if role=='assistant':
            n=len(tok.encode(content,add_special_tokens=False))+suffix_n
            policy+=n
            turns.append(n)
    return policy,turns

rows=[]
for i,r in enumerate(dump):
    ids=full.batch['responses'][i]
    decoded=tok.decode(ids,skip_special_tokens=True)
    assert decoded==r['output'],('dump order/ID mismatch',i)
    n,turns=count_from_decoded(decoded)
    native=int(full.batch['response_mask'][i].sum())
    response_valid=int(full.batch['attention_mask'][i,-len(ids):].sum())
    rows.append({'row':i,'score':r['score'],'native_policy_count':native,
        'decoded_policy_count':n,'difference':n-native,'native_response_count':response_valid,
        'assistant_turn_counts':turns,'decoded_SHA256':hashlib.sha256(decoded.encode()).hexdigest()})
report['native81_calibration']={'rows':rows,'exact_rows':sum(r['difference']==0 for r in rows),
    'total_native':sum(r['native_policy_count'] for r in rows),
    'total_decoded':sum(r['decoded_policy_count'] for r in rows),
    'differences':sorted(set(r['difference'] for r in rows))}
report['sample_texts']=[{'row':i,'score':dump[i]['score'],'head':dump[i]['output'][:750],
    'tail':dump[i]['output'][-350:]} for i in [0,1,2,3,128,129]]
report['finished_unix']=time.time()
assert not torch.cuda.is_initialized()
path=OUT/'dump-policy-count-calibration-v1.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
