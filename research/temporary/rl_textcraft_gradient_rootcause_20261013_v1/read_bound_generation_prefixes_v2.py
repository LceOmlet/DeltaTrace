"""Use the actual TextCraft tokenizer override before auditing saved prefixes.

The model's on-disk Qwen3.5 template is not the task's effective template.
This is CPU-only inspection of the accepted task owner and saved native IDs.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import torch
from transformers import AutoTokenizer
from verl import DataProto

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
source=json.loads((RUN/'source.json').read_bytes())
entry=Path(source['entry'])
spec=importlib.util.spec_from_file_location('bound_task_env',entry/'textcraft_environment_entry.py')
owner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner)
tok=AutoTokenizer.from_pretrained(launch['options']['actor_rollout_ref.model.path'],local_files_only=True,trust_remote_code=True)
unconfigured_hash=hashlib.sha256(tok.chat_template.encode()).hexdigest()
owner.configure_textcraft_tokenizer(tok)
original_schemas=Path(launch['options']['+env.textcraft']['owner_root'])/'AgentGym-RL/verl/workers/rollout/schemas.py'
spec=importlib.util.spec_from_file_location('bound_task_schemas',original_schemas)
schemas=importlib.util.module_from_spec(spec);sys.modules[spec.name]=schemas;spec.loader.exec_module(schemas)
blocks=re.compile(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>',re.S)
rows=[]
for rank in (0,1):
    src=json.loads((OUT/f'dt/rank{rank}-capture.json').read_bytes())
    data=DataProto.load_from_disk(src['input'])
    # Four saved trajectories per rank are enough to check this code path;
    # any discrepancy is retained with exact prefix IDs instead of repaired
    # by a new renderer or an invented tolerance.
    for row_idx in (0,1,2,3):
        ids=data.batch['input_ids'][row_idx][data.batch['attention_mask'][row_idx].bool()].tolist()
        text=tok.decode(ids,skip_special_tokens=False)
        messages=[];turns=[]
        for m in blocks.finditer(text):
            role,content=m.groups()
            if role=='assistant' and len(messages)>=4:
                original=schemas.RolloutHandler.__new__(schemas.RolloutHandler)
                original.messages=[schemas.Message(role=x['role'],content=x['content']) for x in messages]
                generated=original.get_generation_prompt(tok)
                prefix_text=text[:m.start()]+'<|im_start|>assistant\n'
                saved=tok.encode(prefix_text,add_special_tokens=False)
                # Decoder/encoder agreement is independently checked below.
                prefix_len=len(saved)
                roundtrip=saved==ids[:prefix_len]
                first_difference=next((i for i,(a,b) in enumerate(zip(saved,generated)) if a!=b),None)
                turns.append({'turn':len(turns),'saved_prefix_length':len(saved),
                    'owner_generation_prefix_length':len(generated),'saved_ID_roundtrip_exact':roundtrip,
                    'owner_generation_equals_native_prefix':generated==saved,
                    'first_difference':first_difference})
            messages.append({'role':role,'content':content})
        rows.append({'rank':rank,'native_row':row_idx,'turns':turns})
report={'scope':__doc__,'model_GPU_calls':0,'effective_task_template':tok.chat_template,
    'effective_template_SHA256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),
    'unconfigured_model_template_SHA256':unconfigured_hash,
    'configuration_owner':{'path':str(entry/'textcraft_environment_entry.py'),
        'SHA256':hashlib.sha256((entry/'textcraft_environment_entry.py').read_bytes()).hexdigest()},
    'template_artifact':{'path':str(entry/'textcraft_qwen_template.json'),
        'SHA256':hashlib.sha256((entry/'textcraft_qwen_template.json').read_bytes()).hexdigest()},
    'native_schema':{'path':str(original_schemas),'SHA256':hashlib.sha256(original_schemas.read_bytes()).hexdigest()},
    'rows':rows,'limits':'Eight trajectories, actual owner template, retained complete assistant turns only; not universal historical identity.'}
assert not torch.cuda.is_initialized()
path=OUT/'bound-generation-prefixes-v2.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
