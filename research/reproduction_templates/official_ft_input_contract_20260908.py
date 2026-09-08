import os,sys,json,hashlib
from pathlib import Path
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
sys.path[:0]=['${ARTIFACT_ROOT}/codex_official_ft_entry_20260908_v2/deps','${ARTIFACT_ROOT}/codex_official_ft_entry_20260908_v1/deps','${PRIVATE_MOUNT_PATH}']
from transformers import AutoTokenizer
from flashtrace.improved import keep_token_indices
import torch
root=Path('${PRIVATE_MOUNT_PATH}')
cp='${PRIVATE_MOUNT_PATH}'
tok=AutoTokenizer.from_pretrained(cp,local_files_only=True);tok.pad_token=tok.eos_token
record=json.loads(Path('${FLASHTRACE_ROOT}/exp/exp2/data/niah_mq_q2.jsonl').read_text().splitlines()[0])
old=json.loads(Path('${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1/results.json').read_bytes())['cases'][0]
target=record['target'];gen=tok(target,add_special_tokens=False)['input_ids']+[tok.eos_token_id]
old_text=tok.apply_chat_template([{'role':'user','content':'Context: '+record['prompt']+'\n\n\nQuery: '}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
texts={'historical075':old_text,'current_public_chat_true':tok.apply_chat_template([{'role':'user','content':record['prompt']}],tokenize=False,add_generation_prompt=True),'current_exp2_default':record['prompt']}
r={'model_loads':0,'model_forwards':0,'new_quality_queries':0,'modes':{},'diagnostic':'Tokenizer-only input contract audit, not an attribution or model implementation.'}
for mode,text in texts.items():
 ids=tok(text,add_special_tokens=False)['input_ids']+gen
 r['modes'][mode]={'prompt_length':len(ids)-len(gen),'target_length':len(gen),'total_length':len(ids),'input_ids_sha256':hashlib.sha256(torch.tensor(ids,dtype=torch.long).numpy().tobytes()).hexdigest()}
assert r['modes']['historical075']['input_ids_sha256']==old['input_metadata']['input_ids_sha256']
new=tok(record['prompt'],add_special_tokens=False,return_offsets_mapping=True)
labels=[record['prompt'][a:b] for a,b in new['offset_mapping']]
gold=sorted({j for j,(a,b) in enumerate(new['offset_mapping']) for needle in record['metadata']['needle_spans'] if a<needle['span'][1] and b>needle['span'][0]})
keep=keep_token_indices(labels)
r.update(current_user_token_count=len(labels),current_gold=gold,current_keep=keep,
 current_gold_equals_historical=(gold==old['mapping']['gold_user_token_indices']),
 current_keep_equals_historical=(keep==old['mapping']['keep_local_indices']),
 current_prompt_tokens_equal_historical_ids_after_first=(new['input_ids'][1:]==tok(' '+record['prompt'],add_special_tokens=False)['input_ids'][1:]))
Path('/tmp/qwen35_official_input_contract_20260908.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k not in ['current_gold','current_keep']}))
