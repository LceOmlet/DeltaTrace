"""Decode preserved inputs and classify actual deletions by prompt segment."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
TOKENIZER=BASE/'audit/qwen35_niah_tokenizer_20260913.json'
tok=json.loads(TOKENIZER.read_bytes())
vocab={v:k for k,v in tok['model']['vocab'].items()}
vocab.update({r['id']:r['content'] for r in tok['added_tokens']})
bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
for x in range(256):
    if x not in bs:bs.append(x);cs.append(256+n);n+=1
byte_decoder={chr(c):x for x,c in zip(bs,cs)}
def decode_token(i):return bytes(byte_decoder.get(c,ord(c)) for c in vocab[i])

rows=[];summaries=[]
for path in sorted((BASE/'audit/qwen35_existing_runtime_20260913/full_dynamic_with_ifr').glob('niah_*/results.json')):
    task=path.parent.name
    data=BASE/'audit/published_flashtrace/table1-data-v1/extracted/data'/(task+'.jsonl')
    raw=[json.loads(s) for s in data.read_text(encoding='utf-8').splitlines()]
    report=json.loads(path.read_bytes())
    assert hashlib.sha256(data.read_bytes()).hexdigest()==report['evaluation_settings']['tasks'][task]['cache_sha256']
    for r in report['cases']:
        ex=raw[r['index']]
        parts=[decode_token(r['input_ids'][i]) for i in r['user_positions']]
        body=b''.join(parts)
        assert body==(' '+ex['prompt']).encode('utf-8')
        answer=b''.join(decode_token(i) for i in r['input_ids'][r['prompt_length']:])
        assert answer==(ex['target']+'<|im_end|>').encode('utf-8')
        ends=np.cumsum([len(x) for x in parts]);starts=np.r_[0,ends[:-1]]
        # Fixed task template: final newline begins the retrieval query and answer prefix.
        query_start=body.rfind(b'\nWhat ')+1
        assert query_start>0
        intro_end=body.find(b'\n')
        keep=set(r['keep'])
        groups={'query':{i for i in keep if ends[i]>query_start},
                'intro':{i for i in keep if starts[i]<intro_end}}
        for name in ('query','intro'):
            assert groups[name]
        groups['gold']=set(r['gold'])&keep
        m=r['metrics'];curves={'dt':m.get('DT_signed',m['DT_positive']),'ft':m['FT_K1']}
        row=dict(task=task,index=r['index'],query_tokens=len(groups['query']),intro_tokens=len(groups['intro']))
        for method,curve in curves.items():
            for step in (1,2,4,5,10):
                deleted=set(curve['deleted_user_indices'][step])
                for group,indices in groups.items():
                    row[f'{method}_{group}_fraction_deleted_{step}']=len(deleted&indices)/len(indices)
                    row[f'{method}_{group}_count_deleted_{step}']=len(deleted&indices)
        rows.append(row)
    rr=[x for x in rows if x['task']==task]
    summary=dict(task=task,n=len(rr),means={k:float(np.mean([x[k] for x in rr])) for k in rr[0] if k not in ('task','index')})
    summaries.append(summary)
    print(task,{k:round(v,4) for k,v in summary['means'].items() if k in ('query_tokens','intro_tokens') or ('fraction' in k and k.endswith('_4'))})
assert len(rows)==600
with (OUT/'deletion_segments.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'deletion_segments.json').write_text(json.dumps(dict(cases=600,model_calls=0,
    tokenizer_url='https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/tokenizer.json',
    tokenizer_sha256=hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
    validation='Every decoded user prompt and complete target matches the pinned source text byte for byte. Dataset hashes checked against preserved run settings.',
    query_rule='Tokens overlapping the final newline-What query and answer prefix; includes query syntax and repeated keys.',tasks=summaries),indent=2)+'\n',encoding='utf-8')
