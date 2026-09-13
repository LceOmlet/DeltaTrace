"""Inspect verified native token roles in the fixed original screening cases."""
import hashlib,json,re
from pathlib import Path
import numpy as np

OWN=Path(__file__).resolve().parent;BASE=OWN.parents[3]
TOKENIZER=BASE/'audit/qwen35_niah_tokenizer_20260913.json'
t=json.loads(TOKENIZER.read_bytes());vocab={v:k for k,v in t['model']['vocab'].items()}
vocab.update({r['id']:r['content'] for r in t['added_tokens']})
bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
for x in range(256):
    if x not in bs:bs.append(x);cs.append(256+n);n+=1
decoder={chr(c):x for x,c in zip(bs,cs)}
def token_bytes(i):return bytes(decoder.get(c,ord(c)) for c in vocab[i])

rows=[];examples=[]
for file in sorted((OWN/'raw/stage_b_v2').glob('niah_*/results.json')):
    c=json.loads(file.read_bytes());a=json.loads((OWN/'raw/stage_a_v1'/file.parent.name/'results.json').read_bytes())
    task=c['dataset'];data=BASE/'audit/published_flashtrace/table1-data-v1/extracted/data'/(task+'.jsonl')
    source=[json.loads(line) for line in data.read_text(encoding='utf-8').splitlines()][c['index']]
    parts=[token_bytes(c['input_ids'][j]) for j in c['user_positions']]
    text=' '+source['prompt'];assert b''.join(parts)==text.encode('utf-8')
    ends=np.cumsum([len(x) for x in parts]);starts=np.r_[0,ends[:-1]];keep=set(c['keep'])
    def overlap(start,end):
        left=len(text[:start].encode('utf-8'));right=len(text[:end].encode('utf-8'))
        return {j for j in keep if starts[j]<right and ends[j]>left}
    query=set(a['query']);gold=set(c['gold'])&keep;qkeys=set();nkeys=set();nvalues=set()
    for needle in source['metadata']['needle_spans']:
        key=needle['key'];value=needle['answer'];left,right=[x+1 for x in needle['span']]
        for m in re.finditer(re.escape(key),text):
            hit=overlap(m.start(),m.end());qkeys|=hit&query
            if left<=m.start() and m.end()<=right:nkeys|=hit&gold
        for m in re.finditer(re.escape(value),text):
            if left<=m.start() and m.end()<=right:nvalues|=overlap(m.start(),m.end())&gold
    assert query.isdisjoint(gold) and qkeys and nkeys and nvalues
    groups=dict(query_key=qkeys,query_other=query-qkeys,needle_value=nvalues,needle_key=nkeys-nvalues,
        needle_other=gold-nvalues-nkeys,remaining=keep-query-gold)
    assert set.union(*groups.values())==keep
    assert sum(map(len,groups.values()))==len(keep)
    with np.load(file.with_name('vectors.npz'),allow_pickle=False) as vectors:
        for method,label in [('DT_original','DT_original_signed'),('DT_gdn_symmetric','DT_gdn_symmetric_signed'),('FT_K1','FT_K1_positive')]:
            weights=vectors[method].astype(np.float64);total=float(weights[c['keep']].sum())
            for group,indices in groups.items():
                if not indices:continue
                row=dict(dataset=task,index=c['index'],method=method,group=group,count=len(indices),signed_credit_fraction=float(weights[list(indices)].sum()/total))
                for step in (1,2,4,6):
                    eid=c['curves'][label]['evaluations'][step];deleted=set(c['evaluations'][eid]['deleted'])
                    row[f'fraction_deleted_{step}']=len(deleted&indices)/len(indices)
                rows.append(row)
            # Example top query tokens are descriptive only; examples are fixed, not selected for a favorable result.
            if c['index']==11:
                order=sorted(query,key=lambda j:-weights[j])[:12]
                examples.append(dict(dataset=task,method=method,query_top_tokens=[dict(token=parts[j].decode('utf-8',errors='replace'),score=float(weights[j]),role='key' if j in qkeys else 'other') for j in order]))
means=[]
for task in sorted({x['dataset'] for x in rows}):
    for method in ['DT_original','DT_gdn_symmetric','FT_K1']:
        for group in groups:
            rr=[x for x in rows if (x['dataset'],x['method'],x['group'])==(task,method,group)]
            means.append(dict(dataset=task,method=method,group=group,n=len(rr),**{k:float(np.mean([x[k] for x in rr])) for k in
                ['count','signed_credit_fraction','fraction_deleted_1','fraction_deleted_2','fraction_deleted_4','fraction_deleted_6']}))
(OWN/'token_role_diagnosis.json').write_text(json.dumps(dict(cases=12,tokenizer_sha256=hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
    note='Exact source-text and actual input token alignment; descriptive token-role analysis of the selected development cases, not an independent causal ablation.',rows=rows,means=means,examples=examples),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for r in means:
    if r['dataset']=='niah_mq_q4' and r['group']!='remaining':print({k:r[k] for k in ['method','group','signed_credit_fraction','fraction_deleted_2','fraction_deleted_4']})
