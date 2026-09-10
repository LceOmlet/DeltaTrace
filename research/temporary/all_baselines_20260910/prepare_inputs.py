"""Bind all baseline inputs and retrieval candidates to verified DT/FT records."""
import argparse
import json
from pathlib import Path
from common import HERE,ROOT,OLD,HOTPOT,TASKS,sha,byte_sha,previous_cases

def write(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    a=p.parse_args()
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.tokenizer));old,origins=previous_cases()
    release=json.loads((ROOT/'experiments/official/protocol.json').read_bytes())
    hv=json.loads((HOTPOT/'verification.json').read_bytes());assert hv['status']=='passed'
    for path,digest in hv['output_sha256'].items():assert sha(HOTPOT/path)==digest
    hc={c['index']:c for c in json.loads((HOTPOT/'candidates.json').read_bytes())['cases']}
    hl={c['index']:c for c in json.loads((HOTPOT/'labels.json').read_bytes())['cases']}
    inputs=[];candidates=[];labels=[];counts={}
    for task in TASKS:
        path=a.cache/(task+'.jsonl');assert sha(path)==release['tasks'][task]['cache_sha256']
        cache=[json.loads(l) for l in path.read_bytes().splitlines()];counts[task]=len(cache)
        assert len(cache)==(48 if task=='hotpotqa_long' else 100)
        for index,item in enumerate(cache):
            row,_=old[task,index];enc=tok.encode(item['target'],add_special_tokens=False)
            lo,hi=item['indices_to_explain'];answer=item['target'][enc.offsets[lo][0]:enc.offsets[hi][1]]
            target=item['target'] if task=='hotpotqa_long' else answer
            assert row['target']==target and row['target_mode']==('full' if task=='hotpotqa_long' else 'answer_only')
            ids=row['input_ids'];gen=tok.encode(target+tok.id_to_token(ids[-1]),add_special_tokens=False)
            assert gen.ids==ids[row['prompt_length']:]
            text_tokens=[(target+tok.id_to_token(ids[-1]))[lo:hi] for lo,hi in gen.offsets]
            weights=[float(j<len(gen.ids)-1 and text.strip() not in ('',',','.')) for j,text in enumerate(text_tokens)]
            assert weights==row['target_weights'] and sum(weights)>0
            inputs.append(dict(dataset=task,index=index,prompt=item['prompt'],target=target,target_mode=row['target_mode'],
                input_ids=ids,input_sha256=row['input_sha256'],prompt_length=row['prompt_length'],target_length=row['target_length'],
                user_positions=row['user_positions'],target_weights=weights,original_target_sha256=byte_sha(item['target'].encode()),
                original_answer_span=[lo,hi],references=row['references']))
            promptenc=tok.encode(' '+item['prompt'],add_special_tokens=False)
            if task=='hotpotqa_long':
                c=hc[index];assert c['old_keep']==row['keep']
                candidates.append(dict(dataset=task,index=index,**{k:c[k] for k in ('units','all_groups','eligible_groups','all_body_tokens')}))
                labels.append(dict(dataset=task,index=index,official_keys=hl[index]['official_keys'],gold_all_body_tokens=hl[index]['gold_all_body_tokens']))
            else:
                candidates.append(dict(dataset=task,index=index,keep=row['keep'],offsets=promptenc.offsets,source_span=row['source_span']))
                labels.append(dict(dataset=task,index=index,gold=row['gold']))
    assert len(inputs)==len(candidates)==len(labels)==448
    write('inputs.json',dict(cases=inputs))
    write('candidates.json',dict(cases=candidates))
    write('labels.json',dict(cases=labels))
    receipt=dict(status='verified_448_inputs_match_saved_dt_ft',counts=counts,case_count=448,
        full_analysis_sha256=sha(OLD/'full_recall/analysis.json'),hotpot_analysis_sha256=sha(HOTPOT/'analysis.json'),
        hotpot_verification_sha256=sha(HOTPOT/'verification.json'),tokenizer_sha256=sha(a.tokenizer),
        cache_sha256={t:sha(a.cache/(t+'.jsonl')) for t in TASKS},origins=origins,
        file_sha256={n:sha(HERE/n) for n in ('inputs.json','candidates.json','labels.json')},builder_sha256=sha(Path(__file__)))
    (HERE/'input_audit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k in ('status','case_count','counts')},indent=2))

if __name__=='__main__':main()
