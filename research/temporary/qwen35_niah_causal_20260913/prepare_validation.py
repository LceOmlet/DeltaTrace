"""Frozen counterfactual-key/value construction; run after candidate selection."""
import argparse,copy,hashlib,json,random,re
from pathlib import Path

ADJECTIVES='amber ancient autumn azure brisk bronze calm cedar clear clever cool coral crimson crisp distant eager early gentle golden grand green hidden hollow humble icy ivory jade keen kind lavender lively lunar mellow merry misty modern modest narrow nimble noble northern olive orange pale patient peaceful polar purple quiet rapid red rocky royal rustic sage scarlet silver smooth snowy solar stable steady subtle sunny swift tender tiny violet vivid warm western wild winter wise yellow young'.split()
NOUNS='anchor apple arch arrow atlas badger basket beacon birch blossom bridge brook cabin canyon cedar cherry circle cloud comet compass coral crane creek crown crystal dawn deer delta dune eagle elm falcon feather fern field finch flame forest fountain fox garden gate glacier grove harbor hawk heron hill island ivy kite lake lantern lark leaf maple meadow moon mountain oak ocean orbit otter owl panda path pebble pine planet pond rabbit raven reed ridge river robin sail seal shell shore sparrow spring star stone summit swan tiger tower trail valley wave willow wind wing wolf wood wren'.split()
TASKS=['niah_mq_q2','niah_mq_q4','niah_mq_q8','niah_mv_v2','niah_mv_v4','niah_mv_v8']
SOURCE_INDICES=list(range(5,100,10))
SEED=13720260913

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--candidate-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();receipt=json.loads(a.candidate_receipt.read_bytes())
    assert receipt['status']=='frozen_before_validation' and receipt['candidate']
    rng=random.Random(SEED);cases=[];sources=[];used_keys=set();used_values=set()
    for task in TASKS:
        file=a.source/(task+'.jsonl');records=[json.loads(x) for x in file.read_text(encoding='utf-8').splitlines()]
        sources.append(dict(dataset=task,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        for index in SOURCE_INDICES:
            raw=records[index];metadata=raw['metadata'];old_spans=metadata['needle_spans']
            old_keys=sorted({x['key'] for x in old_spans});old_values=sorted({x['answer'] for x in old_spans})
            replacements={};all_text=raw['prompt']+'\n'+raw['target']
            for key in old_keys:
                while True:
                    fresh=rng.choice(ADJECTIVES)+'-'+rng.choice(NOUNS)
                    if fresh not in all_text and fresh not in used_keys:break
                replacements[key]=fresh;used_keys.add(fresh)
            for value in old_values:
                assert re.fullmatch(r'\d{7}',value)
                while True:
                    fresh=str(rng.randrange(1000000,10000000))
                    if fresh not in all_text and fresh not in used_values:break
                replacements[value]=fresh;used_values.add(fresh)
            pattern=re.compile('|'.join(re.escape(k) for k in sorted(replacements,key=len,reverse=True)))
            def rewrite(s):return pattern.sub(lambda m:replacements[m.group()],s)
            prompt=rewrite(raw['prompt']);target=rewrite(raw['target'])
            def rewrite_object(x):
                if isinstance(x,str):return rewrite(x)
                if isinstance(x,list):return [rewrite_object(v) for v in x]
                if isinstance(x,dict):return {k:rewrite_object(v) for k,v in x.items()}
                return x
            meta=rewrite_object(copy.deepcopy(metadata))
            for k in ('judge_response','length','length_w_model_temp','token_position_answer'):meta.pop(k,None)
            for original,new in zip(old_spans,meta['needle_spans']):
                start,end=original['span'];new['span']=[len(rewrite(raw['prompt'][:start])),len(rewrite(raw['prompt'][:end]))]
                assert prompt[new['span'][0]:new['span'][1]]==rewrite(raw['prompt'][start:end])
                assert new['answer'] in prompt[new['span'][0]:new['span'][1]]
                assert new['key'] in prompt[new['span'][0]:new['span'][1]]
                if 'snippet' in new:assert new['snippet']==prompt[new['span'][0]:new['span'][1]]
            assert meta['boxed_answer'] in target
            assert all(x not in prompt+target for x in replacements)
            assert set(replacements.values())<=set(pattern_value for pattern_value in replacements.values() if pattern_value in prompt+target)
            cases.append(dict(dataset=task,index=10000+index,prompt=prompt,target=target,metadata=meta,
                source_index=index,replacements=replacements,source_prompt_sha256=hashlib.sha256(raw['prompt'].encode()).hexdigest(),
                source_target_sha256=hashlib.sha256(raw['target'].encode()).hexdigest()))
    assert len(cases)==60
    out=dict(version='counterfactual-niah-keys-values-v1',seed=SEED,source_indices=SOURCE_INDICES,
        candidate_receipt_sha256=hashlib.sha256(a.candidate_receipt.read_bytes()).hexdigest(),sources=sources,cases=cases,
        scope='Sixty new key/value assignments on preserved benchmark text templates. Paired exact-input comparisons; no model generation. This tests within-template generalization and is not a new independently sampled natural benchmark.')
    with a.output.open('x',encoding='utf-8') as f:f.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))

if __name__=='__main__':main()
