"""Independently verify Qwen3.5 recovery tokenization without loading a model."""
import argparse, json, hashlib, math, sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ids_sha=lambda x:hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    env=json.loads((a.run_root/'environment_dynamic.json').read_bytes())['qwen35']
    prepared_path=a.run_root/'receipts/paper_recovery_inputs.json'
    prepared=json.loads(prepared_path.read_bytes())
    code=a.run_root/'repo_dynamic'
    protocol=json.loads((code/'experiments/qwen35_comparison/paper_recovery/protocol.json').read_bytes())
    sys.path.insert(0,str(code/'experiments/official'))
    from evidence_protocol import source_span, select_source_tokens, reference_token_ids
    from hotpot_retrieval_v3 import all_token_groups
    from hotpot_evidence import token_groups
    tok=Tokenizer.from_file(str(Path(env['checkpoint'])/'tokenizer.json'))
    assert sha(Path(env['checkpoint'])/'tokenizer.json')==prepared['tokenizer_sha256']==env['checkpoint_config_sha256']['tokenizer.json']
    caches={}
    for task in protocol['tasks']:
        cache=Path(env['official_root'])/'exp/exp2/data'/f'{task}.jsonl'
        assert sha(cache)==protocol['tasks'][task]['cache_sha256']
        caches[task]=[json.loads(s) for s in cache.read_text().splitlines()]
    assert prepared['protocol_sha256']==sha(code/'experiments/qwen35_comparison/paper_recovery/protocol.json')
    assert prepared['environment_sha256']==sha(a.run_root/'environment_dynamic.json')
    receipt=[];body_total=0
    for r in prepared['cases']:
        prompt=tok.encode(r['formatted_prompt'],add_special_tokens=False)
        source=tok.encode(' '+r['prompt'],add_special_tokens=False)
        # EOS is the final token recorded in the actual input, not a hard-coded model ID.
        eos=r['input_ids'][-1]
        target_text=r['target']+tok.decode([eos],skip_special_tokens=False)
        target=tok.encode(target_text,add_special_tokens=False)
        assert prompt.ids+target.ids==r['input_ids']
        assert len(prompt.ids)==r['prompt_length'] and len(target.ids)==r['target_length']
        assert ids_sha(r['input_ids'])==r['input_sha256']
        positions=r['user_positions']
        assert len(positions)==len(source.ids)
        keep=[i for i,t in enumerate(source.ids) if tok.decode([t],skip_special_tokens=False).strip() not in ('',',','.')]
        assert keep==r['author_keep']
        original=caches[r['dataset']][r['index']]
        gold_spans=[(x['span'][0]+1,x['span'][1]+1) for x in original['metadata']['needle_spans']]
        gold=[i for i,(s,e) in enumerate(source.offsets) if e>s and any(s<hi and e>lo for lo,hi in gold_spans)]
        assert gold==r['gold']
        span=source_span(r['dataset'],r['prompt'])
        assert span==r['source_span']
        eligible=select_source_tokens(span,source.offsets,keep,gold)
        assert eligible==r['keep']
        assert all(r['input_ids'][positions[j]]==source.ids[j] for j in eligible)
        weights=[float(j<len(target.ids)-1 and target_text[lo:hi].strip() not in ('',',','.')) for j,(lo,hi) in enumerate(target.offsets)]
        assert weights==r['target_weights']
        assert [j for j,w in enumerate(weights) if w]==r['target_offsets']
        reference=reference_token_ids(r['input_ids'],[positions[j] for j in keep],eos)
        assert reference==r['reference_ids'] and ids_sha(reference)==r['reference_sha256']
        assert reference[r['prompt_length']:]==r['input_ids'][r['prompt_length']:]
        fraction=protocol['tasks'][r['dataset']]['fraction']
        if r['dataset']=='hotpotqa_long':
            groups=all_token_groups(' '+r['prompt'],source.offsets,r['units'])
            assert groups==r['all_groups']
            assert token_groups(' '+r['prompt'],source.offsets,eligible,r['units'])==r['eligible_groups']
            body=sorted(j for u,g in zip(r['units'],groups) if u['kind']=='sentence' for j in g)
            assert body==r['all_body_tokens']
            assert all(r['input_ids'][positions[j]]==source.ids[j] for j in body)
            assert len(body)==len(set(body));body_total+=len(body)
            assert r['target']==original['target']
            n=len(body)
        else:
            assert r['target_mode']=='answer_only' and r['target'] in original['target']
            n=len(eligible)
        receipt.append(dict(dataset=r['dataset'],index=r['index'],input_sha256=r['input_sha256'],
            reference_sha256=r['reference_sha256'],target_tokens=len(target.ids),target_seeds=sum(weights),
            source_token_count=n,budget=math.ceil(n*fraction)))
    assert len(receipt)==448
    result=dict(status='passed_independent_cpu_tokenization_audit',cases=448,model_loads=0,model_calls=0,
        input_sha256=sha(prepared_path),tokenizer_sha256=prepared['tokenizer_sha256'],
        hotpot_all_body_tokens=body_total,records=receipt)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()

