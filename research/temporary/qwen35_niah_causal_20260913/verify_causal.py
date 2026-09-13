"""Independent saved-input and token-response verification for stage A."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def bf16_sum(x):
    bits=np.array(np.float32(np.asarray(x,dtype=np.float64).sum())).view(np.uint32)
    rounded=((int(bits)+0x7fff+((int(bits)>>16)&1))&0xffff0000)
    return float(np.array(rounded,dtype=np.uint32).view(np.float32))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    status=json.loads((a.run/'status.json').read_bytes());assert status['status']=='complete'
    curves=0;inputs=0;sums=0
    for meta in status['cases']:
        file=a.run/meta['path']/'results.json';assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['results_sha256']
        c=json.loads(file.read_bytes());assert c['status']=='complete'
        ids=np.array(c['input_ids'],dtype=np.int64);pos=np.array(c['user_positions']);keep=set(c['keep'])
        assert set(c['query']).isdisjoint(c['nonquery']) and set(c['query'])|set(c['nonquery'])==keep
        with np.load(file.with_name('token_logprobs.npz'),allow_pickle=False) as lp:
            assert set(lp.files)==set(c['evaluations'])
            for key,e in c['evaluations'].items():
                changed=ids.copy();changed[pos[e['deleted']]]=ids[-1]
                assert hashlib.sha256(changed.tobytes()).hexdigest()==e['input_sha256']
                assert np.array_equal(changed[c['prompt_length']:],ids[c['prompt_length']:])
                assert set(e['deleted'])<=keep
                v=lp[key];assert len(v)==c['target_length'] and np.isfinite(v).all()
                assert ((v.view(np.uint32)&0xffff)==0).all()
                assert bf16_sum(v)==e['full']
                for target in ('answer','answer_content'):
                    offsets=c['answer_span'] if target=='answer' else c['answer_content']
                    assert bf16_sum(v[offsets])==e[target]
                    assert abs(v[offsets].astype(np.float64).sum()-e[target+'_fp64'])<1e-12
                inputs+=1;sums+=3
        for name,curve in c['curves'].items():
            seq=[c['evaluations'][x] for x in curve['evaluations']]
            assert len(seq)==21 and not seq[0]['deleted']
            for prev,nxt in zip(seq,seq[1:]):assert set(prev['deleted'])<=set(nxt['deleted'])
            if name.endswith('_restore_query'):assert all(set(x['deleted']).isdisjoint(c['query']) for x in seq)
            if name.endswith('_only_query'):assert all(set(x['deleted'])<=set(c['query']) for x in seq)
            curves+=1
    assert len(status['cases'])==12 and inputs==status['model_calls']
    receipt=dict(status='passed',cases=12,curves=curves,actual_inputs_rehashed=inputs,native_bf16_sums_recomputed=sums,
        note='All unique inputs preserve fixed target IDs. Saved BF16 token responses independently reproduce native full and selected-target sums.')
    (a.run/'independent_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
