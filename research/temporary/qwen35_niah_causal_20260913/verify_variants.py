"""Independent CPU audit of native scores, deletion ranks, MAS and Recall bounds."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from verify_causal import bf16_sum

def auc(x):return float((x.sum()-x[0]/2-x[-1]/2)/(len(x)-1))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    status=json.loads((a.run/'status.json').read_bytes());assert status['status']=='complete'
    inputs=curves=recalls=ambiguous=0;max_density_error=0.0
    for meta in status['cases']:
        file=a.run/meta['path']/'results.json';assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['results_sha256']
        c=json.loads(file.read_bytes());assert c['status']=='complete'
        ids=np.array(c['input_ids'],dtype=np.int64);pos=np.array(c['user_positions']);keep=np.array(c['keep']);gold=set(c['gold'])&set(keep)
        with np.load(file.with_name('vectors.npz'),allow_pickle=False) as v,np.load(file.with_name('token_logprobs.npz'),allow_pickle=False) as lp:
            for name in ('vectors','token_logprobs'):assert hashlib.sha256(file.with_name(name+'.npz').read_bytes()).hexdigest()==c[name+'_sha256']
            for key,e in c['evaluations'].items():
                actual=ids.copy();actual[pos[e['deleted']]]=ids[-1]
                assert hashlib.sha256(actual.tobytes()).hexdigest()==e['input_sha256']
                assert np.array_equal(actual[c['prompt_length']:],ids[c['prompt_length']:])
                assert bf16_sum(lp[key])==e['score'];inputs+=1
            for label,curve in c['curves'].items():
                method,view=label.rsplit('_',1);w=v[method].astype(np.float64)
                if view=='positive':w=np.maximum(w,0)
                path=[c['evaluations'][key] for key in curve['evaluations']];assert len(path)==21
                scores=np.array([e['score'] for e in path]);y=np.minimum.accumulate(np.clip((scores-scores[-1])/abs(scores[0]-scores[-1]),0,1))
                assert np.allclose(y,curve['normalized'],rtol=0,atol=1e-12);assert abs(auc(y)-curve['rise'])<1e-12
                previous=set()
                for step,e in enumerate(path):
                    deleted=set(e['deleted']);remaining=set(keep)-deleted
                    assert previous<=deleted<=set(keep)
                    assert len(deleted)==step*(len(keep)//20)+min(step,len(keep)%20)
                    if deleted and remaining:assert min(w[list(deleted)])>=max(w[list(remaining)])
                    previous=deleted
                if view=='positive':
                    density=np.array([1-w[e['deleted']].sum()/w[keep].sum() for e in path]) if w[keep].sum()>0 else np.linspace(1,0,21)
                    err=float(np.max(np.abs(density-curve['density'])));max_density_error=max(max_density_error,err);assert err<1e-6
                    corrected=np.clip(y+np.abs(y-np.array(curve['density'])),0,1)
                    if corrected.max()==corrected.min():corrected=np.linspace(1,0,21)
                    else:corrected=(corrected-corrected.min())/(corrected.max()-corrected.min())
                    assert abs(auc(corrected)-curve['mas'])<1e-12
                curves+=1
            candidates={name:d['recall'] for name,d in c['methods'].items() if 'recall' in d}
            candidates.update(FT_K1=c['FT_K1']['recall'],FT_K3=c['FT_K3_recall'])
            for method,value in candidates.items():
                if not gold:assert value is None;continue
                w=np.maximum(v[method],0);k=int(math.ceil(.1*len(keep)));cutoff=np.sort(w[keep])[-k]
                above={j for j in keep if w[j]>cutoff};ties={j for j in keep if w[j]==cutoff};slots=k-len(above)
                lo=(len(above&gold)+max(0,slots-len(ties-gold)))/len(gold)
                hi=(len(above&gold)+min(slots,len(ties&gold)))/len(gold)
                assert lo-1e-12<=value<=hi+1e-12
                ambiguous+=int(lo!=hi);recalls+=1
            if 'DT_endpoint_symmetric_full_sequence' in v:
                assert np.array_equal(v['DT_endpoint_symmetric_full_sequence'],.5*(v['DT_original_full_sequence']-v['DT_reversed_endpoints_full_sequence']))
    receipt=dict(status='passed',cases=len(status['cases']),actual_inputs_rehashed=inputs,curves_recomputed=curves,
        recall_records_checked=recalls,recall_records_with_nonunique_cutoff=ambiguous,max_float32_density_rounding_difference=max_density_error)
    (a.run/'independent_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
