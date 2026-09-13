"""Verify preserved model inputs, score vectors and original 20-step curves on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sha=lambda data:hashlib.sha256(data).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
auc=lambda x:float((x.sum()-x[0]/2-x[-1]/2)/(len(x)-1))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.code/'repo_dynamic/experiments/official'))
    from recovery_diagnostics import reported_recovery_diagnostics
    root=a.snapshot/'full_dynamic_with_ifr'
    identity=read(root/'identity.json')
    expected={(x['dataset'],x['index']):x['input_sha256'] for x in read(root/'input_preflight.json')['cases']}
    protocol=read(a.code/'repo_dynamic/experiments/qwen35_comparison/protocol.json')
    source=read(a.snapshot/'repo_dynamic/deltatrace/clean/sources.json')
    assert identity['driver_sha256']==sha((a.code/'repo_dynamic/experiments/official/evaluate.py').read_bytes())
    assert identity['clean_sources_sha256']==sha((a.snapshot/'repo_dynamic/deltatrace/clean/sources.json').read_bytes())
    assert identity['score_views_sha256']==sha((a.code/'repo_dynamic/experiments/official/score_views.py').read_bytes())
    assert identity['protocol_sha256']==sha((a.code/'repo_dynamic/experiments/qwen35_comparison/protocol.json').read_bytes())
    assert identity['environment_sha256']==sha((a.code/'environment_dynamic.json').read_bytes())
    tasks=[];cases=0;curves=0;deletion_inputs=0;proofs=0;signed_runs=0;recalls=0
    for file in sorted(root.glob('*/results.json')):
        report=read(file)
        task=file.parent.name
        assert report['status']=='complete' and report['generation_calls']==0
        assert report['family']=='qwen35' and report['ft_source']=='live'
        assert report['DT_score_view']=='signed-rise' and report['evaluation_protocol']=='released-v1'
        assert {r['index'] for r in report['cases']}==set(range(protocol['tasks'][task]['count']))
        for name in ('driver_sha256','clean_sources_sha256','score_views_sha256','environment_sha256'):
            assert report[name]==identity[name],(task,name)
        vectors_path=file.with_name('vectors.npz')
        assert report['vectors_sha256']==sha(vectors_path.read_bytes())
        with np.load(vectors_path,allow_pickle=False) as v:
            for row in report['cases']:
                key=f"{task}_{row['index']}"
                ids=np.asarray(row['input_ids'],dtype=np.int64)
                assert sha(ids.tobytes())==row['input_sha256']==expected[task,row['index']]
                assert row['status']=='complete' and len(ids)==row['prompt_length']+row['target_length']
                eos=int(ids[-1]);positions=np.asarray(row['user_positions'])
                keep=np.asarray(row['keep'])
                assert len(keep)==len(set(keep.tolist())) and set(keep)<=set(range(len(positions)))
                signed=v[key+'_DT_signed_full']
                positive=v[key+'_DT_positive_prompt']
                assert signed.shape==ids.shape and np.isfinite(signed).all()
                assert np.array_equal(positive,np.maximum(signed[positions].astype(np.float32),0))
                score_vectors={'DT_positive':positive,'DT_signed':signed[positions].astype(np.float32),
                               'FT_K1':v[key+'_FT_K1_prompt'],'ifr-tokenwise':v[key+'_ifr-tokenwise_prompt']}
                endpoint_pairs=[]
                for method,curve in row['metrics'].items():
                    if 'scores' not in curve:continue
                    scores=np.asarray(curve['scores'],dtype=np.float64)
                    normalized=np.minimum.accumulate(np.clip((scores-scores[-1])/abs(scores[0]-scores[-1]),0,1))
                    density=np.asarray(curve['density'])
                    alignment=np.abs(normalized-density)
                    corrected=np.clip(normalized+alignment,0,1)
                    if corrected.max()==corrected.min():
                        corrected=np.linspace(1,0,len(scores))
                    else:corrected=(corrected-corrected.min())/(corrected.max()-corrected.min())
                    assert len(scores)==min(20,len(keep))+1
                    for field,value in [('normalized_model_response',normalized),('alignment_penalty',alignment),('corrected_scores',corrected)]:
                        assert np.allclose(curve[field],value,rtol=0,atol=1e-12),(key,method,field)
                    assert abs(curve['rise']-auc(normalized))<1e-12
                    assert abs(curve['mas']-auc(corrected))<1e-12
                    assert abs(curve['rise_plus_ap']-auc(normalized+alignment))<1e-12
                    endpoint_pairs.append(scores[[0,-1]])
                    weights=np.asarray(score_vectors[method],dtype=np.float32)
                    base,rem=divmod(len(keep),len(scores)-1)
                    previous=set()
                    for step,(deleted,digest) in enumerate(zip(curve['deleted_user_indices'],curve['actual_input_hashes'])):
                        selected=set(deleted)
                        assert previous<=selected<=set(keep.tolist())
                        actual=ids.copy();actual[positions[deleted]]=eos
                        assert sha(actual.tobytes())==digest
                        assert np.array_equal(actual[row['prompt_length']:],ids[row['prompt_length']:])
                        # Original EOS positions can be deleted without changing an input.
                        deleted_budget=step*base+min(step,rem)
                        already_eos={j for j in keep if ids[positions[j]]==eos}
                        assert len(selected)<=deleted_budget<=len(selected|already_eos)
                        remaining=set(keep.tolist())-selected-already_eos
                        if selected and remaining:
                            assert weights[list(selected)].min()>=weights[list(remaining)].max(),(key,method,step,'rank')
                        previous=selected
                        deletion_inputs+=1
                    assert previous=={j for j in keep if ids[positions[j]]!=eos}
                    if method=='DT_signed':
                        assert curve['MAS_is_valid_for_this_view'] is False
                    curves+=1
                assert all(np.array_equal(endpoint_pairs[0],x) for x in endpoint_pairs[1:]),(key,'endpoint mismatch')
                metric=row['metrics']['DT']
                assert metric['views']=={'rise':'signed','mas':'positive_part','needle':'positive_part'}
                assert metric['mas']==row['metrics']['DT_positive']['mas']
                assert metric['needle']==row['metrics']['DT_positive']['needle']
                proof=metric['signed_RISE_reuse_proof']
                if proof is None:
                    assert metric['rise']==row['metrics']['DT_signed']['rise']
                    signed_runs+=1
                else:
                    assert metric['rise']==row['metrics']['DT_positive']['rise']
                    w=score_vectors['DT_signed'][keep];nz=w[w>0]
                    assert len(nz)==len(np.unique(nz))
                    zero=proof['zero_step']
                    assert row['metrics']['DT_positive']['normalized_model_response'][zero]==0
                    assert proof['positive_count']==len(nz) and proof['deletion_budget_at_zero']<=len(nz)
                    assert all(score_vectors['DT_signed'][j]>0 for j in row['metrics']['DT_positive']['deleted_user_indices'][zero])
                    proofs+=1
                if row['gold']:
                    for method in ('DT_positive','FT_K1','ifr-tokenwise'):
                        curve=row['metrics'][method]
                        d=reported_recovery_diagnostics(score_vectors[method],keep,row['gold'],curve['needle'])
                        assert d==curve['needle_diagnostics']
                        recalls+=1
                    d=reported_recovery_diagnostics(v[key+'_FT_K3_prompt'],keep,row['gold'],row['FT_K3_needle'])
                    assert d==row['FT_K3_needle_diagnostics']
                    recalls+=1
                else:
                    assert row['metrics']['DT']['needle'] is None and task in ('math','morehopqa')
                cases+=1
        tasks.append(dict(dataset=task,cases=len(report['cases']),results_sha256=sha(file.read_bytes()),
                          vectors_sha256=sha(vectors_path.read_bytes())))
    out=dict(status='passed_completed_snapshot_audit',cases=cases,tasks=tasks,
        complete_full_benchmark=cases==1243,curves_recomputed=curves,deletion_inputs_rehashed=deletion_inputs,
        signed_RISE_direct_cases=signed_runs,signed_RISE_proven_reuse_cases=proofs,
        raw_recall_records_verified=recalls,
        note='VT/HotpotQA raw token Recall above is verified only as a legacy diagnostic; it is not the paper recovery result.')
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='tasks'},indent=2))
if __name__=='__main__':main()

