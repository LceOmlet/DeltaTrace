"""Independently verify fresh positive-view author metrics and retained signed scores."""
import ast,hashlib,json,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace'
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_positive_metrics_fixed8_20260909_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        dest=(D/name).resolve();assert dest.is_relative_to(D.resolve()) and Path(name).name==name
        if dest.exists():assert dest.read_bytes()==z.read(name)
        else:dest.write_bytes(z.read(name))
r=json.loads((D/'results.json').read_bytes());p=json.loads((D/'protocol.json').read_bytes());rec=json.loads((D/'terminal_receipt.json').read_bytes())
assert r['protocol']==p and r['status']=='positive_metric_view_fixed8_0DT168originalscores_complete' and rec['proc_exists'] is False
for name,item in rec['files'].items():assert sha(D/name)==item['sha256'] and (D/name).stat().st_size==item['bytes']
for name,h in p['files_sha256'].items():assert sha(D/name)==h;ast.parse((D/name).read_bytes())
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
assert r['DT_entered']==r['DT_returned']==r['FT_calls']==r['generation_calls']==0
assert r['scorer_entered']==r['scorer_returned']==168 and r['model_loads']==r['native_eager_diagnostics']==1
assert set(r['cases'])==set(p['quality_cases']) and len(r['cases'])==8
assert sha(D/'vectors.npz')==r['vectors_sha256']
vec=np.load(D/'vectors.npz',allow_pickle=False)
auc=lambda x:float((np.asarray(x).sum()-(x[0]+x[-1])/2)/(len(x)-1))
cases={}
for key,case in r['cases'].items():
    c=case['curve'];source=case['source'];info=case['input'];keep=np.asarray(info['keep'],dtype=int)
    directory=A/'snapshot/tmp'/Path(source['results_path']).parent.name
    assert sha(directory/'results.json')==source['results_sha256'] and sha(directory/'vectors.npz')==source['vectors_sha256']
    old=json.loads((directory/'results.json').read_bytes())['cases'][key]['curves']['control']
    with np.load(directory/'vectors.npz',allow_pickle=False) as prior:
        signed=prior[key+'_control_evaluated'].astype(np.float32)
        assert np.array_equal(vec[key+'_signed'],signed)
    w=vec[key+'_positive'];assert np.array_equal(w,np.maximum(signed,0))
    assert hashlib.sha256(signed.tobytes()).hexdigest()==source['signed_FP32_sha256']
    assert np.isfinite(w).all() and (w>=0).all()
    order=c['sorted_keep'];assert set(order)==set(keep) and len(order)==len(keep)
    assert (np.diff(w[order])<=0).all() and c['returned_forwards']==21
    ids=np.asarray(r['input_freeze_before_model_load'][key]['input_ids'],dtype=np.int64)
    assert hashlib.sha256(ids.tobytes()).hexdigest()==info['input_sha256']
    gold=sorted(set(case['gold'])&set(keep));positive_sort=np.sort(w[keep])[::-1]
    if gold:
        topk=max(1,int(np.ceil(.1*len(keep))));threshold=float(positive_sort[topk-1])
        strict=set(int(v) for v in keep[w[keep]>threshold]);ties=set(int(v) for v in keep[w[keep]==threshold]);slots=topk-len(strict)
        low=(len(strict&set(gold))+max(0,slots-len(ties-set(gold))))/len(gold)
        high=(len(strict&set(gold))+min(slots,len(ties&set(gold))))/len(gold)
        assert low<=c['needle']<=high and c['needle']==c['signed_input_needle']==source['historical_needle']
        needle={'value':c['needle'],'hits':round(c['needle']*len(gold)),'denominator':len(gold),'interval':[low,high]}
    else:assert c['needle'] is None;needle=None
    density=[1.];offset=0;deleted=set();x=ids.copy();total=float(w[keep].sum(dtype=np.float64))
    for step,receipt in enumerate(c['input_receipts']):
        if step:
            size=len(keep)//20+int(step<=len(keep)%20);group=order[offset:offset+size];offset+=size
            deleted.update(group);x[group]=p['input_eos_token_id'] if 'input_eos_token_id' in p else 248046
            density.append(density[-1]-float(w[group].sum(dtype=np.float64))/total)
        assert receipt['deleted_positions']==sorted(deleted)
        assert hashlib.sha256(x.tobytes()).hexdigest()==receipt['input_sha256']
    assert offset==len(keep) and np.max(np.abs(np.asarray(density)-c['density']))<3e-6
    scores=np.asarray(c['scores']);response=np.minimum.accumulate(np.clip((scores-scores[-1])/abs(scores[0]-scores[-1]),0,1))
    penalty=np.abs(response-np.asarray(c['density']));corrected=np.clip(response+penalty,0,1)
    corrected=(corrected-corrected.min())/(corrected.max()-corrected.min()) if corrected.max()>corrected.min() else np.linspace(1,0,21)
    for value,name in [(response,'normalized_model_response'),(penalty,'alignment_penalty'),(corrected,'corrected_scores')]:assert np.max(np.abs(value-c[name]))<1e-12
    metrics=[auc(response),auc(corrected),auc(response+penalty)];assert np.max(np.abs(np.asarray(metrics)-c['return_metrics']))<1e-12
    same=[i for i,(a,b) in enumerate(zip(c['input_receipts'],old['input_receipts'])) if a['input_sha256']==b['input_sha256']]
    ft={}
    for name,ref in case['historical_FT_reference']['curves'].items():
        metric=ref.get('original_return_metrics',ref.get('return_metrics'));assert metric is not None
        ft[name]={'RISE':metric[0],'MAS':metric[1],'positive_DT_minus_FT_RISE':metrics[0]-metric[0],'positive_DT_minus_FT_MAS':metrics[1]-metric[1],'needle':ref['needle']}
    cases[key]={'positive':{'RISE':metrics[0],'MAS':metrics[1],'needle':needle},
        'historical_signed':{'RISE':source['historical_signed_metrics'][0],'MAS':source['historical_signed_metrics'][1]},
        'positive_minus_historical_signed':{'RISE':metrics[0]-source['historical_signed_metrics'][0],'MAS':metrics[1]-source['historical_signed_metrics'][1]},
        'historical_FT':ft,'same_actual_input_steps':same,'max_native_score_drift_on_same_inputs':max(abs(scores[i]-old['scores'][i]) for i in same),
        'endpoint_native_score_drift':[scores[i]-old['scores'][i] for i in (0,20)],
        'changed_mask_steps':[i for i in range(21) if i not in same],
        'original_curve':{k:c[k] for k in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']},
        'score_source_sha256':source['vectors_sha256'],'peak_allocated_full_model_resident':case['peak_allocated_full_model_resident']}
groups={}
for group in ['NI','MH','all']:
    rows=[v for k,v in cases.items() if group=='all' or (k.startswith('niah') if group=='NI' else k.startswith('more'))]
    entry={'n':len(rows),'means':{view:{m:float(np.mean([v[view][m] for v in rows])) for m in ['RISE','MAS']} for view in ['positive','historical_signed']},
        'versus_signed_counts':{m:{'better':sum(v['positive_minus_historical_signed'][m]<0 for v in rows),'equal':sum(v['positive_minus_historical_signed'][m]==0 for v in rows),'worse':sum(v['positive_minus_historical_signed'][m]>0 for v in rows)} for m in ['RISE','MAS']},
        'FT':{name:{m:{'mean':float(np.mean([v['historical_FT'][name][m] for v in rows])), 'DT_better_count':sum(v['positive'][m]<v['historical_FT'][name][m] for v in rows)} for m in ['RISE','MAS']} for name in ['FT0','FT3']}}
    gold=[v['positive']['needle'] for v in rows if v['positive']['needle']]
    entry['needle']={'hits':sum(v['hits'] for v in gold),'denominator':sum(v['denominator'] for v in gold),'valid_examples':len(gold)} if gold else None
    groups[group]=entry
out={'status':'positive_fixed8_original_metrics_and_score_view_independently_verified','results_sha256':sha(D/'results.json'),
    'protocol_sha256':sha(D/'protocol.json'),'receipt_sha256':sha(D/'terminal_receipt.json'),'vectors_sha256':sha(D/'vectors.npz'),
    'cases':cases,'groups':groups,'seconds':r['seconds'],'calls':r['calls'],'new_model_loads':1,'eager_initializations':1,'DT':0,'FT':0,'original_scoring_forwards':168,
    'scope':'User-authorized positive score preprocessing, new original curves on all fixed8 current-C scores. Original signed vectors and historical signed metrics retained. Unchanged finite propagation and native FA/FLA. Historical FT and signed curves are different-process references; same-input drifts are explicit. No new method, sign validation or attribution-speed claim.'}
(A/'dt_positive_metrics_fixed8_summary_20260909.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'status':out['status'],'groups':groups,'seconds':r['seconds'],'drifts':{k:v['max_native_score_drift_on_same_inputs'] for k,v in cases.items()}},ensure_ascii=False))
