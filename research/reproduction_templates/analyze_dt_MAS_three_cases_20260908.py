"""Independently verify the bounded MAS study, including any stopped attempts."""
import ast,hashlib,json,zipfile,importlib.util
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
spec=importlib.util.spec_from_file_location('mas_diagnosis',R/'research/reproduction_templates/diagnose_NI0_MAS_raw_response.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
auc=mod.auc
jobs=[];cases={}
for version in [1,2]:
    d=A/f'snapshot${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v{version}'
    receipt=json.loads((d/'terminal_receipt.json').read_bytes());assert not receipt['proc_exists']
    with zipfile.ZipFile(d/'review_bundle.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert Path(name).name==name
            f=d/name;raw=z.read(name)
            if f.exists():assert f.read_bytes()==raw
            else:f.write_bytes(raw)
    for name,entry in receipt['files'].items():
        raw=(d/name).read_bytes();assert sha(raw)==entry['sha256'] and len(raw)==entry['bytes']
    result=json.loads((d/'results.json').read_bytes());p=json.loads((d/'protocol.json').read_bytes());assert p==result['protocol']
    for name,want in p['files_sha256'].items():assert sha((d/name).read_bytes())==want;ast.parse((d/name).read_bytes())
    for name,want in p['unchanged_finite_runtime_sha256'].items():assert sha((d/name).read_bytes())==want
    vec=np.load(d/'vectors.npz');assert sha((d/'vectors.npz').read_bytes())==result['vectors_sha256']
    control=result['control']['DT'];prior=np.load(A/'snapshot${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1/signed_result.npz')['signed']
    now=vec['niah_mq_q2_0_DT_full'];rel=float(np.linalg.norm(now-prior)/np.linalg.norm(prior))
    assert abs(rel-control['versus_frozen_relative_L2'])<1e-12
    for key,row in [('control',result['control'])]+list(result['cases'].items()):
        if 'DT' not in row:continue
        dt=row['DT'];assert len(dt['layers'])==32
        assert all(v['replay_relative_L2']==0 for v in dt['layers'].values())
        assert all(v.get('FA_auxiliary_relative_L2',0)==0 for v in dt['layers'].values())
        kinds=[c['kind'] for c in dt['calls']]
        assert sum(k.startswith('native_replay_') for k in kinds)==32
        assert sum(k.startswith('finite_decoder_') for k in kinds)==32
        assert sum(k.startswith('public_FA_LSE_') for k in kinds)==8
        if key!='control':
            assert np.isclose(vec[key+'_DT_full'].sum(),dt['signed_sum'],rtol=0,atol=1e-10)
    jobs.append({'version':version,'status':result['status'],'receipt':receipt,'job_seconds':result['job_seconds'],
        'model_loads':result['model_loads'],'DT_calls':result['DT_calls'],'FT_calls':result['FT_calls'],
        'native_scoring_forwards':result['scoring_forwards_completed'],'control_relative_L2':rel,'control_needle':control['needle'],
        'error':result.get('error'),'calls':result['calls']})
    if result['status']!='fixed_DT_FT_three_author_cases_complete':continue
    assert result['sources_before']==result['sources_after'] and result['weight_stats_before']==result['weight_stats_after']
    assert result['DT_calls']==4 and result['FT_calls']==3 and result['scoring_forwards_entered']==result['scoring_forwards_completed']==315
    for key,case in result['cases'].items():
        info=case['input'];keep=info['keep'];keep_set=set(keep);methods={};endpoint_hashes=[]
        assert case['FT']['input_receipts'][0]['sha256']==info['input_sha256']
        assert set(case['curves'])==set(p['method_order'])
        for name,row in case['curves'].items():
            w=np.asarray(vec[key+'_'+name],dtype=np.float32);assert w.shape==(info['prompt_length'],) and np.isfinite(w).all()
            order=row['sorted_keep'];assert len(order)==len(keep) and set(order)==keep_set and np.all(np.diff(w[order])<=0)
            assert np.isclose(float(w[keep].sum(dtype=np.float32)),row['attr_sum'],rtol=1e-6)
            offset=0;deleted=set();density=[1.]
            for i,entry in enumerate(row['input_receipts']):
                if i:
                    size=len(keep)//20+(i<=len(keep)%20);group=order[offset:offset+size];offset+=size;deleted.update(group)
                    density.append(density[-1]-float(w[group].sum(dtype=np.float32))/row['attr_sum'] if row['attr_sum']>0 else 1-i/20)
                assert entry['deleted_positions']==sorted(deleted)
            assert offset==len(keep) and np.max(np.abs(np.asarray(density)-row['density']))<2e-6
            g=np.asarray(row['scores']);response=np.minimum.accumulate(np.clip((g-g[-1])/abs(g[0]-g[-1]),0,1))
            penalty=np.abs(response-np.asarray(row['density']));corrected=np.clip(response+penalty,0,1)
            corrected=(corrected-corrected.min())/(corrected.max()-corrected.min())
            if np.isnan(corrected).any():corrected=np.linspace(1,0,21)
            for value,col in [(response,'normalized_model_response'),(penalty,'alignment_penalty'),(corrected,'corrected_scores')]:assert np.max(np.abs(value-row[col]))<1e-12
            assert np.max(np.abs(np.asarray([auc(response),auc(corrected),auc(response+penalty)])-row['return_metrics']))<1e-12
            methods[name]=mod.diagnose(row)
            methods[name]['needle']=case['DT']['needle'] if name=='DT' else case['FT']['needle'][int(name[-1])]
            endpoint_hashes.append((row['input_receipts'][0]['input_sha256'],row['input_receipts'][-1]['input_sha256']))
        assert len(set(endpoint_hashes))==1 and endpoint_hashes[0]==(info['input_sha256'],info['baseline_sha256'])
        deltas={name:{m:methods['DT'][m]-methods[name][m] for m in ['original_RISE','original_MAS']} for name in p['method_order'] if name!='DT'}
        cases[key]={'input':info,'methods':methods,'DT_minus_FT':deltas,
            'DT_seconds_with_diagnostics':case['DT']['complete_attribution_seconds_with_diagnostics'],
            'DT_peak_allocated':case['DT']['peak_allocated'],'FT_peak_allocated':case['FT']['peak_allocated']}
summary={'status':'bounded_MAS_study_independently_audited','jobs':jobs,'cases':cases,
    'original_FT_modified':False,'finite_mathematics_changed':False,'original_metrics_changed':False,
    'sample_count_new_scored':len(cases),'goal_complete':False,
    'limits':['Original cached trajectories and metric functions; external input formatter retains matching raw model input, not stock paper-table reproduction.',
              'Historical development samples, not independent confirmation. Keep NI and Morehop separate. Default BF16 scoring ties unresolved.',
              'NI0 control is a rerun of the same sample, not an additional data point. Preserve all failed execution guards and native precision drift.',
              'Attribution timings contain different initial call and diagnostic overhead; this is a quality study, not a steady comparative speed benchmark.']}
(A/'dt_MAS_three_cases_summary_20260908.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':summary['status'],'jobs':[{k:j[k] for k in ['version','status','job_seconds','DT_calls','FT_calls','native_scoring_forwards','control_relative_L2']} for j in jobs],
 'cases':{key:{name:{k:m[k] for k in ['needle','original_RISE','original_MAS','original_alignment_AUC']} for name,m in row['methods'].items()} for key,row in cases.items()}}))
