"""Independently recalculate saved dQ errors; retain failed native-layout attempts."""
import hashlib,json,statistics,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
review=[]
for version in [1,2,3]:
    F=A/f'snapshot${ARTIFACT_ROOT}/codex_fa_two_sweeps_operator_20260907_v{version}'
    with zipfile.ZipFile(F/'review_bundle.zip') as z:
        for n in z.namelist():
            assert '/' not in n and '\\' not in n
            (F/n).write_bytes(z.read(n))
    d=json.loads((F/'results.json').read_text());p=d['protocol']
    assert p==json.loads((F/'protocol.json').read_text())
    assert sha(F/'study.py')==p['study_sha256']
    for n,h in p['sources'].items():assert sha(F/n)==h
    assert d['attributions']==d['native_model_forwards']==d['VJPs']==d['quality_queries']==0
    for repeat,c in enumerate(d['comparisons']):
        rows=[next(x for x in d['calls'] if x['repeat']==repeat and x['name']==n) for n in ['old','two_sweeps']]
        for k in ['tau','center','dk','dv']:
            assert c[k]['exact'] and rows[0]['outputs'][k]==rows[1]['outputs'][k]
        arrays=[]
        for n,row in zip(['old','two_sweeps'],rows):
            item=c['dq'][n+'_array'];assert sha(F/item['file'])==item['sha256']
            raw=np.load(F/item['file'],allow_pickle=False)
            assert hashlib.sha256(raw.tobytes()).hexdigest()==row['outputs']['dq']['sha256']
            arrays.append(raw.astype(np.float64))
        a,b=arrays;delta=b-a;rms=float(np.sqrt(np.mean(a*a)));error_rms=float(np.sqrt(np.mean(delta*delta)))
        assert np.isfinite(b).all()
        for key,value in {'old_RMS':rms,'error_RMS':error_rms,'relative_L2':error_rms/rms,
            'max_abs_difference':float(abs(delta).max()),'max_abs_over_old_RMS':float(abs(delta).max())/rms,
            'sign_flips':int(np.count_nonzero(a*b<0))}.items():
            assert np.isclose(c['dq'][key],value,rtol=1e-12,atol=1e-15),(key,value,c['dq'][key])
    passed=d['status']=='two_sweeps_operator_numerical_screen_passed'
    if passed:
        assert len(d['calls'])==6 and len(d['comparisons'])==3
        assert all(c['dq']['relative_L2']<=0.001 and c['dq']['max_abs_over_old_RMS']<=0.01 for c in d['comparisons'])
    else:
        assert d['status']=='failed' and len(d['calls'])==2
        assert any(c['dq']['relative_L2']>0.001 or c['dq']['max_abs_over_old_RMS']>0.01 for c in d['comparisons'])
    times={n:[x['seconds'] for x in d['calls'] if x['name']==n and not x['warm']] for n in ['old','two_sweeps']}
    ratio=statistics.median(times['two_sweeps'])/statistics.median(times['old']) if passed else None
    review.append({'version':version,'status':d['status'],'raw_sha256':sha(F/'results.json'),
        'protocol_sha256':sha(F/'protocol.json'),'library_sha256':d['build_library']['sha256'],
        'operator_calls':len(d['calls']),'numerical_gate_passed':passed,'comparisons':d['comparisons'],
        'measured_seconds':times,'new_to_old_median_ratio':ratio,
        'calls':d['calls'],'whole_model_screen_allowed':bool(passed and ratio<=1.05)})
out={'status':'local_actual_tensor_results_verified','attempts':review,
    'total_operator_calls':sum(r['operator_calls'] for r in review),'attributions':0,'quality_queries':0,
    'latest_whole_model_screen_allowed':review[-1]['whole_model_screen_allowed'],
    'scope':'Old original NI0 layer35. All failed gates retained; no new benchmark, full-model quality/sign or speed claim.'}
(A/'fa_two_sweeps_operator_summary_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'status':out['status'],'total_operator_calls':out['total_operator_calls'],
    'latest':{k:review[-1][k] for k in ['status','numerical_gate_passed','measured_seconds','new_to_old_median_ratio','whole_model_screen_allowed','comparisons']}}))
