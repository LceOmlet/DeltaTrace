"""Verify the bounded target ablation's raw vectors, calls and original curves."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    p=HERE/'qwen35_target_scope';r=json.loads((p/'results.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
    assert r['status']=='complete' and r['driver_sha256']==sha((HERE/'diagnose_qwen35_target_scope.py').read_bytes())
    assert r['vectors_sha256']==sha((p/'vectors.npz').read_bytes())
    z=np.load(p/'vectors.npz',allow_pickle=False);assert len(z.files)==8 and all(np.isfinite(z[n]).all() for n in z.files)
    assert [c['case'] for c in r['cases']]==['niah_mq_q2_0','niah_mq_q2_1','morehopqa_0','morehopqa_1']
    assert r['FT_calls']==r['generation_calls']==0
    native_metric_calls=0;rows=[]
    for c in r['cases']:
        assert len(c['DT_root_calls'])==2 and c['DT_root_calls'][0]['input_sha256']==c['DT_root_calls'][1]['input_sha256']
        out={'case':c['case']}
        for scope in ('whole','answer'):
            d=c['details'][scope];m=c['metrics'][scope]
            assert np.isclose(sum(d['target_logp1'])-sum(d['target_logp0']),d['root_effect'],atol=1e-8,rtol=0)
            assert np.isclose(z[c['case']+'_'+scope+'_signed_full'].sum(),d['signed_sum'],atol=1e-8,rtol=0)
            out.update({scope+'_'+k:m[k] for k in ('rise','mas','needle')})
            out[scope+'_root_effect']=d['root_effect'];out[scope+'_target_count']=len(c['target_offsets'][scope])
            for view in ('positive','signed'):
                if view not in m:continue
                curve=m[view];native_metric_calls+=len(curve['actual_input_hashes'])
                assert len(curve['actual_input_hashes'])==21 and curve['actual_input_hashes'][0]==c['input_sha256']
                assert curve['MAS_valid']==(view=='positive')
            assert m['mas']==m['positive']['mas']
            if 'signed' in m:assert m['rise']==m['signed']['rise']
            else:assert m['signed_reuse_proof'] and m['rise']==m['positive']['rise']
        out['RISE_delta_answer_minus_whole']=out['answer_rise']-out['whole_rise']
        out['MAS_delta_answer_minus_whole']=out['answer_mas']-out['whole_mas'];rows.append(out)
    assert native_metric_calls==r['metric_root_calls']==294
    assert sum(c['name'].endswith('_DT') for c in r['calls'])==8
    summary={'status':'verified','result_sha256':sha((p/'results.json').read_bytes()),'vectors_sha256':r['vectors_sha256'],
             'DT_calls':8,'original_metric_root_calls':294,'FT_calls':0,'generation_calls':0,'rows':rows,
             'decision':'Stop answer-only target candidate: signed RISE and positive MAS worsen in all four cases, and both NI needle scores fall. Keep the whole-generation target.',
             'scope':'This rejects a target-scope replacement. It does not prove how much the fixed FT seed difference explains cross-model method gaps, or exclude GDN allocation issues.'}
    (p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',newline='\n')
    with (p/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
