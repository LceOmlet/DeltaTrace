"""Verify directly reproduced native B4 row variation; no model execution."""
import hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_batch_endpoint_diagnostic_20260907_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete' and p==json.loads((A/'batch_endpoint_diagnostic_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'batch_endpoint_diagnostic_20260907.py')
parent_file=A/'snapshot'/p['batch_parent'].lstrip('/');assert sha(parent_file)==p['batch_parent_sha256']
parent=json.loads(parent_file.read_text())
assert d['checkpoint_before']==d['checkpoint_after']==parent['checkpoint_before']==parent['checkpoint_after']
assert d['native_sources_before']==d['native_sources_after']==parent['native_sources_before']==parent['native_sources_after']
for k,v in p['budget'].items():assert d[k]==v
assert len(d['records'])==16
out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'budget':p['budget'],'cases':[],
     'scope':'Direct unchanged original eager evaluator reproduces the observed B4 endpoint values exactly. This isolates the discrepancy from the new coroutine/finite attribution code; it does not establish the exact underlying backend arithmetic cause.'}
for dataset,index in p['selection']:
    row=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
    for endpoint,position in [('clean',0),('deleted',-1)]:
        entries=[r for r in d['records'] if (r['dataset'],r['idx'],r['endpoint'])==(dataset,index,endpoint)]
        assert len(entries)==4 and {(r['batch'],r['repeat']) for r in entries}=={(1,0),(1,1),(4,0),(4,1)}
        for r in entries:
            assert r['identical_inputs_within_batch'] and r['input_ids_sha256']==row['input_ids_sha256']
            a=np.asarray(r['token_logprobs']);assert a.shape==(r['batch'],len(row['input_ids'])-row['prompt_len']) and np.isfinite(a).all()
            # Stored half token values give exact half reduction only after final
            # rounding. Bound this independently by half's representable ULP.
            reduced=a.sum(axis=1).astype(np.float16).astype(float)
            assert np.array_equal(reduced,np.asarray(r['sums']))
            assert r['cost']['native_forwards']==1 and r['cost']['native_forward_trajectories']==r['batch'] and r['cost']['native_decoder_layer_calls']==36 and r['cost']['extra_replay_calls']==0
        first=next(r for r in entries if r['batch']==4 and r['repeat']==0)
        repeated=next(r for r in entries if r['batch']==4 and r['repeat']==1)
        assert first['token_logprobs']==repeated['token_logprobs']
        expected=[row['metrics'][m]['raw_curve'][position] for m in ['single','batch2','batch4','historical_FT']]
        assert first['sums']==expected
        out['cases'].append({'dataset':dataset,'idx':index,'endpoint':endpoint,'native_B4_sums':first['sums'],
             'native_B1_sums':[r['sums'][0] for r in entries if r['batch']==1],
             'B4_row_spread':max(expected)-min(expected),'directly_matches_original_batch_run':True,'B4_repeat_token_values_exact':True})
(A/'batch_endpoint_diagnostic_summary_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
