import json,hashlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_order_norm_localization_20260908_v1'
sha=lambda b:hashlib.sha256(b).hexdigest();r=json.loads((D/'normgate_geometry_followup.json').read_bytes())
p=json.loads((D/'normgate_geometry_protocol.json').read_bytes());receipt=json.loads((D/'normgate_geometry_receipt.json').read_bytes())
assert r['status']=='CPU_geometry_followup_complete' and r['source_experiment_status']=='failed' and not receipt['proc_exists']
assert r['GPU_calls']==r['model_calls']==r['native_model_operator_calls']==r['parameter_searches']==0
assert sha((D/'results.json').read_bytes())==r['source_results_sha256']==p['source_results_sha256']
for name,item in receipt['files'].items():
    f=R/'research/reproduction_templates'/name if name.endswith('.py') else D/name
    raw=f.read_bytes();assert len(raw)==item['bytes'] and sha(raw)==item['sha256']
assert r['self_sha256']==p['script_sha256']
max_error=0.
for key,case in r['cases'].items():
    assert case['input_artifact_sha256_before']==case['input_artifact_sha256_after']==p['artifact_sha256']
    for step,point in case['steps'].items():
        t=point['terms'];assert abs(t['RMS_reference_effect_from_actual_o']+t['current_RMS_error']-t['current_theoretical_RMS_prediction'])<1e-7
        for name,value in t.items():
            assert abs(value-sum(h['terms'][name] for h in point['per_head']))<1e-7
            assert abs(value-sum(g[name] for g in point['token_groups'].values()))<1e-7
        max_error=max(max_error,*point['max_abs_checks'].values(),*map(abs,point['total_closure_checks'].values()))
        assert max_error<1e-7
print(json.dumps({'status':'independent_CPU_audit_passed','seconds':r['CPU_job_seconds'],'max_closure':max_error,
    'new_GPU':0,'source_experiment_still_failed':True,'candidate_original_metrics_measured':False}))
