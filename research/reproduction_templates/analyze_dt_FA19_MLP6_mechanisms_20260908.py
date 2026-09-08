"""Independent NumPy/hash audit of native FA contrasts and MLP mechanism ledgers."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;S=A/'snapshot/tmp';sha=lambda b:hashlib.sha256(b).hexdigest()
jobs=['dt_fa19_native_hybrid_20260908_v1','dt_fa19_native_hybrid_20260908_v2','dt_mlp6_mechanism_20260908_v1']
records={}
for j in jobs:
    d=S/('codex_'+j);r=json.loads((d/'results.json').read_bytes());receipt=json.loads((d/'terminal_receipt.json').read_bytes())
    assert not receipt['proc_exists']
    for n,e in receipt['files'].items():
        raw=(d/n).read_bytes();assert len(raw)==e['bytes'] and sha(raw)==e['sha256'],(j,n)
    for n,h in r['protocol']['files_sha256'].items():
        raw=(d/n).read_bytes();assert sha(raw)==h,(j,n)
        if n.endswith('.py'):ast.parse(raw)
    records[j]=r
failed,fa,mlp=[records[j] for j in jobs]
assert failed['status']=='failed' and failed['native_FA_calls_entered']==failed['native_FA_calls_returned']==0
assert fa['status']=='eleven_native_FA_hybrid_contrasts_complete'
assert fa['native_FA_calls_entered']==fa['native_FA_calls_returned']==11
assert [c['label'] for c in fa['calls']]==fa['protocol']['native_call_schedule']
assert all(c['status']=='returned' for c in fa['calls'])
assert fa['private_artifact_sha256_before']==fa['private_artifact_sha256_after']==fa['protocol']['private_artifact_sha256']
assert fa['FA_interface_sha256_before']==fa['FA_interface_sha256_after']==fa['protocol']['installed_FA_interface_sha256']
assert all(fa[k]==0 for k in ['model_calls','DT_calls','scorer_calls','FT_calls','backward_calls','generation_calls'])
source_raw=(S/'codex_dt_decoder19_6_conditional_20260908_v1/results.json').read_bytes()
assert sha(source_raw)==fa['protocol']['source_results_sha256'];source=json.loads(source_raw)
arrays=np.load(S/'codex_dt_fa19_native_hybrid_20260908_v2/signed_token_contrasts.npz',allow_pickle=False)
out={'status':'native_FA_and_MLP_mechanism_ledgers_independently_checked','sign':'prediction_minus_actual',
     'results_sha256':{j:sha((S/('codex_'+j)/'results.json').read_bytes()) for j in jobs},
     'FA_failed_pre_call':{'seconds':failed['job_seconds'],'entered':0,'returned':0,'error':failed['error']},
     'FA_replays':fa['replays'],'FA_saved_B1_vs_B2_drift':fa['saved_B1_vs_B2_endpoint_drift'],
     'FA_points':{},'MLP_points':{},'MLP_saved_vs_recomputed':mlp['saved_vs_recomputed_mnorm']}
for step,row in fa['points'].items():
    fields=row['fields'];get=lambda n:fields[n]['net']
    assert row['input_receipt']==source['points'][step]['input_receipt']
    error=source['points'][step]['layer_decompositions']['19']['terms']['finite_FA_core_including_seed_cast']
    assert abs(get('core_error_at_stored_seed')-error)<1e-7
    terms=['routing_at_baseline_values_prediction_error','content_at_clean_routing_prediction_error','routing_content_interaction_contrast']
    assert abs(sum(get(n) for n in terms)-error)<1e-7
    assert abs(get('core_error_at_stored_seed')-get('core_error_at_BF16_seed')-get('BF16_minus_stored_seed_actual_effect'))<1e-7
    for n,x in fields.items():
        v=arrays[step+'_'+n];assert np.isfinite(v).all()
        for name,value in [('net',v.sum()),('positive_sum',v.clip(min=0).sum()),('negative_sum',v.clip(max=0).sum()),('absolute_sum',np.abs(v).sum())]:
            assert abs(float(value)-x[name])<1e-7,(step,n,name)
        assert abs(sum(g['fields'][n]['net'] for g in row['groups'].values())-x['net'])<1e-7
    assert sum(g['count'] for g in row['groups'].values())==source['input']['total_length']
    out['FA_points'][step]={'core_error':error,'terms':{n:fields[n] for n in terms},
         'seed_cast':fields['BF16_minus_stored_seed_actual_effect'],'groups':row['groups'],'checks':row['checks']}
assert mlp['status']=='MLP6_actual_mechanism_diagnostic_complete'
assert mlp['diagnostic_graph_calls_entered']==mlp['diagnostic_graph_calls_returned']==1 and mlp['weight_tensors_read']==3
assert mlp['sources_before']==mlp['sources_after'] and mlp['saved_vs_recomputed_mnorm']['relative_L2']==0
assert all(mlp[k]==0 for k in ['model_loads','DT_calls','FT_calls','native_score_calls','generation_calls','finite_FA_calls','finite_FLA_calls'])
for w in mlp['weight_receipts'].values():assert w['tensor_sha256_before']==w['tensor_sha256_after']
for step,row in mlp['points'].items():
    assert len(row['terms'])==10
    total=sum(x['total'] for x in row['terms'].values())
    assert abs(total-row['saved_prediction_minus_actual'])<1e-7
    expected=(source['B2_decompositions']['6'] if step=='B2' else source['points'][step]['layer_decompositions']['6'])['terms']['MLP_combined']
    assert abs(total-expected)<1e-7
    for name,x in row['terms'].items():
        assert abs(sum(x['per_token'])-x['total'])<1e-7
        assert abs(x['positive']+x['negative']-x['total'])<1e-7
        assert abs(sum(v['total'] for v in x['groups'].values())-x['total'])<1e-7
    out['MLP_points'][step]={'error':row['saved_prediction_minus_actual'],
        'terms':{n:{k:v for k,v in x.items() if k!='per_token'} for n,x in row['terms'].items()},'closure_residual':row['closure_residual']}
out['budget']={'native_FA_calls':11,'MLP_graphs':1,'semantic_GEMMs':3,'weight_tensors_read':3,
    'model_DT_scorer_FT_generation_calls':0,'new_samples':0,'job_seconds_including_failed_FA':sum(x['job_seconds'] for x in [failed,fa,mlp])}
out['scope']='Actual retained MH1 operators and original conditional inputs only; no new quality metric or candidate effect. FA location grouping is not original-source causality. MLP coefficients were explicitly recomputed; their output mnorm matches the actual production vector.'
(A/'dt_FA19_MLP6_mechanisms_summary_20260908.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':out['status'],'budget':out['budget'],'FA_mid':{k:v['net'] for k,v in out['FA_points']['10']['terms'].items()},'MLP_mid':{k:v['total'] for k,v in out['MLP_points']['10']['terms'].items()}}))
