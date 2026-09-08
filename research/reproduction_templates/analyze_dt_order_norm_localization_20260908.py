"""Independent scalar/vector audit, no model or candidate metric calls."""
import json,hashlib,ast,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_order_norm_localization_20260908_v1';sha=lambda b:hashlib.sha256(b).hexdigest()
p=json.loads((D/'protocol.json').read_bytes());r=json.loads((D/'results.json').read_bytes())
receipt=json.loads((D/'terminal_receipt.json').read_bytes());assert not receipt['proc_exists']
assert r['status']=='failed' and 'Frozen 2% vector guard failed' in r['error']
post=json.loads((D/'post_failure_source_receipt.json').read_bytes())
assert post['sources_match'] and post['weight_stats_match']
assert r['model_loads']==1 and r['DT_calls']==2 and r['scoring_forwards']==4 and r['FT_calls']==r['generation_calls']==0
# Counter commits after finish(), so it contains NI only. Both DT calls returned
# and helper executes before either baseline guard: the true total is two.
assert r['extra_conv_preactivation_calls']==1
assert sum(x['kind']=='current' for x in r['finite_FLA_calls'])==48
assert sum(x['kind']=='reverse' for x in r['finite_FLA_calls'])==2
for name,want in p['files_sha256'].items():
    raw=(D/name).read_bytes();assert sha(raw)==want,name;ast.parse(raw)
prior=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_dt_conditional_boundaries_20260908_v1/results.json').read_bytes())
focused=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_dt_layer0_conditional_20260908_v1/results.json').read_bytes())
v=np.load(D/'vectors.npz');old=np.load(A/'snapshot${ARTIFACT_ROOT}/codex_dt_conditional_boundaries_20260908_v1/vectors.npz')
s={'status':r['status'],'receipt':receipt,'protocol_sha256':sha((D/'protocol.json').read_bytes()),
   'results_sha256':sha((D/'results.json').read_bytes()),'vectors_sha256':sha((D/'vectors.npz').read_bytes()),
   'bundle_sha256':sha((D/'review_bundle.zip').read_bytes()),'job_seconds':r['job_seconds'],
   'budget':p['budget'],'actual_budget':{'model_loads':1,'native_eager_B1_diagnostics':1,'DT_calls':2,
      'native_decoder_replays':64,'finite_decoders':64,'aux_FA_calls':16,'current_finite_FLA':48,
      'extra_reverse_finite_FLA':2,'extra_native_FLA_adjoint_stages':4,'extra_GDN_and_conv_preactivation_and_autograd':2,
      'scoring_forwards':4,'FT':0,'generation':0,'new_samples':0},
   'sources_unchanged':True,'native_weights_unchanged':True,'cases':{},'failure':r['error'],
   'sign_convention':'Main summary prediction_minus_actual throughout; complete-input source has opposite sign.',
   'metric_scope':'Frozen baseline deletion sets only. No candidate-new-sort RISE/MAS and no speed benchmark.'}
for key,row in r['cases'].items():
    relative=float(np.linalg.norm(v[key+'_DT_full']-old[key+'_DT_full'])/np.linalg.norm(old[key+'_DT_full']))
    assert abs(relative-row['parent_vector_relative_L2'])<1e-12
    if not row['scoring_points']:
        assert key=='morehopqa_1' and relative>.02
        s['cases'][key]={'status':'stopped_before_scoring','relative_L2':relative,
            'candidate_vector_saved_but_not_quality_accepted':True};continue
    assert relative<=.02
    info=row['input'];P=info['prompt_length'];keep=np.array(info['keep']);current=v[key+'_DT_full'][:P].astype(np.float32).astype(float)
    candidate=v[key+'_layer0_order_average_full'][:P].astype(np.float32).astype(float)
    out={'status':'first_case_complete_before_guard_failure','baseline_vector_identical':False,'relative_L2':relative,
         'order_average':row['order_average'],'points':{},'normgate_CPU_algebra_seconds':row['normgate_CPU_algebra_seconds'],
         'private_normgate_input_artifact':row['private_normgate_input_artifact'],'coefficient_artifact':row['coefficient_artifact']}
    for step,point in row['scoring_points'].items():
        assert point['original_native_score']==point['prior_original_native_score']
        assert point['input_receipt']==prior['cases'][key]['curve']['input_receipts'][int(step)]
        if step=='0':continue
        ledger=point['layer0_decomposition'];ng=point['normgate_decomposition'];order=point['FLA_endpoint_order'];complete=point['complete_input_order_average']
        assert abs(ledger['actual_minus_predicted']-sum(ledger['terms'].values()))<1e-7
        assert abs(sum(ng['terms'].values())-ng['measured_prediction_minus_actual'])<1e-7
        assert abs(ng['measured_prediction_minus_actual']+ledger['terms']['GDN_fused_norm_gate'])<1e-7
        per={k:np.array(x) for k,x in ng['per_token_terms'].items()}
        assert max(abs(float(x.sum())-ng['terms'][k]) for k,x in per.items())<1e-7
        assert np.max(np.abs(sum(per.values())-np.array(ng['per_token_measured_prediction_minus_actual'])))<1e-7
        assert max(abs(sum(g[k] for g in ng['token_group_terms'].values())-ng['terms'][k]) for k in ng['terms'])<1e-7
        assert abs(sum(ng['product_subdivision'].values())-ng['terms']['product_B2_baseline'])<1e-7
        for name in ['RMS','SiLU']:
            assert abs(ng['operator_subdivision'][name+'_B1_endpoint_operator']+ng['operator_subdivision'][name+'_B2_minus_B1_operator']-ng['terms'][name+'_conditional_B2_operator'])<1e-7
        a=order['current']['prediction_minus_actual'];b=order['reverse']['prediction_minus_actual']
        assert abs(a+ledger['terms']['GDN_FLA_including_raw_g_exp'])<1e-7
        assert abs((a+b)*.5-order['average_prediction_minus_actual'])<1e-7
        positions=point['input_receipt']['deleted_positions'];actual=row['scoring_points']['0']['original_native_score']-point['original_native_score']
        eps=float(current[positions].sum())-actual;eps_new=float(candidate[positions].sum())-actual
        assert abs(eps+complete['current_error'])<1e-7 and abs(eps_new+complete['candidate_error'])<1e-7
        out['points'][step]={'normgate_terms':ng['terms'],'normgate_measured':ng['measured_prediction_minus_actual'],
          'normgate_groups':ng['token_group_terms'],'normgate_operator_subdivision':ng['operator_subdivision'],
          'normgate_product_subdivision':ng['product_subdivision'],'normgate_max_token_closure':ng['max_abs_token_telescoping_error'],
          'FLA_current':a,'FLA_reverse':b,'FLA_average':(a+b)*.5,'FLA_antisymmetric':(a-b)*.5,
          'whole_current_prediction_minus_actual':eps,'whole_candidate_prediction_minus_actual':eps_new,
          'whole_abs_error_change':abs(eps_new)-abs(eps),'complete_input':complete}
    s['cases'][key]=out
(A/'dt_order_norm_localization_summary_20260908.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
print(json.dumps({'seconds':s['job_seconds'],'cases':{k:v.get('points',v) for k,v in s['cases'].items()}},indent=2))
