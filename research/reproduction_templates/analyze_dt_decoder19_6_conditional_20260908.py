"""Verify actual selected-decoder ledgers and identify the largest measured terms."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_decoder19_6_conditional_20260908_v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((D/'results.json').read_bytes());p=r['protocol'];receipt=json.loads((D/'terminal_receipt.json').read_bytes())
assert r['status']=='decoder19_6_actual_conditional_observation_complete' and not receipt['proc_exists']
for n,want in p['files_sha256'].items():
    raw=(D/n).read_bytes();assert sha(raw)==want,n;ast.parse(raw)
for n,e in receipt['files'].items():
    raw=(D/n).read_bytes();assert sha(raw)==e['sha256'] and len(raw)==e['bytes'],n
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
assert r['DT_calls']==r['DT_calls_entered']==1 and r['scoring_forwards_entered']==r['scoring_forwards_returned']==4
assert r['FT_calls']==r['generation_calls']==r['extra_operator_calls']==0
assert r['finite_callback_counts']=={'finite_FA':{'entered':8,'returned':8},'finite_FLA':{'entered':24,'returned':24}}
details=r['DT_details'];assert details['norm_gate_rules']=={'0':'symmetric'}
kinds=[c['kind'] for c in details['calls']]
assert sum(s.startswith('native_replay_') for s in kinds)==32
assert sum(s.startswith('public_FA_LSE_') for s in kinds)==8
assert sum(s.startswith('finite_decoder_') for s in kinds)==32
assert p['capture_steps']==[0,1,10,20] and p['selected_layers']==[6,19]
parent=A/'snapshot${ARTIFACT_ROOT}/codex_dt_normgate_production_MH_metrics_20260908_v1/results.json'
assert sha(parent.read_bytes())==p['frozen_metrics_sha256'];old=json.loads(parent.read_bytes())['cases']['morehopqa_1']['curves']['candidate']
vectors=np.load(D/'vectors.npz',allow_pickle=False);full=vectors['current_full'];evaluated=vectors['current_evaluated']
assert np.array_equal(full[:r['input']['prompt_length']].astype(np.float32),evaluated)
assert np.isfinite(full).all() and np.isfinite(evaluated).all()
out={'status':'actual_decoder19_6_conditional_ledgers_independently_checked','result_sha256':sha((D/'results.json').read_bytes()),
 'sign_convention':'prediction_minus_actual','actual_budget':{'model_loads':1,'native_eager_warms':1,'complete_DT_calls':1,
 'native_decoder_replays':32,'finite_FA_calls':8,'finite_FLA_calls':24,'auxiliary_FA_calls':8,'original_scorer_forwards':4,
 'extra_operator_calls':0,'FT_calls':0,'generation_calls':0,'new_samples':0},
 'job_seconds':r['job_seconds'],'current_vector_vs_prior_production':r['production_vector_drift_report_only'],
 'B2_root_effect':r['B2_actual_root_effect'],'prior_B2_root_effect':r['prior_production_root_effect'],
 'B2_decompositions':r['B2_decompositions'],'private_artifact':r['private_artifact'],'points':{}}
for step in ['0','1','10','20']:
    point=r['points'][step];assert point['input_receipt']==old['input_receipts'][int(step)]
    assert point['prior_actual_native_score']==old['scores'][int(step)]
    assert point['native_score_minus_prior']==point['actual_native_score']-point['prior_actual_native_score']
    for layer,rec in point['capture_receipts'].items():
        assert rec['decoder_calls']=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}
        expected=({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if layer=='19' else {'module':1,'conv':1,'FLA':1,'stage':1})
        assert rec['mixer_calls']==expected
        if layer=='6':assert not rec['initial_cache']['has_previous_state'] and rec['initial_cache']['layer_index']==6
    q={'actual_native_score':point['actual_native_score'],'native_score_minus_prior':point['native_score_minus_prior']};out['points'][step]=q
    if step=='0':continue
    deleted=point['input_receipt']['deleted_positions'];actual=r['points']['0']['actual_native_score']-point['actual_native_score']
    pred=float(evaluated[deleted].astype(np.float64).sum())
    assert abs(pred-point['complete_input']['evaluated_prediction'])<1e-8
    assert actual==point['complete_input']['actual_effect']
    assert abs(pred-actual-point['complete_input']['prediction_minus_actual'])<1e-8
    coarse=point['coarse'];a=coarse['boundary_contractions'];regions=coarse['regions']
    names=['0','1','6','7','19','20','32','norm'];assert set(a)==set(names)
    for l,h in zip(names,names[1:]):assert abs(regions[l+'_to_'+h]-(a[l]-a[h]))<1e-8
    assert abs(regions['norm_to_actual_score']-(a['norm']-actual))<1e-8
    assert abs(sum(regions.values())-(a['0']-actual))<1e-7
    q.update(complete_input=point['complete_input'],coarse_regions=regions,
      coarse_positive_error=sum(max(0,v) for v in regions.values()),coarse_negative_error=sum(min(0,v) for v in regions.values()),layers={})
    for layer,ledger in point['layer_decompositions'].items():
        assert ledger['sign_convention']=='prediction_minus_actual'
        error=ledger['input_contraction']-ledger['output_contraction'];terms=ledger['terms']
        assert abs(error-ledger['prediction_minus_actual'])<1e-8 and abs(sum(terms.values())-error)<1e-7
        assert abs(error-regions[layer+'_to_'+str(int(layer)+1)])<1e-6
        ranked=sorted(terms.items(),key=lambda x:abs(x[1]),reverse=True)
        q['layers'][layer]={'error':error,'positive_terms':sum(max(0,v) for v in terms.values()),
          'negative_terms':sum(min(0,v) for v in terms.values()),'terms':terms,'largest_absolute_terms':ranked[:5],
          'coarse_vs_focused_difference':error-regions[layer+'_to_'+str(int(layer)+1)]}
out['scope']='One previously used author MH1 case, actual current same-pass coefficients and original frozen deletion inputs. Ledger closure is an accounting check, not proof of intervention success. Native/B2/B1 drift retained. No new curve, MAS improvement, or culprit inside a combined boundary is inferred.'
out['next']='Use the largest actual early/middle operator discrepancy to select a mechanism-specific diagnostic or minimal intervention; retain signed compensation and allEOS controls. No rule sweep or benchmark expansion follows automatically.'
(A/'dt_decoder19_6_conditional_summary_20260908.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':out['status'],'seconds':out['job_seconds'],'points':{k:v for k,v in out['points'].items() if k in ['1','10']}}))
