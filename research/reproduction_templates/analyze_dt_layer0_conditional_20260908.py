"""Audit layer0 actual-capture identities, signed terms and frozen score inputs."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_layer0_conditional_20260908_v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((D/'results.json').read_bytes());receipt=json.loads((D/'terminal_receipt.json').read_bytes())
assert r['status']=='layer0_conditional_complete' and not receipt['proc_exists']
for n,e in receipt['files'].items():assert sha((D/n).read_bytes())==e['sha256'],n
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    for n in z.namelist():
        assert '/' not in n and '\\' not in n;raw=z.read(n)
        if (D/n).exists():assert (D/n).read_bytes()==raw
        else:(D/n).write_bytes(raw)
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
C=A/'snapshot${ARTIFACT_ROOT}/codex_dt_conditional_boundaries_20260908_v1';coarse=json.loads((C/'results.json').read_bytes())
assert sha((C/'results.json').read_bytes())==r['protocol']['parent_results_sha256']
v=np.load(D/'vectors.npz');v0=np.load(C/'vectors.npz');cases={}
for key,row in r['cases'].items():
    assert np.array_equal(v[key+'_DT_full'],v0[key+'_DT_full']) and row['parent_vectors_equal']
    pointout={}
    for step in [0,1,10,20]:
        x=row['scoring_points'][str(step)];before=coarse['cases'][key]
        assert x['input_receipt']==before['curve']['input_receipts'][step]
        assert x['original_native_score']==before['curve']['scores'][step]
        cap=x['capture'];assert cap['mixer_calls']=={'module':1,'conv':1,'FLA':1,'stage':1}
        assert not cap['initial_cache']['has_previous_state']
        if not step:continue
        d=x['layer0_decomposition'];terms=d['terms'];assert len(terms)==16
        assert abs(sum(terms.values())-d['actual_minus_predicted'])<1e-9
        assert abs(d['actual_minus_predicted']-before['decomposition'][str(step)]['decoder_errors']['0'])<1e-9
        b2=row['B2_endpoint_decomposition']['terms'];b1=row['B1_allEOS_endpoint_decomposition']['terms']
        pointout[str(step)]={'global_actual_minus_predicted':before['decomposition'][str(step)]['actual_minus_predicted'],
            'decoder0_actual_minus_predicted':d['actual_minus_predicted'],'terms':terms,
            'FLA_B2_endpoint_control':b2['GDN_FLA_including_raw_g_exp'],'FLA_B1_allEOS_control':b1['GDN_FLA_including_raw_g_exp'],
            'normgate_B2_endpoint_control':b2['GDN_fused_norm_gate'],'normgate_B1_allEOS_control':b1['GDN_fused_norm_gate'],
            'other_terms_sum':sum(v for k,v in terms.items() if k not in ['GDN_FLA_including_raw_g_exp','GDN_fused_norm_gate'])}
    cases[key]={'points':pointout,'vectors_identical_to_coarse':True,'four_native_scores_identical':True,
        'coefficient_artifact':row['coefficient_artifact'],'paired_feature_CPU_bytes':row['paired_feature_CPU_bytes'],
        'coefficient_CPU_bytes':row['coefficient_CPU_bytes']}
summary={'status':'layer0_original_input_and_signed_decomposition_verified','receipt':receipt,'cases':cases,
    'budget':r['protocol']['budget'],'job_seconds':r['job_seconds'],'FT_modified':False,'finite_rules_modified':False,
    'decision':'Measured decoder0 FLA conditional propagation and fused norm-gate are priority boundaries. FLA includes raw-g exp; gate includes RMS and SiLU. Do not conflate these with proven single-product defects. Early FLA residual offsets gate residual,so global damping or largest-absolute-only ranking is unjustified.',
    'limits':['Two existing cases,three nonclean points each; no quality candidate or independent confirmation.',
        'Current signed terms telescope using frozen upstream coefficients; changing a boundary also changes earlier propagation,so subtracting its error is not an intervention result.',
        'Production coefficients saved privately with hashes; full actual condition activations were transient. Retained scalar contractions alone cannot certify a gate/order candidate.',
        'No added diagnostic conv preactivation calls; actual DT preactivation/backward calls remain part of unchanged production cost.',
        'One actual sample per call; endpoint pair is not multiexample batching.']}
(A/'dt_layer0_conditional_summary_20260908.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':summary['status'],'job_seconds':r['job_seconds'],'step10':{k:x['points']['10'] for k,x in cases.items()}}))
