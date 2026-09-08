"""Freeze one layer19 P0 full-input pilot; no dispatch or compilation."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
source_protocol=A/'dt_supported_secant_whole_pilot_protocol_20260909.json'
sp=json.loads(source_protocol.read_bytes())
files={'study.py':(R/'research/reproduction_templates/dt_PV_layer19_whole_pilot_20260909.py').read_bytes()}
optional_names={'qwen35_dense_finite_runner.py','qwen35_decoder_finite.py'}
mapping_changes={}
for name,want in sp['files_sha256'].items():
    if name=='study.py' or name.startswith('vendor_fa_supported_secant'):continue
    matches=[root/name for root in (R/'research/runtime',R/'core',R/'research/reproduction_templates') if (root/name).is_file()]
    assert len(matches)==1,(name,matches)
    raw=matches[0].read_bytes()
    if name in optional_names:
        assert b'content0' in raw
        mapping_changes[name]={'previous_sha256':want,'current_sha256':sha(raw)}
    else:assert sha(raw)==want,('Unexpected shared source change',name)
    files[name]=raw
assert set(mapping_changes)==optional_names
assert b'attention_pv_rules' in files['qwen35_dense_finite_runner.py']
assert b"pv_rule='content1'" in files['qwen35_decoder_finite.py']
for name,raw in files.items():ast.parse(raw,filename=name)
pv=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_PV_orders_20260909_v2'
rr=json.loads((pv/'results.json').read_bytes())
assert rr['status']=='MH0_FA19_PV_orders_1publicFA2finiteFA_local_complete'
assert rr['native_FA_returned']==1 and rr['finite_FA_returned']==2
assert all(row['bitwise_equal'] and row['relative_L2']==0 for row in rr['same_LSE_control_vs_saved_coefficient_drift'].values())
local_effects={step:{method:rr['points'][step]['methods'][method]['fields']['core_prediction_minus_actual']
                    for method in ('same_LSE_control','reversed','diagnostic_average')} for step in ('3','10','20','B2')}
remote='${ARTIFACT_ROOT}/codex_dt_PV_layer19_whole_pilot_20260909_v1'
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p=copy.deepcopy(sp)
for key in ('candidate_library','candidate_library_sha256','mathematical_and_numerical_definition','historical_scope'):
    p.pop(key,None)
protected=[row for row in sp['protected_sources'] if 'supported_secant' not in row['path']]
protected += [{'path':'/tmp/'+pv.name+'/'+name,'sha256':sha((pv/name).read_bytes())} for name in ('results.json','protocol.json')]
p.update({
 'scope':'Only decoder19 PV interaction allocation changes from content1 to content0 using the same original finite FA library once. Both methods retain current layer0 norm symmetric plus complete FLA endpoint average. Other7FA remain content1. No supported-secant rule, new library/native operator, score transform or generation.',
 'attention_pv_rules':{'control':{},'candidate':{'19':'content0'}},
 'rule_scope':{'control':'Current original logmean FA content1 at all8FA with empty attention_pv_rules, common layer0 symmetric norm and finite FLA endpoint average.',
               'candidate':'Same current method except attention_pv_rules={19:content0}; same actual backend object/library. The existing q/k/LSE endpoint pairs swap and V0 slot uses real V1 only at decoder19. Upstream, QK midpoint, logmean rule, model, original captured gate/projection endpoints, and other7FA remain as control.'},
 'hypothesis':'Current MH0 actual FA19 hybrid contrasts locate large PV reference-content mismatch; direct same-backend endpoint-order local test greatly reduced early/mid complete FAcore error with zero current coefficient replay drift. This supports one explicit layer19 development candidate. Conditional local improvement does not prove full-input or original-metric improvement; downstream finite propagation, signed cancellation and NI needle must be checked.',
 'local_candidate_evidence':{'results_path':'/tmp/'+pv.name+'/results.json','results_sha256':sha((pv/'results.json').read_bytes()),
    'same_LSE_control_coefficients_bitwise_equal':True,'actual_local_core_effects':local_effects,
    'scope':'Actual saved MH0 layer19 endpoints and upstream only. Negative/positive terms retained. Smaller local net error is evidence for a candidate, not a MAS result or proof of token credit.'},
 'selection_scope':'Layer19 was chosen from actual current MH0 conditional boundary and operator evidence. This is a used-case development choice, not a general all-layer/model rule, independent heldout validation or layer-selection sweep. NI1 is the fixed original needle protection case.',
 'runtime_optional_mapping_change':mapping_changes,
 'runtime_change_scope':'Two attribution files add an explicit optional validated layer->PV rule mapping and content0 operand selection. Default remains content1 with empty mapping. Runner and decoder mathematical changes are reviewed separately before root dispatch. No official FT/native model/FA/FLA source or library is modified.',
 'protected_sources':protected,
 'budget':{'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_attributions':4,
   'control_DT':2,'candidate_DT':2,'DT_B2_roots':4,'native_decoder_replays':128,'finite_decoder_calls':128,
   'public_FA_LSE_calls':32,'finite_FA_calls':32,'original_finite_FA_calls':32,'finite_FA_phases':96,
   'content0_FA_calls':2,'content1_FA_calls':30,'finite_GDN_callback_sites':96,'finite_FLA_backend_calls':100,
   'native_FLA_adjoint_stage_calls':200,'common_layer0_average_wrapper_calls':4,'common_additional_FLA_backend_calls':4,
   'finite_GDN_conv_preactivation_calls':96,'finite_GDN_conv_autograd_calls':96,'original_metric_curves':4,
   'native_B1_scorer_forwards':84,'FT_calls':0,'generation_calls':0,'new_FA_extension_compiles':0,
   'new_candidate_operators':0,'new_samples':0,'repeat_timing_runs':0,'wall_time_seconds':600},
 'source_reuse':'Exact frozen supported-pilot input, author remap and original evaluator orchestration, which reused successful current-profile stability/clean-secant flows. Supported backend removed entirely from payload. All remaining dependencies byte-identical except two explicit reviewed optional-PV mapping files; original vendor wrapper/library shared between methods.',
 'cost':'MH0 C->P0, NI1 P0->C, one observation per method/case in one resident full model. Identical finite FA core/phase counts and existing three-phase GEMMs; P0 uses a different actual reference V endpoint and pair selection, not an extra backend. All default native/replay/finite operations, common endpoint-average overhead, CPU copies and peaks remain timed. No focused layer observer or additional activation capture. Pilot cost only; no proven stable speed equality.',
 'candidate_data_boundary':'Candidate finite path receives only the normal original B2 endpoints and production upstream. Explicit layer19 selection is a frozen configuration, not dispatch by execution count or by observed deletion outcomes. Deletion activations, masks, gold and historical attribution vectors never enter the candidate operator.',
 'launch_prerequisite':'PREPARED ONLY. Actual PVv2 local evidence has been recorded; root must review this exact new mapping, study and payload before dispatch. The separately frozen supported-secant whole pilot remains prepared and is not chained or executed by this study.',
 'deferred_supported_pilot':{'local_protocol_file':source_protocol.name,'sha256':sha(source_protocol.read_bytes()),'status':'prepared_not_launched_by_this_study'},
 'decision':'Complete raw input-mask prediction-minus-actual ledger, original RISE/MAS and NI1 needle determine next action, not local FAcore magnitude or endpoint conservation alone. Report every regression, positive/negative cancellation, actual cost and root/scorer equality. No auto full-layer replacement, combined softmax change, extra cases, timing repeats or layer/weight scan.',
 'stop':'First source/input/span/layout/count/nonfinite/original-metric invariant failure or600seconds; preserve partial evidence. No retry, fallback, data substitution, final score clipping, changed precision, extra operator/score, FT change or additional candidate.',
 'files_sha256':{name:sha(raw) for name,raw in files.items()}})
assert p['budget']['content0_FA_calls']+p['budget']['content1_FA_calls']==32
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_PV_layer19_whole_pilot_protocol_20260909.json';lp=A/'launch_dt_PV_layer19_whole_pilot_20260909.json'
assert not pp.exists() and not lp.exists(),'Never overwrite a frozen payload.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
 'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget'],'mapping_changes':mapping_changes}))
