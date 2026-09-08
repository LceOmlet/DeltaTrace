"""Freeze one existing supported-secant whole pilot; never launch or compile."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_layer0_fla_endpoint_average_stability_cost_20260909_v1'
dp=json.loads((donor/'protocol.json').read_bytes())
local=A/'snapshot${ARTIFACT_ROOT}/codex_dt_fa19_supported_secant_20260908_v1'
lr=json.loads((local/'results.json').read_bytes());lb=json.loads((local/'build_results.json').read_bytes())
assert lr['status']=='one_supported_secant_saved_layer_candidate_observed'
assert lb['status']=='finite_extension_compiled_not_executed'
assert sha((local/'results.json').read_bytes())==json.loads((local/'terminal_receipt.json').read_bytes())['files']['results.json']['sha256']
files={'study.py':(R/'research/reproduction_templates/dt_supported_secant_whole_pilot_20260909.py').read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(donor/name).read_bytes();assert sha(raw)==want
    locations=[root/name for root in (R/'core',R/'research/runtime',R/'research/reproduction_templates') if (root/name).is_file()]
    assert len(locations)==1 and locations[0].read_bytes()==raw,('Common completed-study source changed',name)
    files[name]=raw
for name in ('vendor_fa_supported_secant_20260908.py','vendor_fa_supported_secant_runner_20260909.py'):
    files[name]=(R/'research/runtime'/name).read_bytes()
assert sha(files['vendor_fa_supported_secant_20260908.py'])==lb['protocol']['files_sha256']['vendor_fa_supported_secant_20260908.py']
for name,raw in files.items():ast.parse(raw,filename=name)
records=copy.deepcopy(dp['fixed_records']);reference_jobs=[]
for shard,keys in [(0,('niah_mq_q2_0','morehopqa_0')),(1,('niah_mq_q2_1',))]:
    folder=A/f'snapshot${ARTIFACT_ROOT}/codex_dt_fixed_eight_case_regression_20260909_s{shard}_v1'
    result=json.loads((folder/'results.json').read_bytes());pr=json.loads((folder/'protocol.json').read_bytes())
    for key in ('checkpoint','cache_paths','cache_hashes','checkpoint_config_tokenizer_sha256','native_model_sha256','official_source_blob_sha1'):
        assert dp[key]==pr[key],key
    for key in keys:
        records[key]['expected_input']=result['cases'][key]['input']
        records[key]['expected_gold']=result['cases'][key]['gold']
        records[key]['provenance']='Same fixed original cache/tokenizer and source identity as completed eight-case study. Only input/gold identity consumed; no prior scores or attribution vectors are substituted.'
    reference_jobs.append({'path':'/tmp/'+folder.name+'/results.json','sha256':sha((folder/'results.json').read_bytes())})
for key,record in records.items():
    dataset,index=key.rsplit('_',1);raw=(A/'snapshot'/Path(dp['cache_paths'][dataset].lstrip('/'))).read_bytes()
    assert sha(raw)==dp['cache_hashes'][dataset]
    assert sha(raw.decode().splitlines()[int(index)].encode())==record['source_record_sha256']
remote='${ARTIFACT_ROOT}/codex_dt_supported_secant_whole_pilot_20260909_v1'
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p=copy.deepcopy(dp)
for key in ('endpoint_rule','diagnostic_contract_provenance','prior_wholepilot','NI0_old_input_provenance'):
    p.pop(key,None)
quality=[['morehopqa_0','control'],['morehopqa_0','candidate'],['niah_mq_q2_1','candidate'],['niah_mq_q2_1','control']]
lib='/tmp/'+local.name+'/'+lb['protocol']['diagnostic_library_name']
diagnostic=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_qk_softmax_diagnostic_20260909_v1'
assert (diagnostic/'results.json').is_file()
protected=[{'path':dp['finite_FA_library'],'sha256':dp['finite_FA_library_sha256']},
           {'path':lib,'sha256':lb['library']['sha256']}]
for name in ('results.json','build_results.json','protocol.json'):
    protected.append({'path':'/tmp/'+local.name+'/'+name,'sha256':sha((local/name).read_bytes())})
source_name=lb['protocol']['diagnostic_extension_name']
assert sha((local/source_name).read_bytes())==lb['protocol']['files_sha256'][source_name]
protected.append({'path':'/tmp/'+local.name+'/'+source_name,'sha256':sha((local/source_name).read_bytes())})
protected+=reference_jobs
protected.append({'path':'/tmp/'+diagnostic.name+'/results.json','sha256':sha((diagnostic/'results.json').read_bytes())})
p.update({
 'scope':'One already compiled supported-secant softmax rule uniformly at all8FA layers. Both methods retain current norm_gate_rules={0:symmetric} and finite_fla_by_layer={0:unchanged endpoint average}. No changed PV=P1,QK midpoint,native FA/FLA,model,FT,scorer,or final score transform.',
 'case_indices':[['niah_mq_q2',0],['morehopqa',0],['niah_mq_q2',1]],
 'quality_cases':['morehopqa_0','niah_mq_q2_1'],'quality_schedule':quality,
 'call_schedule':[x+['quality'] for x in quality], 'fixed_records':records,
 'candidate_library':lib,'candidate_library_sha256':lb['library']['sha256'],
 'protected_sources':protected,
 'rule_scope':{'control':'Current logmean finite FA on all8FA plus common layer0 symmetric norm and complete finite FLA endpoint average.',
               'candidate':'Same current control except explicit AllFASupportedSecantBackend injected for all8FA; unchanged compiled original candidate library and wrapper. Adapter only changes signature and records counts/shape/bytes, without coefficient arithmetic.'},
 'hypothesis':'Actual current MH0 FA19 and historical MH1 local evidence identify a large off-pair softmax conditional prediction error. Probability-supported correction solves the same ideal local-response/secant constraints while its correction follows probability difference. This addresses a proven Euclidean support weakness; neither rare-key spreading as the observed NI failure cause nor whole-model quality improvement has been established. Uniform operator use avoids fitting a layer to MH0.',
 'mathematical_and_numerical_definition':{key:lb['protocol'][key] for key in ('finite_rule','FP32_actual_definition','ideal_minimum_metric_proof','support_bound_proof','normalization_scope','stable_denominator','degenerate_rule')},
 'historical_scope':'The Euclidean all8FA pilot genuinely regressed NI1 and is not this candidate. Supported secant was tested only at old MH1 FA19; a local early error increase alone did not disprove complete-input value. This is the first whole-model pilot of this specific supported rule and current common FLA repair, not a guaranteed recovery.',
 'budget':{'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_attributions':4,
   'control_DT':2,'candidate_DT':2,'DT_B2_roots':4,'native_decoder_replays':128,'finite_decoder_calls':128,
   'public_FA_LSE_calls':32,'finite_FA_calls':32,'original_finite_FA_calls':16,'supported_finite_FA_calls':16,
   'finite_FA_phases':96,'finite_GDN_callback_sites':96,'finite_FLA_backend_calls':100,
   'native_FLA_adjoint_stage_calls':200,'common_layer0_average_wrapper_calls':4,
   'common_additional_FLA_backend_calls':4,'finite_GDN_conv_preactivation_calls':96,'finite_GDN_conv_autograd_calls':96,
   'original_metric_curves':4,'native_B1_scorer_forwards':84,'FT_calls':0,'generation_calls':0,
   'new_FA_extension_compiles':0,'new_candidate_operators':0,'new_samples':0,'repeat_timing_runs':0,'wall_time_seconds':600},
 'source_reuse':'Common runner, finite FLA/norm/projection rules, original model capture, original metrics and span mapping are byte-identical to the completed current-profile stability study. Supported source/wrapper/library are pinned to the already executed local candidate. No compiler process or native monkeypatch is introduced; existing common compiler caches remain as in normal attribution.',
 'cost':'Opposite method order across two cases; one observation per method/case, not a stable speed benchmark. Both common endpoint-average overheads remain. Original supported wrapper has a device synchronization and21 FP32 per-row states, retained in actual latency/peak memory. Adapter adds no GPU operation or CPU row copy. All resident-model peaks, actual decoder stages, setup and common CPU copies retained; no empty_cache or model unloading between methods.',
 'candidate_data_boundary':'Finite candidate receives only normal original B2 endpoint Q/K/V/LSE and production upstream. It never receives partial deletion activations, scores, gold, reference predictions, layer residuals or user-targeted masks. Gold/deletion masks enter only the unchanged evaluator after normal complete attribution.',
 'evaluation_scope':'Exactly84 fresh original B1 scorer calls on the two actual author-processed records and each newly produced vector own21step deletion curve. NI1 author needle retained; MH0 has no invented gold. Same raw prompt+fixed target+EOS pipeline, same evaluator and full signed vectors. No historical scores substituted; fixed control-mask CPU contractions use current-run control scores only.',
 'launch_prerequisite':'PREPARED ONLY. Root must finish interpreting the separately frozen MH0 PV-order result and review this exact payload before dispatch. Preparation does not authorize execution; absence of a prior supported whole test alone is not a reason to launch. No automatic chaining from PV or QK diagnostics.',
 'decision':'Report MH0 and NI1 separately with original RISE/MAS lower-is-better and NI1 needle. A quality regression blocks promotion of this uniform candidate. No automatic repeat, alternative PV combination, layer selection, coefficient weight sweep or extension of the two cases. Local numerical invariants do not certify attribution quality.',
 'stop':'First source/input/span/layout/count/nonfinite/original-metric invariant failure or600seconds. Preserve completed and partial vectors/counts. No retry, fallback backend, precision change, clip, substitute data, extra scoring, native/FT modification or new candidate.',
 'files_sha256':{name:sha(raw) for name,raw in files.items()}})
assert p['budget']['finite_FLA_backend_calls']==4*25 and len(quality)*21==84
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_supported_secant_whole_pilot_protocol_20260909.json';lp=A/'launch_dt_supported_secant_whole_pilot_20260909.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen payload.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
 'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget'],'launch_prerequisite':p['launch_prerequisite']}))
