"""Freeze one actual MH DT plus4original score captures; do not launch."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_normgate_production_20260908_v1';p=json.loads((D/'protocol.json').read_bytes())
files={name:(D/name).read_bytes() for name in p['files_sha256'] if name!='study.py'}
for name,value in files.items():
    assert sha(value)==p['files_sha256'][name],name
    source=R/'research/runtime'/name
    if not source.exists():source=R/'research/reproduction_templates/dt_normgate_production_20260908_v1'/name
    assert sha(source.read_bytes())==sha(value),('runtime changed from authoritative production',name)
files['study.py']=(R/'research/reproduction_templates/dt_decoder19_6_conditional_20260908.py').read_bytes()
helper='decoder19_conditional_decomposition_20260908.py';files[helper]=(R/'research/reproduction_templates'/helper).read_bytes()
for name,value in files.items():ast.parse(value,filename=name)
M=A/'snapshot${ARTIFACT_ROOT}/codex_dt_normgate_production_MH_metrics_20260908_v1'
raw=(M/'results.json').read_bytes();assert sha(raw)=='3b5d2454561b7a7ef6912f1811667cbd1f1a3e32f0277fe54167d3c5dc7eb9a3'
metrics=json.loads(raw);assert metrics['status']=='production_MH_original_metrics_complete'
curve=metrics['cases']['morehopqa_1']['curves']['candidate'];assert curve['production_run']==7 and len(curve['scores'])==21
mp=json.loads((M/'protocol.json').read_bytes())
for name in ['native_model_sha256','installed_FA_interface_sha256','runtime_source_sha256','native_stage_source_sha256',
             'checkpoint_config_tokenizer_sha256','expected_weight_stats','official_source_blob_sha1','checkpoint','cache_paths','cache_hashes']:
    assert mp[name]==p[name],name
production_raw=(D/'results.json').read_bytes();production=json.loads(production_raw)
assert production['status']=='eight_call_production_integration_cost_complete'
for name in ['paired_references','production_norm_gate_rules','call_schedule','modified_production_runtime']:
    p.pop(name,None)
p.update(scope='One actual normal symmetric-layer0 MH1 DT plus4frozen original scorer forwards, observing decoder19FA/6GDN and coarse boundaries. No method change or quality claim.',
    source_authority='6a6f732e45366cbdcfc74c8019b8aac8f1e47beb runtime files; every runtime byte checked against repository and successful production snapshot.',
    cases=['morehopqa_1'],case_indices=[['morehopqa',1]],capture_steps=[0,1,10,20],selected_layers=[6,19],
    coarse_boundaries=['0','1','6','7','19','20','32','norm'],norm_gate_rules={'0':'symmetric'},
    frozen_metrics_path='${ARTIFACT_ROOT}/codex_dt_normgate_production_MH_metrics_20260908_v1/results.json',frozen_metrics_sha256=sha(raw),
    production_results_path='${ARTIFACT_ROOT}/codex_dt_normgate_production_20260908_v1/results.json',production_results_sha256=sha(production_raw),
    production_vectors_path='${ARTIFACT_ROOT}/codex_dt_normgate_production_20260908_v1/vectors.npz',production_vectors_sha256=sha((D/'vectors.npz').read_bytes()),
    method_change='None. Normal runner norm_gate_rules={0:symmetric}; current same32finite passes and existing optional diagnostics on layers6and19. No additional finite or native operator; no changed FT/native model/FA/FLA implementation.',
    diagnostic_change='GDN existing16term ledger converted explicitly to prediction-minus-actual; FA helper uses actual finite coefficients and native captures,groups outputprojection+sigmoid gate because its separate preprojection multiplier is not exposed. No guessed inverse or extra matrix multiplication. B1 profilers scoped serially inside actual layers6then19; coarse capture uses module hooks.',
    budget={'model_loads':1,'native_eager_B1_diagnostics':1,'complete_DT_attributions':1,'native_root_forwards':1,
        'native_decoder_replays':32,'finite_decoder_calls':32,'finite_FLA_calls':24,'finite_FA_calls':8,'public_FA_auxiliary_calls':8,
        'selected_decoder_diagnostics':2,'native_B1_scoring_forwards':4,'original_metric_curves':0,
        'extra_candidate_propagations':0,'extra_operator_calls':0,'FT_traces':0,'generation_calls':0,'new_samples':0,'wall_time_seconds':600},
    acceptance='Frozen actual current/candidate root sources/weights/data hold; scorer input hashes match latest production candidate bins0/1/10/20. B1FA Q/K/V actual full lengths prove freshfullsequence capture without changing scorer cache kwargs. Both ledgers andcoarse telescope close1e-7; all terms prediction-minus-actual. Report actual signed-input predictions,rootB2 andB1score drift; historical2percent vector guard not used. Actual observations only,noquality or repaired-method claim.',
    stop='One DT/four scores only; preserve first exception and partialCPU capture,entered/returned callbacks and unknown unfinished native ledger. No extra replay or scorecurve,no alternative initialization,nativeframework edit,orautomatic newcandidate.',
    costs='Complete normal attribution path plusCPU observational copies/contractions; not a new production-speed measurement. Private artifact keeps selected actual paired/scoring features,existingcoefficients,MLPgate/up/silu/downinput andsmall originalnorm weights/eps includingGDNfusednorm andFAq/knorm; copiedparameter bytes recorded. No largeMLPweight copy/newGEMM. Privateartifact excluded frompublicreviewZIP.',
    files_sha256={name:sha(value) for name,value in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();protocol=A/'dt_decoder19_6_conditional_protocol_20260908.json'
launch=A/'launch_dt_decoder19_6_conditional_20260908.json';assert not protocol.exists() and not launch.exists();protocol.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(value).decode() for name,value in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_decoder19_6_conditional_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
launch.write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'helper_sha256':p['files_sha256'][helper],
    'launch_payload':str(launch),'remote_directory':directory,'budget':p['budget']}))
