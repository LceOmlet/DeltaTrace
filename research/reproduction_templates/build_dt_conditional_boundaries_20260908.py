"""Freeze two original-case diagnostics; no method candidate or FT run."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
parent=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v2';old=json.loads((parent/'protocol.json').read_bytes())
head=A/'snapshot${ARTIFACT_ROOT}/codex_dt_native_logit_rows_NI0_20260908_v1';p=json.loads((head/'protocol.json').read_bytes())
for k in ['run_order','scope','input_sha256','method_change','target','precision','timing','acceptance','batching','stop','budget','files_sha256']:p.pop(k,None)
for k in ['cache_paths','cache_hashes']:p[k]=old[k]
files={name:(head/name).read_bytes() for name in list(p['unchanged_finite_runtime_sha256'])+['native_dense_attention_capture.py','native_target_logit_rows.py','fixed_input_metric_view.py']}
files['qwen35_dense_finite_runner.py']=(R/'research/runtime/qwen35_dense_finite_runner.py').read_bytes()
files['study.py']=(R/'research/reproduction_templates/dt_conditional_boundaries_20260908.py').read_bytes()
for raw in files.values():ast.parse(raw)
for name,want in p['unchanged_finite_runtime_sha256'].items():assert sha(files[name])==want
p.update(scope='NI1 and MH1 existing original cases, conditional-error localization using unchanged DT and original 20-step scoring; not independent confirmation or a new method test.',
    cases=['niah_mq_q2_1','morehopqa_1'],case_indices=[['niah_mq_q2',1],['morehopqa',1]],capture_steps=[0,1,10,20],
    parent_results_path='${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v2/results.json',parent_results_sha256=sha((parent/'results.json').read_bytes()),
    method_change='None. Current native selected output rows; optional detached CPU observer of ACTUAL production coefficients and ACTUAL endpoint/score activations. Original model,FA,FLA,finite modules,full vocabulary,targets,and FT unchanged.',
    decomposition='Actual scorer B1 clean-minus-deleted hidden contractions with frozen DT B2 coefficients. Coarse head+logprob seed combined, final norm,32decoders,and input map telescope to original effect-minus-prediction; original BF16-vs-FP32 score difference separately recorded. B1 allEOS and same-run B2 endpoint controls retained. No theoretical seed replaces actual coefficients.',
    precision='Default native BF16/FA. CPU FP64 contractions only for diagnostic bookkeeping; does not alter production coefficients or original scorer.',
    budget={'model_loads':1,'native_eager_B1_diagnostics':1,'complete_DT_attributions':3,'native_decoder_replays':96,'finite_decoder_calls':96,
        'public_FA_auxiliary_calls':24,'original_metric_curves':2,'native_B1_scoring_forwards':42,'FT_traces':0,'generation_calls':0,'new_samples':0,'wall_time_seconds':600},
    observer_vector_relative_L2_ceiling=0.02,
    acceptance='First NI1 has unobserved+observed full attributions to measure observer drift; save before fixed2% stop. Telescope closes numerically. No quality candidate is promoted from a diagnostic, no FT changes, no arbitrary gate sweep. Report signed and cancelling terms; decoder category includes MLP/norm,not a pure attention claim.',
    stop='Only two existing cases and fixed steps; stop on first failure, preserve partial results. No blind retry or sample expansion. If coarse error does not locate a core, report unresolved rather than promise a fix.',
    costs='All copy/CPU contraction/score-hook costs included as diagnostics; no speedup claim. Hidden coefficient/clean capture CPU O(LTd),GPUone-layer temporary; output logprob diagnostic temporary target_rows*full_vocab counted.',
    files_sha256={k:sha(v) for k,v in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();dst=A/'dt_conditional_boundaries_protocol_20260908.json'
assert not dst.exists();dst.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_conditional_boundaries_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_conditional_boundaries_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'budget':p['budget']}))
