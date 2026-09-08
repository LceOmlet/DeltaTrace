"""One same-forward repair comparison; original inputs/metrics, fixed FT."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_layer0_conditional_20260908_v1';p=json.loads((D/'protocol.json').read_bytes())
files={n:(D/n).read_bytes() for n in p['files_sha256']}
modified={}
for name in ['qwen35_gdn_finite.py','finite_fla_gpu.py']:
    raw=(R/'research/runtime'/name).read_bytes();modified[name]={'previous_sha256':p['unchanged_finite_runtime_sha256'].pop(name),'new_sha256':sha(raw)}
    files[name]=raw
files['study.py']=(R/'research/reproduction_templates/dt_normgate_repair_NI1_20260908.py').read_bytes()
for name in ['layer0_normgate_paired_repair_20260908.py','finite_fla_coefficient_capture_20260908.py','fla_beta_content_mismatch_audit_20260908.py']:
    files[name]=(R/'research/reproduction_templates'/name).read_bytes()
for raw in files.values():ast.parse(raw)
for name,want in p['unchanged_finite_runtime_sha256'].items():assert sha(files[name])==want
p.pop('observer_vector_relative_L2_ceiling',None)
quality=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v2/results.json'
p.update(scope='Single original NI1 paired layer0 norm-gate repair, with original two new-order metric curves and actual control-content states. Old failed MH2% experiment remains failed; this is not its retry.',
 cases=['niah_mq_q2_1'],case_indices=[['niah_mq_q2',1]],
 quality_parent_path='${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v2/results.json',quality_parent_sha256=sha(quality.read_bytes()),
 method_change='Only layer0 attribution norm-times-SiLU uses symmetric endpoint allocation. Current and candidate share same actual root/replayed endpoints and actual decoder upstream; one extra GDN/inputnorm propagation for the paired comparison. RMS,FLA,MLP,other layers and native model unchanged. Default production content1 operations statically identical.',
 diagnostic_change='mixed_coefficients returns its already computed packed L/r0 only when diagnostics=True; first23GDN use ordinary backend, layer0current/candidate use same capture compiled mixed. Native two adjoints unchanged and not duplicated per call. Actual B1 g/h/v_new retained for CPU algebra, no alternate recurrence.',
 modified_attribution_runtime=modified,
 decomposition='Current/candidate full input effect on same frozen old deletion sets and16term actual layer0 ledgers; next CPU beta local mismatch uses real captured native chunk states and actual L/r0. No branch-score proxy for conditional residual.',
 budget={'model_loads':1,'native_eager_B1_diagnostics':1,'complete_DT_attributions':1,'native_decoder_replays':32,'finite_decoder_calls':32,
   'current_finite_FLA_calls':24,'additional_candidate_finite_FLA_calls':1,'additional_GDN_repropagations':1,
   'extra_diagnostic_conv_preactivation_calls':1,'extra_diagnostic_conv_autograd_calls':1,'additional_input_norm_residual_calls':1,
   'public_FA_auxiliary_calls':8,'original_metric_curves':2,'frozen_original_score_calls':4,'native_B1_scoring_forwards':46,
   'FT_traces':0,'generation_calls':0,'new_samples':0,'wall_time_seconds':600,'subsequent_CPU_beta_audit_seconds':180},
 acceptance='Verify same actual d/c/e/upstream identities and shape/scale; fixed mnorm relative1e-6 guard; source,weights,input+gold mapping and official scorer function identity. Use actual full candidate propagation,not subtracting local residual. Report previous vector drift but no old-vector admission threshold in this new paired protocol. Endpoint/current/candidate signed ledgers close1e-7. Compare actual new-sort original RISE/MAS and needle; no automatic promotion from local or onecase improvement.',
 stop='Only one original NI1 pilot. Preserve first runtime failure and partial artifacts, no blind retry. If MAS does not improve, do not expand to MH without a new mechanism decision. Any needle/RISE decline and local group cancellation are explicit adverse evidence; no score calibration or coefficient sweep.',
 costs='Full timings and peak include diagnostic duplication/captures/compilation, not production-candidate overhead. Normal symmetric candidate substitutes same GDN/FLA calls,adds only linear n1/means. Private full L/r0/control states and two coefficient dictionaries saved with hashes; no public prompts or activations. EOS pair is one actual example,not multiexample batch.',
 files_sha256={k:sha(v) for k,v in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();dst=A/'dt_normgate_repair_NI1_protocol_20260908.json';assert not dst.exists();dst.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_normgate_repair_NI1_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_normgate_repair_NI1_20260908.json').write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'budget':p['budget']}))
