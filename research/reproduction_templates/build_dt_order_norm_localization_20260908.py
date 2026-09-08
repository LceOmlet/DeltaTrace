"""Freeze one bounded localization using unchanged native/core sources."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
parent=A/'snapshot${ARTIFACT_ROOT}/codex_dt_conditional_boundaries_20260908_v1'
p=json.loads((A/'dt_layer0_conditional_protocol_20260908.json').read_bytes())
files={n:(parent/n).read_bytes() for n in p['files_sha256']}
files['qwen35_dense_finite_runner.py']=(R/'research/runtime/qwen35_dense_finite_runner.py').read_bytes()
files['study.py']=(R/'research/reproduction_templates/dt_order_norm_localization_20260908.py').read_bytes()
for n in ['finite_fla_endpoint_order_diagnostic_20260908.py','normgate_conditional_decomposition_20260908.py']:
    files[n]=(R/'research/reproduction_templates'/n).read_bytes()
for raw in files.values():ast.parse(raw)
for name,want in p['unchanged_finite_runtime_sha256'].items():assert sha(files[name])==want
p.update(scope='Same two existing cases and coarse frozen deletion sets. Quantify layer0 FLA endpoint-order odd component, compute averaged layer0 complete input prediction, and split actual norm-gate conditional error. Not new-sort quality evidence.',
 method_change='Diagnostic layer0 FLA complete endpoint-role average through SAME finite backend: swap every paired field on batch dimension only; same actual do and scale; average six input coefficients, propagate five coordinates without double-counting alpha/g; unchanged GDN surroundings and input RMS once extra. Production core, FT, native forward and backward unchanged.',
 decomposition='Current/reverse/average FLA projected error plus seven-term exact norm-gate CPU decomposition from actual B2/B1 captures; actual allEOS transfer subdivision and token/head geometry. Complete input effect on fixed CURRENT sets; no candidate-new-sort MAS/RISE.',
 budget={'model_loads':1,'native_eager_B1_diagnostics':1,'complete_DT_attributions':2,'native_decoder_replays':64,'finite_decoder_calls':64,
   'current_finite_FLA_calls':48,'additional_reverse_finite_FLA_calls':2,'additional_native_FLA_adjoint_stages':4,
   'additional_GDN_repropagations':2,'extra_diagnostic_conv_preactivation_calls':2,'extra_diagnostic_conv_autograd_calls':2,
   'additional_input_norm_residual_calls':2,'additional_MLP_or_postnorm_calls':0,
   'public_FA_auxiliary_calls':16,'original_metric_curves':0,'native_B1_scoring_forwards':8,
   'FT_traces':0,'generation_calls':0,'new_samples':0,'wall_time_seconds':600},
 acceptance='Keep fixed2% baseline vector sanity guard and verify all eight frozen scorer hashes. Freeze same upstream mo (1e-6 relative numerical guard), exact endpoint fields/scale/counts, source and weight identity. CPU signed closure 1e-7. Save before failure; report B2 and B1 endpoints plus early/middle errors, never promote a candidate based solely on local improvement.',
 stop='One2case diagnostic, preserve failure and no blind retry. No new metric curves or parameter search. Use actual complete-input effects to reject or retain endpoint-order hypothesis.',
 costs='Layer0 diagnostic CPU features/coefficients, two reverse FLA calls and extra GDN/conv propagations explicitly paid. Default FA/FLA precision; CPU FP64 algebra is bookkeeping only. No global token-square tensors or candidate production speed claim.',
 files_sha256={k:sha(v) for k,v in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();dst=A/'dt_order_norm_localization_protocol_20260908.json';assert not dst.exists();dst.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_order_norm_localization_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_order_norm_localization_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'budget':p['budget']}))
