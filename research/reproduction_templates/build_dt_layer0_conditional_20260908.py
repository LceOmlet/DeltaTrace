"""Freeze the measured layer0 follow-up; no new quality candidate."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
parent=A/'snapshot${ARTIFACT_ROOT}/codex_dt_conditional_boundaries_20260908_v1';p=json.loads((parent/'protocol.json').read_bytes())
files={n:(parent/n).read_bytes() for n in p['files_sha256']}
files['qwen35_dense_finite_runner.py']=(R/'research/runtime/qwen35_dense_finite_runner.py').read_bytes()
files['study.py']=(R/'research/reproduction_templates/dt_layer0_conditional_20260908.py').read_bytes()
for raw in files.values():ast.parse(raw)
for name,want in p['unchanged_finite_runtime_sha256'].items():assert sha(files[name])==want
p.update(scope='Focused decoder0 internals because two actual coarse step10 diagnostics located large error there. Same two existing cases and frozen original scorer inputs; not a method or benchmark change.',
 parent_results_path='${ARTIFACT_ROOT}/codex_dt_conditional_boundaries_20260908_v1/results.json',parent_results_sha256=sha((parent/'results.json').read_bytes()),
 parent_vectors_path='${ARTIFACT_ROOT}/codex_dt_conditional_boundaries_20260908_v1/vectors.npz',parent_vectors_sha256=sha((parent/'vectors.npz').read_bytes()),
 method_change='None. Optional callbacks expose existing diagnostics=True tensors at layer0 only. Native model FA/FLA and finite mathematics unchanged. Scoring capture passively accepts actual fresh empty cache, never changing forward/config/frame locals.',
 decomposition='16 signed contraction differences: input/post RMS,MLP,residuals,projections,conv+SiLU,QK L2,beta/raw-g maps,FLA state including raw-g exp,mo cast,fused norm-gate. B2 and B1allEOS controls; no theoretical native forward.',
 budget={'model_loads':1,'native_eager_B1_diagnostics':1,'complete_DT_attributions':2,'native_decoder_replays':64,'finite_decoder_calls':64,
   'public_FA_auxiliary_calls':16,'original_metric_curves':0,'native_B1_scoring_forwards':8,'extra_diagnostic_conv_preactivation_calls':0,
   'FT_traces':0,'generation_calls':0,'new_samples':0,'wall_time_seconds':600},
 acceptance='Keep previous fixed2% vector sanity guard,save before any failure. Score precisely existing coarse steps0,1,10,20 and assert all eight model input hashes. Source,weight,actual-capture and contraction closure checks. No gate/seed/FLA candidate is tested or promoted.',
 stop='One2case focused diagnostic only; preserve first failure and partial artifacts, no blind retry. Use completed localization to rank root-cause repair hypotheses; no automatic gate sweep.',
 costs='CPU actual coefficients/paired features retained only for layer0; coefficients.pt saved privately,hash/bytes recorded. No global sequence-square attribution tensors. All diagnostic copy,contractions,and eight real scorer calls counted. No production speed claim.',
 files_sha256={k:sha(v) for k,v in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();dst=A/'dt_layer0_conditional_protocol_20260908.json';assert not dst.exists();dst.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_layer0_conditional_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_layer0_conditional_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'budget':p['budget']}))
