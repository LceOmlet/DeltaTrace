"""Freeze one MLP candidate; original finite files and all FT sources stay fixed."""
import ast,base64,hashlib,json,shlex,zlib,time
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
old=A/'snapshot${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1';r=json.loads((old/'results.json').read_bytes());p=json.loads((old/'protocol.json').read_bytes())
assert r['status']=='DT_content_P1_same_official_NI0_input_completed'
files={}
for name,digest in p['files_sha256'].items():
 if name=='study.py':continue
 raw=(old/name).read_bytes();assert sha(raw)==digest;files[name]=raw
files['study.py']=(R/'research/reproduction_templates/dt_mlp_content1_NI0_20260908.py').read_bytes()
files['mlp_content1_finite.py']=(R/'research/runtime/mlp_content1_finite.py').read_bytes()
for raw in files.values():ast.parse(raw)
p.update(scope='Single DT MLP content1 allocation versus unchanged symmetric DT; same authenticated actual588-token NI0 endpoints. FT source/method/controller never changed.',
 endpoint_directory='${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1',
 endpoint_artifacts_sha256={**{name:r['artifacts'][name]['sha256'] for name in ['actual_native_root.pt','actual_packed_target_logits.pt','signed_result.npz']},
  'results.json':sha((old/'results.json').read_bytes()),'protocol.json':sha((old/'protocol.json').read_bytes())},
 expected_weight_stats=r['weight_stats_after'],
 hypothesis='Delta(u*SiLU(g))=SiLU(g1)*Delta(u)+u0*Delta(SiLU(g)); replaces only MLP symmetric interaction allocation. May reduce baseline-gate dilution, may lose gate/routing credit. No guaranteed metric gain.',
 mathematical_cost='Three projection-transpose GEMMs per MLP in each rule, identical to current DT; no extra per-method root or attribution pass required in production.',
 cache_contract='Use only actual original NI0 root+kwargs+full-target logits with frozen hashes, same checkpoint/model/FA/FLA sources. Recompute every real decoder once and compare with saved endpoint outputs. Symmetric control must reproduce saved full signed vector and recovery; no605-token cache.',
 budget={'model_loads':1,'new_root_forwards':0,'original_decoder_replays_B2':32,'finite_decoder_calls_per_rule':32,'finite_decoder_calls_total':64,
  'finite_FA_total':16,'finite_FLA_total':48,'public_FA_auxiliary_LSE':8,'native_conv_auxiliary_forwards':48,'native_conv_auxiliary_backwards':48,
  'shared_logprob_seed':1,'shared_final_norm_seed':1,'original_needle_CPU_calls':2,'original_FT_runs':0,'generation_calls':0,'deletion_queries':0,'new_samples':0,'wall_time_seconds':600},
 stop='One fixed MLP candidate, NI0 only. Preserve first runtime/source/input/nonfinite failure; no automatic retry or FT modification. Compare all signed scores, residuals and original needle before deciding small Morehop/original RISE/MAS follow-up. No parameter scans or gold-guided selection.',
 cost='Shared authenticated native endpoints and one decoder replay per layer for two attribution rules. Report load and paired propagation including cold compilation/IO separately. Do not compare this cached shared experiment time directly with full original FT or prior whole DT time. The prior actual endpoint-generation and authentication costs remain in lineage.',
 baseline_default_unchanged=True,files_sha256={k:sha(v) for k,v in files.items()})
for key in ['added_observation','stop','cost']:
 assert isinstance(p[key],str)
files['protocol.json']=json.dumps(p,indent=2).encode();path=A/'dt_mlp_content1_NI0_protocol_20260908.json';assert not path.exists();path.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_mlp_content1_NI0_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_mlp_content1_NI0_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'candidate_sha256':p['files_sha256']['mlp_content1_finite.py'],'budget':p['budget']}))
