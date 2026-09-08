import base64,json,hashlib,shlex
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_order_norm_localization_20260908_v1'
sha=lambda b:hashlib.sha256(b).hexdigest();name='normgate_geometry_followup_20260908.py';raw=(R/'research/reproduction_templates'/name).read_bytes()
assert sha(raw)=='648cfd06ed049a044f294d7105bb8eb7313c0350747f944b51696710ba370004'
p={'kind':'adaptive_read_only_CPU_followup_after_saved_NI_result','script_sha256':sha(raw),
 'source_results_sha256':sha((D/'results.json').read_bytes()),'fixed_steps':[1,10,20],
 'scope':'Three exact algebra decompositions: J1 conditional RMS curvature plus parallel/perpendicular finite-operator mismatch; endpoint-conserving closest-J1 rank-one RMS map; full symmetric norm-gate factor change. No coefficient tuning, native/model/GPU invocation or new metric curves.',
 'budget':{'model':0,'GPU':0,'new_data':0,'original_scorer':0,'parameter_search':0,'CPU_seconds':180},
 'reason':'User asks explicit actual-versus-predicted mismatch together with root cause. Saved NI RMS error has low EOS radii but mostly near-chord conditions; do not infer a repair from geometry correlation.',
 'limits':'CPU theoretical attribution algebra only; original experiment remains failed on MH guard. No promotion or complete candidate metrics.',
 'artifact_sha256':json.loads((D/'results.json').read_bytes())['cases']['niah_mq_q2_1']['private_normgate_input_artifact']['sha256']}
pr=json.dumps(p,indent=2).encode();f=A/'normgate_geometry_protocol_20260908.json';assert not f.exists();f.write_bytes(pr)
root='${ARTIFACT_ROOT}/codex_dt_order_norm_localization_20260908_v1';py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
code='import pathlib,json,base64,subprocess;d=pathlib.Path('+repr(root)+');s=d/'+repr(name)+';p=d/"normgate_geometry_protocol.json";assert not s.exists() and not p.exists();s.write_bytes(base64.b64decode('+repr(base64.b64encode(raw).decode())+'));p.write_bytes(base64.b64decode('+repr(base64.b64encode(pr).decode())+'));f=(d/"normgate_geometry.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(s),str(d)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(d/"normgate_geometry.pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"scope":"CPU only"}))'
(A/'launch_normgate_geometry_followup_20260908.json').write_text(json.dumps({'cmd':py+' -c '+shlex.quote(code),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(pr),'script_sha256':sha(raw)}))
