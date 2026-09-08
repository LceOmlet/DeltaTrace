"""One reviewed recovery; preserve v1 failure and original 2% control ceiling."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
old=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v1'
p=json.loads((old/'protocol.json').read_bytes());r=json.loads((old/'results.json').read_bytes())
assert r['status']=='failed' and r['DT_calls']==1 and r['FT_calls']==0 and r['scoring_forwards_completed']==0
files={name:(old/name).read_bytes() for name in p['files_sha256']}
files['study.py']=(R/'research/reproduction_templates/dt_MAS_three_cases_recovery_20260908.py').read_bytes()
files['qwen35_dense_finite_runner.py']=(R/'research/runtime/qwen35_dense_finite_runner.py').read_bytes()
for raw in files.values():ast.parse(raw)
for name,want in p['unchanged_finite_runtime_sha256'].items():assert sha(files[name])==want
p['recovery']={'parent_protocol_sha256':sha((old/'protocol.json').read_bytes()),'parent_results_sha256':sha((old/'results.json').read_bytes()),
 'prior_status':'failed_original_control_guard;not_reclassified',
 'reason':'v1 changed native initialization from load-eager/B1 diagnostic/set-FA to load-FA. Its NI0 vector differed3.975%,although every actual replay andFA auxiliary matched its same-run root. Restore the original native execution order,not finite formulas. Also distinguish standalone actual-root FP32 logprob from compiled-seed logprob in bookkeeping.',
 'changes':['Original native eager model loading plus one original B1 diagnostic before public switch toFA,matching the successful frozen NI0 execution order.',
            'RootG diagnostic uses actual-logit standalone FP32 logsoftmax exactly as the original driver. Separately report the compiled-seed effect;finite seed and all finite formulas unchanged.'],
 'unchanged_guards':'Preserve2% vector control ceiling and2.5 percentage-point needle drift; do not relax them after failure.',
 'no_claim':'No claim yet that execution order caused the drift or that default precision is incorrect. One bounded recovery can test whether restoration is sufficient.'}
p['budget']['native_eager_B1_control_diagnostics']=1
p['files_sha256']={name:sha(raw) for name,raw in files.items()}
files['protocol.json']=json.dumps(p,indent=2).encode()
dst=A/'dt_MAS_three_cases_recovery_protocol_20260908.json';assert not dst.exists();dst.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v2'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_MAS_three_cases_recovery_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'budget':p['budget']}))
