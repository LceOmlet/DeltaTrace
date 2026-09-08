"""Frozen same-rule MH1 check after NI1 original metrics improved."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_normgate_repair_NI1_20260908_v1';p=json.loads((D/'protocol.json').read_bytes());ni=json.loads((D/'results.json').read_bytes())
assert ni['status']=='NI1_paired_normgate_repair_original_metrics_complete'
c=ni['cases']['niah_mq_q2_1'];cur,can=c['curves']['current'],c['curves']['candidate']
assert can['return_metrics'][1]<cur['return_metrics'][1] and can['return_metrics'][0]<=cur['return_metrics'][0] and can['needle']>=cur['needle']
assert abs(c['scoring_points']['1']['complete_input']['candidate']['prediction_minus_actual'])<abs(c['scoring_points']['1']['complete_input']['current']['prediction_minus_actual'])
assert abs(c['scoring_points']['10']['complete_input']['candidate']['prediction_minus_actual'])<abs(c['scoring_points']['10']['complete_input']['current']['prediction_minus_actual'])
files={n:(D/n).read_bytes() for n in p['files_sha256']}
# Only dataset selection/status scope changes. Every computational source remains frozen.
s=files['study.py'].decode().replace("assert p['case_indices']==[['niah_mq_q2',1]]","assert p['case_indices']==[['morehopqa',1]]")
s=s.replace('Paired layer0 norm-gate repair on one original NI1 example.','Same frozen paired layer0 norm-gate repair on original MH1.')
s=s.replace('NI1 development comparison only.','MH1 existing-case confirmation only, not an independent holdout.')
s=s.replace('No MH confirmation or production speed claim;','No independent confirmation or production speed claim;')
s=s.replace("r['status']='NI1_paired_normgate_repair_original_metrics_complete'","r['status']='MH1_paired_normgate_repair_original_metrics_complete'")
files['study.py']=s.encode()
for raw in files.values():ast.parse(raw)
assert all(sha(files[n])==want for n,want in p['files_sha256'].items() if n!='study.py')
p.update(scope='One existing MoreHopQA1 check of EXACT NI1 symmetric-layer0 candidate after joint NI1 original MAS,RISE,needle and fixed-effect improvement. No rule tuning, no independent holdout claim.',
 cases=['morehopqa_1'],case_indices=[['morehopqa',1]],
 decision_source_NI1_results_sha256=sha((D/'results.json').read_bytes()),
 decision_source_NI1_metrics={'current':cur['return_metrics'],'candidate':can['return_metrics'],'needle':[cur['needle'],can['needle']]},
 stop='One existing MH1 only; no expanded sample set or automatic rule sweep. Preserve failure; no old-vector threshold retry or framework change. Even joint quality success needs complete production cost/integration validation.',
 files_sha256={k:sha(v) for k,v in files.items()})
p['budget']['subsequent_CPU_beta_audit_seconds']=0 # NI-specific CPU diagnosis; do not claim a skipped audit as MH evidence.
files['protocol.json']=json.dumps(p,indent=2).encode();dst=A/'dt_normgate_repair_MH1_protocol_20260908.json';assert not dst.exists();dst.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_normgate_repair_MH1_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_normgate_repair_MH1_20260908.json').write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'computational_sources_identical_to_NI1':True,'budget':p['budget']}))
