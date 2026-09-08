"""One frozen original-input tangent rule, only for the measured K norm boundary."""
from pathlib import Path
import json,hashlib,ast,base64,zlib,shlex
A=Path('audit');R=Path('DeltaTrace');sha=lambda b:hashlib.sha256(b).hexdigest()
old=json.loads((A/'dt_MH2_GDN1_K_path_scout_protocol_20260909.json').read_bytes());audit=json.loads((A/'dt_MH2_GDN1_K_path_scout_summary_20260909.json').read_bytes())
assert audit['status']=='MH2_K_path_scout_independent_CPU_audit_passed_rejected'
stem='dt_MH2_GDN1_K_input_supported_20260909';remote='${ARTIFACT_ROOT}/codex_'+stem+'_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
files={'study.py':(R/'research/reproduction_templates'/(stem+'.py')).read_bytes(),'input_supported_l2_20260909.py':(R/'research/reproduction_templates/input_supported_l2_20260909.py').read_bytes()}
for raw in files.values():ast.parse(raw)
p={k:old[k] for k in ['source','geometry_result','geometry_result_sha256','steps','budget']}
p.update(scope='One CPU-only input-supported finite L2 candidate at actual saved GDN1 K; no model, FA/FLA or real native backward is executed.',
 formula='J=J1+(delta_n-J1 delta_x) delta_x^T/||delta_x||^2; zero endpoint delta uses J1. Real normalization formula defines J1, native captured normalized outputs remain actual references.',
 rationale='Actual partial raw K has large off-chord displacement (error-weighted .407/.519 of input radius); exact fixed-path average worsened both partial errors. Preserve the same endpoint chord and use original-input tangent response off that chord, where endpoint conservation does not determine a response. Mean relative radius distance to input .182/.196 versus EOS .280/.278; not all rows closer input (weighted fraction .510/.474), so this is a testable local hypothesis, not a proven superiority.',
 prior_failures='The same supported principle failed earlier FA softmax and composite decay scouts. Those failures are retained; operator-specific K geometry does not establish success for this distinct local application or whole method.',
 decision='One discrete, parameter-free operator. Inspect both fixed actual early/mid signed and coordinate-consistent absolute errors before any complete propagation. No full metric call unless evidence justifies it, no continuous weights or score calibration.',
 files_sha256={name:sha(raw) for name,raw in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();pp=A/(stem.replace('_20260909','_protocol_20260909')+'.json');lp=A/('launch_'+stem+'.json');assert not pp.exists() and not lp.exists()
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader='import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))'
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -B -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':sha(files['study.py']),'helper_sha256':sha(files['input_supported_l2_20260909.py'])}))
