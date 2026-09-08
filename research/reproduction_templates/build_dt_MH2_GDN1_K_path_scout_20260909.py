"""Freeze a single analytic-path normalization candidate, using only B2 endpoints."""
from pathlib import Path
import json,hashlib,ast,base64,zlib,shlex
A=Path('audit');R=Path('DeltaTrace');D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_GDN1_QK_norm_geometry_20260909_v1';sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((D/'results.json').read_bytes());s=json.loads((A/'dt_MH2_GDN1_QK_norm_geometry_summary_20260909.json').read_bytes())
assert s['results_sha256']==sha((D/'results.json').read_bytes()) and s['status']=='MH2_QK_norm_geometry_independent_CPU_audit_passed'
stem='dt_MH2_GDN1_K_path_scout_20260909';remote='${ARTIFACT_ROOT}/codex_'+stem+'_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
files={'study.py':(R/'research/reproduction_templates'/(stem+'.py')).read_bytes(),'integrated_l2_path_20260909.py':(R/'research/reproduction_templates/integrated_l2_path_20260909.py').read_bytes()}
for raw in files.values():ast.parse(raw)
p={'scope':'One fixed analytic path-average L2 Jacobian for GDN1 K; saved actual partial native normalized outputs remain reference. No mask-dependent coefficients, weights, model or native kernel changes.',
 'source':r['protocol']['source'],'geometry_result':'/tmp/'+D.name+'/results.json','geometry_result_sha256':sha((D/'results.json').read_bytes()),
 'steps':['3','10','20','B2'],'quadrature_rows':32,'quadrature_nodes':128,
 'formula':'Integral_0^1 [I/r(t)-x(t)x(t)^T/r(t)^3] dt, x(t)=x0+t(x1-x0), r=sqrt(sum(x*x)+1e-6). Closed-form rank-two plus identity action. Same fixed EOS/input endpoints only; native defaults unchanged.',
 'rationale':'Observed inverse-radius and radial-feedback conditional mismatch. Current rule uses one endpoint-mean radial direction; analytic path average accounts for the direction and radius throughout the fixed chord, preserving the real endpoint identity. No guarantee on other partial directions, so require actual local and original whole metrics.',
 'budget':{'wall_time_seconds':120,'model':0,'GPU':0,'DT':0,'FA':0,'FLA':0,'scorer':0,'FT':0,'generation':0},
 'decision':'Reject if actual early/mid token-head errors materially worsen; local improvement alone never promotes production. Preserve all endpoint/partial, prompt/response and FP32/64 checks. No automatic other candidates or full benchmark.',
 'files_sha256':{name:sha(raw) for name,raw in files.items()}}
files['protocol.json']=json.dumps(p,indent=2).encode();pp=A/(stem.replace('_20260909','_protocol_20260909')+'.json');lp=A/('launch_'+stem+'.json');assert not pp.exists() and not lp.exists()
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader='import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))'
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -B -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':sha(files['study.py']),'helper_sha256':sha(files['integrated_l2_path_20260909.py'])}))
