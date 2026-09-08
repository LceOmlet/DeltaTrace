"""Freeze one CPU-only audit of saved actual normalization states."""
from pathlib import Path
import json,hashlib,ast,base64,zlib,shlex
A=Path('audit');R=Path('DeltaTrace');D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_two_decoder_internal_20260909_v1'
sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_bytes())
r=read(D/'results.json');s=read(A/'dt_MH2_two_decoder_internal_summary_20260909.json')
assert s['results_sha256']==sha((D/'results.json').read_bytes()) and s['status']=='MH2_two_decoder_internal_independent_CPU_audit_passed'
stem='dt_MH2_GDN1_QK_norm_geometry_20260909';remote='${ARTIFACT_ROOT}/codex_'+stem+'_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
files={'study.py':(R/'research/reproduction_templates'/ (stem+'.py')).read_bytes(),'signed_secant_rules.py':(R/'core/signed_secant_rules.py').read_bytes()}
for name,raw in files.items():ast.parse(raw)
native='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/lib/python3.12/site-packages/fla/modules/l2norm.py'
scalar='${ARTIFACT_ROOT}/codex_dt_MH2_GDN1_scalar_composition_20260909_v1/results.json'
p={'scope':'Actual MH2 saved native Q/K norm boundary: scalar inverse-radius vs radial-feedback mismatch and native precision, not a new forward, candidate or benchmark.',
 'source':{'result':'/tmp/'+D.name+'/results.json','result_sha256':sha((D/'results.json').read_bytes()),'private':'/tmp/'+D.name+'/'+r['private_artifact']['file'],'private_sha256':r['private_artifact']['sha256']},
 'scalar_result':scalar,'scalar_result_sha256':sha((A/'snapshot'/scalar.lstrip('/')).read_bytes()),
 'native_source_receipts':[{'path':native,'sha256':sha((A/'snapshot'/native.lstrip('/')).read_bytes())}],
 'steps':['3','10','20','B2'],'budget':{'wall_time_seconds':120,'model':0,'GPU':0,'DT':0,'FA':0,'FLA':0,'scorer':0,'FT':0,'generation':0},
 'interpretation':'Analytic CPU64 normalization is an intermediate algebraic coordinate; captured native normalized outputs remain the reference. All contractions compact repeated heads at the same token. Conditional C/A radii are used only to explain errors, not to define a deployable candidate from evaluation masks.',
 'files_sha256':{name:sha(raw) for name,raw in files.items()}}
files['protocol.json']=json.dumps(p,indent=2).encode();pp=A/(stem.replace('_20260909','_protocol_20260909')+'.json');lp=A/('launch_'+stem+'.json')
assert not pp.exists() and not lp.exists()
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader='import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))'
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -B -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':sha(files['study.py']),'launcher':str(lp)}))
