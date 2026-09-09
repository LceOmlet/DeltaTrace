"""Freeze six clean-payload native routing contrasts using existing actual operands."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((A/'dt_FA19_joint_path_scout_20260909_protocol.json').read_bytes());sources=old['sources']
for src in sources:
    if src['case']=='MH3':remote='${ARTIFACT_ROOT}/codex_dt_MH3_FA19_combined_20260909_v1/vectors.npz'
    else:remote='${ARTIFACT_ROOT}/codex_dt_'+src['case']+'_FA19_native_hybrid_20260909_v1/signed_token_contrasts.npz'
    local=A/'snapshot'/remote.lstrip('/');assert local.is_file();src.update(reference_vectors=remote,reference_vectors_sha256=sha(local))
study=R/'research/reproduction_templates/dt_FA19_retained_reference_20260909.py';ast.parse(study.read_bytes())
p={'sources':sources,'FA_interface_sha256':old['FA_interface_sha256'],'native_FA_kwargs':old['native_FA_kwargs'],
    'budget':{'cases':3,'steps':[3,10],'native_FA':6,'native_backward':0,'model':0,'DT':0,'scorer':0,'FT':0,'generation':0,'compile':0,'wall_seconds':180},
    'reason':'The same route change is negative under actual retained contents. Test whether replacing EOS by clean V_C is actually closer, separating reference error from already-known route-slope error. Compare only same output-query coordinates, never choose a per-case reference using these masks.',
    'decision':'Describe both fixed references and intrinsic route/content interaction. No interpolation-weight fitting, candidate promotion or whole-metric claim; record failures rather than assume clean V is more faithful.',
    'files_sha256':{'study.py':sha(study)}}
files={'study.py':study.read_bytes(),'protocol.json':json.dumps(p,indent=2).encode()};stem='dt_FA19_retained_reference_20260909'
pp=A/(stem+'_protocol.json');lp=A/('launch_'+stem+'.json');assert not pp.exists() and not lp.exists()
remote='${ARTIFACT_ROOT}/codex_'+stem+'_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':hashlib.sha256(files['protocol.json']).hexdigest(),'budget':p['budget']}))
