"""A separate bounded recovery using v2's saved states, never rerun the root."""
import ast,base64,hashlib,json,shlex,zipfile,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_gdn_integration_20260908_v2'
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        path=(D/name).resolve();assert path.is_relative_to(D.resolve())
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
r=json.loads((D/'results.json').read_bytes());assert r['status']=='failed' and r['finite_completed']==2 and r['native_GDN_backward_attempts']==1
assert "counts['FLA']==counts['conv']==counts['conv_backward']==1" in r['error']
p=json.loads((D/'protocol.json').read_bytes())
p.update(parent_directory='${ARTIFACT_ROOT}/codex_qwen35_gdn_integration_20260908_v2',parent_sha256=sha((D/'results.json').read_bytes()),
    endpoints_sha256=next(x['sha256'] for x in r['artifacts'] if x['file']=='native_first_GDN_endpoints.pt'),
    reuse_inductor_cache='${ARTIFACT_ROOT}/codex_qwen35_gdn_integration_20260908_v2/inductor_cache',
    maximum_model_loads=0,maximum_root_forward_attempts=0,maximum_finite_GDN_calls=2,
    maximum_native_module_loads=1,maximum_native_GDN_forward_calls=1,maximum_native_GDN_backward_calls=1,
    maximum_auxiliary_paired_conv_forward_calls=2,maximum_auxiliary_paired_conv_backward_calls=2,
    purpose='Recover full input vectors and equal-endpoint gradient limit after a main-thread profiler count assertion. Use saved native states and the exact original GDN class with nine verified checkpoint tensors; no new model/root pass.',
    observation_correction='Use an actual CausalConv1dFnBackward graph-node hook; record callback/main thread IDs. Do not infer missing backward from a missing main-thread profile event.',
    combined_family_budget='Including preserved v1/v2: two full model loads/root attempts, one completed root, one standalone native GDN module load, at most four finite GDN calls and two local ordinary GDN backward calls. Explicitly one extra finite and native call beyond the original screen; no quality calls.',
    stop='One saved-state recovery only. Persist vectors immediately after each completed stage. Stop on failure; do not reload the full model or repeat root forward.')
files={'study.py':(A/'qwen35_gdn_saved_recovery_20260908.py').read_bytes(),
    'qwen35_gdn_finite.py':(R/'research/runtime/qwen35_gdn_finite.py').read_bytes(),
    'finite_fla_gpu.py':(R/'research/runtime/finite_fla_gpu.py').read_bytes(),
    'signed_secant_rules.py':(R/'core/signed_secant_rules.py').read_bytes()}
for raw in files.values():ast.parse(raw)
p['files_sha256']={k:sha(v) for k,v in files.items()}
files['protocol.json']=json.dumps(p,indent=2).encode();(A/'qwen35_gdn_saved_recovery_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_gdn_saved_recovery_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_gdn_saved_recovery_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'parent_sha256':p['parent_sha256'],'source_sha256':p['files_sha256']}))
