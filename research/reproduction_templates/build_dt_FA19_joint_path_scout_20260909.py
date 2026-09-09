"""Predeclare native joint-operator quadrature on MH0/MH2/MH3 actual saved states."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
sources=[]
for name in ['MH0','MH2']:
    donor=A/'snapshot/tmp'/('codex_dt_'+name+'_FA19_native_hybrid_20260909_v1')
    dp,dr=read(donor/'protocol.json'),read(donor/'results.json')
    assert read(donor/'terminal_receipt.json')['proc_exists'] is False
    src={'case':name,'private_path':dp['private_artifact_path'],'private_sha256':dp['private_artifact_sha256'],
        'results_path':dp['source_results_path'],'results_sha256':dp['source_results_sha256'],
        'frozen_core_errors':{s:dr['points'][s]['fields']['core_error_at_stored_seed']['net'] for s in ['3','10','20']},
        'hybrid_results_path':'/tmp/'+donor.name+'/results.json','hybrid_results_sha256':sha(donor/'results.json')}
    sources.append(src)
S=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH3_FA19_combined_20260909_v1';sp,sr=read(S/'protocol.json'),read(S/'results.json')
assert read(S/'terminal_receipt.json')['proc_exists'] is False and sr['status']=='MH3_FA19_combined_1DT4score11FA_complete'
private=sr['cases']['morehopqa_3']['private_artifact']
sources.append({'case':'MH3','private_path':'/tmp/'+S.name+'/'+private['file'],'private_sha256':private['sha256'],
    'results_path':'/tmp/'+S.name+'/results.json','results_sha256':sha(S/'results.json'),
    'frozen_core_errors':{s:sr['FA19_native_probe']['points'][s]['fields']['core_error']['net'] for s in ['3','10','20']}})
study=R/'research/reproduction_templates/dt_FA19_joint_path_scout_20260909.py';ast.parse(study.read_bytes())
p={'scope':'Frozen local full-attention joint-QKV path scout on three real author-cache cases; both methods defined before calls, no masks in coefficients.',
    'sources':sources,'nodes':[.5,.5-3**.5/6,.5+3**.5/6],'methods':{'joint_midpoint':{'nodes':[0],'weights':[1]},'joint_gauss2':{'nodes':[1,2],'weights':[.5,.5]}},
    'FA_interface_sha256':sp['installed_FA_interface_sha256'],'native_FA_kwargs':dp['native_FA_kwargs'],
    'reason':'Native derivative of the complete FA operator at a joint interpolated QKV state retains the correlation between routing and payload within each quadrature node. Current separate endpoint local rules expose large reference-content and routing errors on all three actual cases. This is a distinct approximate operator-path hypothesis, not a guarantee.',
    'budget':{'cases':3,'midpoint_native_FA':3,'gauss2_native_FA':6,'input_endpoint_replay_FA':3,'native_FA_total':12,'native_autograd':9,'model':0,'DT':0,'scorer':0,'FT':0,'generation':0,'compilation':0,'wall_seconds':240},
    'decision':'Report all early/mid/full-endpoint signed errors and native replay drift for both fixed approximations. Whole propagation requires a meaningful cross-case local gain and tolerable endpoint error; do not expand quadrature nodes or compensate final scores merely to make the test pass. Reject or defer if cheap approximations are unreliable. No quality, complexity-speed or memory-acceptance conclusion from local calls.',
    'files_sha256':{'study.py':sha(study)}}
files={'study.py':study.read_bytes(),'protocol.json':json.dumps(p,indent=2).encode()};stem='dt_FA19_joint_path_scout_20260909'
pp=A/(stem+'_protocol.json');lp=A/('launch_'+stem+'.json');assert not pp.exists() and not lp.exists()
remote='${ARTIFACT_ROOT}/codex_'+stem+'_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':hashlib.sha256(files['protocol.json']).hexdigest(),'budget':p['budget']}))
