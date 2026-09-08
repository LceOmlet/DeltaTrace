from pathlib import Path
import json,hashlib,ast,base64,zlib,shlex
A=Path('audit');R=Path('DeltaTrace');D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_two_decoder_internal_20260909_v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((D/'results.json').read_bytes());s=json.loads((A/'dt_MH2_two_decoder_internal_summary_20260909.json').read_bytes())
assert s['results_sha256']==sha((D/'results.json').read_bytes()) and s['status']=='MH2_two_decoder_internal_independent_CPU_audit_passed'
raw=(R/'research/reproduction_templates/dt_MH2_decay_block_scout_20260909.py').read_bytes();ast.parse(raw)
f={'study.py':raw};remote='${ARTIFACT_ROOT}/codex_dt_MH2_decay_block_scout_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p={'scope':'CPU64 local x-to-decay block secant scout using actual GDN1 native inputs and original Wa/Alog/bias tensors; no model forward or production precision claim.',
 'source':{'result':'/tmp/'+D.name+'/results.json','result_sha256':sha((D/'results.json').read_bytes()),'private':'/tmp/'+D.name+'/'+r['private_artifact']['file'],'private_sha256':r['private_artifact']['sha256']},
 'checkpoint':r['protocol']['checkpoint'],'expected_weight_stats':r['protocol']['expected_weight_stats'],
 'budget':{'wall_time_seconds':120,'model':0,'GPU':0,'DT':0,'FA':0,'FLA':0,'scorer':0,'FT':0,'generation':0},
 'interpretation':'One candidate M=J1+(dy-J1dx)dxT/(dxTdx), grouping original input projection and scalar decay. Min Frobenius change to original-input analytic Jacobian under endpoint constraint; projection transfer retained. Real scalar secants cannot change their slope while preserving their fixed endpoint identity. This CPU64 scout is mathematical local evidence only; alpha is analytic exp of actual native rawg, not a captured native chunk output. No method/default/kernel/FA/FLA changed and no benchmark improvement claimed.',
 'files_sha256':{'study.py':sha(raw)}}
f['protocol.json']=json.dumps(p,indent=2).encode();pp=A/'dt_MH2_decay_block_scout_protocol_20260909.json';lp=A/'launch_dt_MH2_decay_block_scout_20260909.json';assert not pp.exists() and not lp.exists()
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in f.items()}).encode())).decode()
loader='import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))'
pp.write_bytes(f['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}));print(json.dumps({'study_sha256':sha(raw),'protocol_sha256':sha(f['protocol.json'])}))
