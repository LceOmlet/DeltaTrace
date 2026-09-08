import ast,json,hashlib,base64,zlib,shlex
from pathlib import Path
A=Path('audit');R=Path('DeltaTrace');H=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_FA19_native_hybrid_20260909_v1';S=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_two_decoder_internal_20260909_v1'
sha=lambda b:hashlib.sha256(b).hexdigest();h=json.loads((H/'results.json').read_bytes());s=json.loads((S/'results.json').read_bytes());audit=json.loads((A/'dt_MH2_FA19_native_hybrid_summary_20260909.json').read_bytes())
assert audit['results_sha256']==sha((H/'results.json').read_bytes()) and audit['status']=='MH2_FA19_native_hybrid_independent_CPU_audit_passed'
assert all(x['bitwise_equal'] for x in h['replays'].values())
raw=(R/'research/reproduction_templates/dt_MH2_FA19_native_vjp_scout_20260909.py').read_bytes();ast.parse(raw)
remote='${ARTIFACT_ROOT}/codex_dt_MH2_FA19_native_vjp_scout_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p={'scope':'Actual native input-endpoint FA VJP scout at FA19. Keep fixed stored finite upstream and all actual Q/K/V/O; diagnose changing only this attention core. No whole model/DT/scorer/FT/new generation or new attention implementation.',
 'source':{'results':'/tmp/'+S.name+'/results.json','private':'/tmp/'+S.name+'/'+s['private_artifact']['file'],'private_sha256':s['private_artifact']['sha256']},
 'boundary_results':s['protocol']['source_boundary']['results_path'],'FA_interface_sha256':h['protocol']['installed_FA_interface_sha256'],
 'native_FA_kwargs':h['protocol']['native_FA_kwargs'],'argument_provenance':h['protocol']['argument_provenance'],
 'protected_sources':h['protocol']['argument_provenance']['sources']+[{'path':'/tmp/'+d.name+'/'+n,'sha256':sha((d/n).read_bytes())} for d,names in [(S,['results.json','protocol.json']),(H,['results.json','protocol.json'])] for n in names]+[{'path':s['protocol']['source_boundary']['results_path'],'sha256':s['protocol']['source_boundary']['results_sha256']}],
 'budget':{'wall_seconds':120,'public_FA_forward':1,'public_native_autograd_backward':1,'whole_model':0,'DT':0,'scorer':0,'FT':0,'generation':0,'new_candidates':1},
 'hypothesis':'Actual input VJP uses original V1 and local softmax Jacobian, removing EOS V0 route-reference and fixed two-endpoint softmax mismatch together. It may underestimate finite/saturated changes; signed endpoint residual must be reported, not forced to zero. Only an actual local scout; a positive result requires full propagation and original quality plus cost.',
 'precision':'Public original BF16 FA with native BF16 upstream cast; compact native GQA K/V gradients retained, no headwise shadow expansion. CPU64 contractions only for diagnostics. No altered native backward or private FA buffer/API call.',
 'files_sha256':{'study.py':sha(raw)}}
f={'study.py':raw,'protocol.json':json.dumps(p,indent=2).encode()};pp=A/'dt_MH2_FA19_native_vjp_scout_protocol_20260909.json';lp=A/'launch_dt_MH2_FA19_native_vjp_scout_20260909.json';assert not pp.exists() and not lp.exists()
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in f.items()}).encode())).decode()
loader='import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))'
pp.write_bytes(f['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}));print(json.dumps({'protocol_sha256':sha(f['protocol.json']),'study_sha256':sha(raw),'budget':p['budget']}))
