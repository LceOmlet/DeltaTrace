"""Freeze the existing eleven-call native FA hybrid diagnostic on actual MH0 operands."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
S=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_internal_20260909_v1';sp=json.loads((S/'protocol.json').read_bytes());sr=json.loads((S/'results.json').read_bytes())
assert sr['status']=='MH0_current_FA19_1native_kwargs5replay1finite_internal_complete' and sr['protocol']==sp
receipt=json.loads((S/'terminal_receipt.json').read_bytes());assert receipt.get('pid_alive',receipt.get('proc_exists',False)) is False
for name,item in receipt['files'].items():assert sha((S/name).read_bytes())==(item if isinstance(item,str) else item['sha256'])
assert sr['sources_before']==sr['sources_after'] and sr['weight_stats_before']==sr['weight_stats_after']
summary=A/'dt_MH0_FA19_internal_summary_20260909.json';su=json.loads(summary.read_bytes());assert su['status']=='MH0_FA19_internal_independent_CPU_audit_passed'
assert su['results_sha256']==sha((S/'results.json').read_bytes())
B=A/'snapshot'/sp['source_boundary']['results_path'].lstrip('/');br=json.loads(B.read_bytes());assert sha(B.read_bytes())==sp['source_boundary']['results_sha256']
assert br['cases']['morehopqa_0']['input']==sr['input']
V=A/'snapshot${ARTIFACT_ROOT}/codex_dt_fa19_native_hybrid_20260908_v2';vp=json.loads((V/'protocol.json').read_bytes());vr=json.loads((V/'results.json').read_bytes())
assert vr['status']=='eleven_native_FA_hybrid_contrasts_complete' and vr['native_FA_calls_returned']==11
helper='decoder19_conditional_decomposition_20260908.py';hraw=(R/'research/reproduction_templates'/helper).read_bytes()
assert hraw==(S/helper).read_bytes()==(V/helper).read_bytes() and sha(hraw)==sp['files_sha256'][helper]
sraw=(R/'research/reproduction_templates/dt_MH0_FA19_native_hybrid_20260909.py').read_bytes()
oldraw=(R/'research/reproduction_templates/dt_fa19_native_hybrid_20260908_v2.py').read_bytes();assert oldraw==(V/'study.py').read_bytes()
oldtree,newtree=ast.parse(oldraw),ast.parse(sraw)
for name in ['signed_stats','project','tensor_layout','native_call']:
    get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(get(oldtree))==ast.dump(get(newtree)),('Changed native computation',name)
# Main mathematical decomposition is retained byte-for-byte from successful V2.
a="        delta_q = (actual_points['0']['query'].double()";b="        save()\n    assert r['native_FA_calls_entered']"
assert oldraw.decode().replace('\r\n','\n').split(a)[1].split(b)[0]==sraw.decode().replace('\r\n','\n').split(a)[1].split(b)[0]
files={'study.py':sraw,helper:hraw};artifact=sr['private_artifact']
assert artifact['sha256']=='1637779cd62c6ffd05641600fc83eda87d9488ff99db1f78f4baaae0d7a93e0d' and artifact['bytes']==1272488075
schedule=['replay_0','replay_3','replay_10','replay_20','X_C0']
for step in ['3','10','20']:schedule.extend(['X_A0_'+step,'X_CA_'+step])
assert len(schedule)==11
actual=sr['points']['0']['native_dense_arguments'];assert all(v['native_dense_arguments']==actual for v in sr['points'].values())
kwargs=copy.deepcopy(vp['native_FA_kwargs']);kwargs.update(actual)
assert kwargs==vp['native_FA_kwargs']
p=copy.deepcopy(vp);p.pop('v1_failure_provenance',None)
remote='${ARTIFACT_ROOT}/codex_dt_MH0_FA19_native_hybrid_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p.update({
    'scope':'Fixed current MH0 decoder19 native FA hybrid contrasts on actual saved replayed coordinates and coefficients; source transfer to original boundary was audited. No model, DT, scorer, FT, backward, probability matrix or new candidate.',
    'version':'MH0_source_adaptation_of_successful_v2',
    'successful_template_provenance':{'files':[{'remote_path':'/tmp/'+V.name+'/'+name,'sha256':sha((V/name).read_bytes())} for name in ['study.py','protocol.json','results.json']],
        'native_calls':11,'job_seconds':vr['job_seconds'],'computation_identity':'Native call/project/layout/signed-stats AST and main qk/value/three-term/seed/group formulas byte-identical to successfulV2. Only source-schema/identity adapter and early step3 replace MH1 step1. Standard transpose.to(cuda).contiguous retained after H2D.'},
    'source_results_path':'/tmp/'+S.name+'/results.json','source_results_sha256':sha((S/'results.json').read_bytes()),
    'source_protocol_path':'/tmp/'+S.name+'/protocol.json','source_protocol_sha256':sha((S/'protocol.json').read_bytes()),
    'source_boundary_results_path':sp['source_boundary']['results_path'],'source_boundary_results_sha256':sp['source_boundary']['results_sha256'],
    'private_artifact_path':'/tmp/'+S.name+'/'+artifact['file'],'private_artifact_sha256':artifact['sha256'],'private_artifact_bytes':artifact['bytes'],
    'source_internal_audit':{'file':summary.name,'sha256':sha(summary.read_bytes()),'scope':'All nine terms+three transfer token closures passed; regenerated m19 and native outputs match saved current boundary, all transfer terms zero. This does not pre-establish the hybrid decomposition result.'},
    'input_sha256':sr['input']['input_sha256'],'case':'morehopqa_0','decoder_index':19,'fixed_steps':['3','10','20'],
    'frozen_input_receipts':{step:br['cases']['morehopqa_0']['points'][step]['input_receipt'] for step in ['0','3','10','20']},
    'original_core_errors_including_seed_cast':{step:sr['conditional_ledgers'][step]['replayed_9term_ledger']['terms']['finite_FA_core_including_seed_cast'] for step in ['3','10','20']},
    'installed_FA_interface_sha256':sp['installed_FA_interface_sha256'],'native_model_sha256':sp['native_model_sha256'],
    'checkpoint_config_path':sp['checkpoint']+'/config.json','checkpoint_config_sha256':sp['checkpoint_config_tokenizer_sha256']['config.json'],
    'scaling_verification':'Direct actual native dense FA argument captured identically in all five current source replays. Retained original native AST/config assertion corroborates actual scale; do not infer a replacement scale.',
    'native_FA_kwargs':kwargs,'native_call_schedule':schedule,
    'budget':{'native_FA_calls':11,'actual_output_replays':4,'hybrid_operator_calls':7,'model_calls':0,'DT_calls':0,'scorer_calls':0,'FT_calls':0,'backward_calls':0,'generation_calls':0,'new_samples':0,'extra_warmup_calls':0,'wall_time_seconds':180},
    'artifact_contract':'Current actual private1.272GB capture stays remote and is hash-verified before/after. No weights/model load or raw activations in public payload/review ZIP; only signed scalar/token contrasts and provenance are exported.',
    'acceptance':'Current source/config/private/artifact/native argument identities and actual GPU layouts hold. Recompute exact saved core before any GPU FA. Existing three-term/seed/group closures below1e-7; exactly11 entered/returned native FA calls. Every replay drift retained, no candidate or bitwise admission threshold.',
    'files_sha256':{name:sha(raw) for name,raw in files.items()}})
for name,data in files.items():ast.parse(data,filename=name)
files['protocol.json']=json.dumps(p,indent=2).encode();pp=A/'dt_MH0_FA19_native_hybrid_protocol_20260909.json';lp=A/'launch_dt_MH0_FA19_native_hybrid_20260909.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen artifacts.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(data).decode() for name,data in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'payload':str(lp),'remote_directory':remote,'budget':p['budget']}))
