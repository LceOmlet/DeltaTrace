"""Freeze C/candidate paired propagation on already captured actual MH3 states; zero scoring."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_GDN1_K_remaining_20260909_s2_v1'
p=copy.deepcopy(json.loads((D/'protocol.json').read_bytes()))
s=(R/'research/reproduction_templates/dt_GDN1_K_remaining_20260909.py').read_text()
start=s.index('    remaining_pairs=');end=s.index('    assert sha(Path(native.__file__)',start)
s=s[:start]+'''    assert p['call_schedule']==[['morehopqa_3','control','diagnostic'],['morehopqa_3','candidate','diagnostic']]
    assert p['quality_schedule']==[]
    r['completion_scope']='Two attributions only; conditional propagation diagnosis on saved actual native states, no metric curve.'
'''+s[end:]
start=s.index('    # Original scorer forwards,');end=s.index("    r['finite_counts']=",start)
s=s[:start]+'''    assert r['scorer_entered']==r['scorer_returned']==0
    r['conditional_propagation'],more=compare(observers,vectors,p)
    vectors.update(more);np.savez_compressed(A/'vectors.npz',**vectors)
'''+s[end:]
s=s.replace("observer=None)","observer=observers.setdefault(method,Observer()))")
s=s.replace("r['DT_entered']<4","r['DT_entered']<2").replace("r['DT_returned']==4","r['DT_returned']==2")
s=s.replace('BudgetedFinite(original_fa,16)','BudgetedFinite(original_fa,8)').replace('reuse_scalar_products=False),100)','reuse_scalar_products=False),50)')
s=s.replace('len(key_calls)<2','len(key_calls)<1').replace('len(key_calls)==2','len(key_calls)==1')
s=s.replace('returned==(16)','returned==(8)').replace('finite_fla.returned==100','finite_fla.returned==50').replace('candidate_backend.returned==4','candidate_backend.returned==2')
s=s.replace("r['status']='GDN1_K_input_supported_4DT100FLA84score_complete'","r['status']='MH3_K_lower_propagation_2DT0score_complete'")
s=s.replace('    import causal_conv1d','    import causal_conv1d\n    from mh3_k_lower_propagation_20260909 import Observer,compare\n    observers={}')
a=s.index("    r['cost_scope']=");b=s.index('\n',a)
s=s[:a]+"    r['cost_scope']='One model load and original NI0 eager initialization,2DT/0score,16finiteFA/48phases,50finiteFLA/100nativeadjoints,64native decoder replays. Passive coefficients only at boundaries0/1/2 and GDN1 K. Actual partial inputs/states reused from prior1DT4score diagnostic; cross-process native/source drift reported separately. No new candidate,FT,generation,full metric curve or performance acceptance.'"+s[b:]
study=R/'research/reproduction_templates/dt_MH3_K_lower_propagation_20260909.py';assert not study.exists();study.write_text(s,encoding='utf-8')
files={'study.py':study.read_bytes()}
for name,want in p['files_sha256'].items():
    if name=='study.py':continue
    raw=(D/name).read_bytes();assert sha(raw)==want;files[name]=raw
name='mh3_k_lower_propagation_20260909.py';files[name]=(R/'research/reproduction_templates'/name).read_bytes()
S=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH3_native_K_boundaries_20260909_v1'
sr=json.loads((S/'results.json').read_bytes());assert sr['status']=='MH3_native_K_boundaries_1DT4score_complete'
assert json.loads((S/'terminal_receipt.json').read_bytes())['proc_exists'] is False
private=sr['cases']['morehopqa_3']['private_artifact']
p.update(case_indices=[['niah_mq_q2',0],['morehopqa',3]],quality_cases=[],quality_schedule=[],
    call_schedule=[['morehopqa_3','control','diagnostic'],['morehopqa_3','candidate','diagnostic']])
p['native_state_source']={'private_path':'/tmp/'+S.name+'/'+private['file'],'private_sha256':private['sha256'],
    'results_path':'/tmp/'+S.name+'/results.json','results_sha256':sha((S/'results.json').read_bytes())}
p['budget']={'model_loads':1,'native_eager_initializations':1,'DT':2,'original_scores':0,'finite_FA':16,'finite_FLA':50,'native_FLA_adjoints':100,'native_decoder_replays':64,'wall_time_seconds':600}
p['scope']='Distinguish actual K norm candidate change from lower GDN1 and GDN0 propagation changes under the same upstream, fixed original MH3 masks3/10/20. No new quality curve.'
p['decision']='Keep C/candidate upstream equality checks and exact signed telescoping. A locally better norm may be worse after approximating the lower mixed-input state; locate this rather than modifying another arbitrary rule.'
p['protected_sources'] += [{'path':'/tmp/'+S.name+'/results.json','sha256':sha((S/'results.json').read_bytes())}]
p['files_sha256']={n:sha(raw) for n,raw in files.items()}
for n,raw in files.items():ast.parse(raw,filename=n)
files['protocol.json']=json.dumps(p,indent=2).encode();stem='dt_MH3_K_lower_propagation_20260909'
pp=A/(stem+'_protocol.json');lp=A/('launch_'+stem+'.json');assert not pp.exists() and not lp.exists()
remote='${ARTIFACT_ROOT}/codex_'+stem+'_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'budget':p['budget']}))
