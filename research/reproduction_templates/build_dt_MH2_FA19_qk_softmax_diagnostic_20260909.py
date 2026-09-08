"""Freeze current MH2 QK/softmax diagnosis using the existing compiled library; no launch."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
S=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_two_decoder_internal_20260909_v1';H=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_FA19_native_hybrid_20260909_v1';V=A/'snapshot${ARTIFACT_ROOT}/codex_dt_fa19_qk_softmax_diagnostic_20260908_v1'
sp,sr=[json.loads((S/name).read_bytes()) for name in ['protocol.json','results.json']]
hp,hr=[json.loads((H/name).read_bytes()) for name in ['protocol.json','results.json']]
vp,vr,vb=[json.loads((V/name).read_bytes()) for name in ['protocol.json','results.json','build_results.json']]
assert sr['status']=='MH2_FA19_GDN1_1native10replay2finite_internal_complete'
assert hr['status']=='MH2_eleven_native_FA_hybrid_contrasts_complete' and hr['native_FA_calls_entered']==hr['native_FA_calls_returned']==11
assert vr['status']=='three_FA19_conditional_score_contractions_complete' and vb['status']=='finite_extension_compiled_not_executed'
assert vb['protocol']==vp and vr['diagnostic_library_sha256']==vb['library']['sha256']
for directory in [S,H,V]:
    receipt=json.loads((directory/'terminal_receipt.json').read_bytes());assert receipt.get('pid_alive',receipt.get('proc_exists',False)) is False
    for name,item in receipt['files'].items():assert sha((directory/name).read_bytes())==(item if isinstance(item,str) else item['sha256']),name
summary=A/'dt_MH2_FA19_native_hybrid_summary_20260909.json';su=json.loads(summary.read_bytes())
assert su['status']=='MH2_FA19_native_hybrid_independent_CPU_audit_passed' and su['results_sha256']==sha((H/'results.json').read_bytes())
assert hp['source_results_sha256']==sha((S/'results.json').read_bytes())
assert hp['input_sha256']==sr['input']['input_sha256']=='c378ebde9024118211213c6bdbf9e2c2fc18a65f2bb7a91d3e1d0707bc430a76'
assert sr['input']['total_length']==710
files={'study.py':(R/'research/reproduction_templates/dt_MH2_FA19_qk_softmax_diagnostic_20260909.py').read_bytes()}
for name,path in [('vendor_fa_conditional_diag_20260908.py',R/'research/runtime'),('vendor_fa_finite_bf16_d256.py',R/'research/runtime'),
    ('vendor_fa_finite_p1_bf16_d256_conditional_diag_20260908.cu',R/'research/prototypes')]:
    raw=(path/name).read_bytes();assert raw==(V/name).read_bytes() and sha(raw)==vp['files_sha256'][name],name;files[name]=raw
oldraw=(V/'study.py').read_bytes();assert oldraw==(R/'research/reproduction_templates/dt_fa19_qk_softmax_diagnostic_20260908.py').read_bytes()
oldtree,newtree=ast.parse(oldraw),ast.parse(files['study.py'])
for name in ['prediction','stats','drift','layout']:
    get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(get(oldtree))==ast.dump(get(newtree)),name
old=oldraw.decode().replace('\r\n','\n');new=files['study.py'].decode().replace('\r\n','\n')
a="        dq = actual['0']['query'].double()";b="        deleted = set("
assert old.split(a)[1].split(b)[0]==new.split(a)[1].split(b)[0],'Conditional diagnostic algebra changed'
assert sp['installed_FA_interface_sha256']==vp['installed_FA_interface_sha256']==hp['installed_FA_interface_sha256']
assert sp['finite_FA_library_sha256']==vp['production_library_sha256']
lib=vb['library'];assert lib['sha256']=='e7a320beaec2ea0642b994f0d1d0ce15211d8f50e26ee523fc0e77bdbd10d99e' and lib['bytes']==463976
private=sr['private_artifact'];assert private['sha256']==hp['private_artifact_sha256'] and private['bytes']==hp['private_artifact_bytes']
kwargs=dict(hp['native_FA_kwargs']);kwargs['return_attn_probs']=True
p=copy.deepcopy(vp)
for key in list(p):
    if key.startswith('next_action_'):p.pop(key)
p.pop('build_budget',None)
remote='${ARTIFACT_ROOT}/codex_dt_MH2_FA19_qk_softmax_diagnostic_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
protected=[]
for directory,names in [(S,['results.json','protocol.json']),(H,['results.json','protocol.json','signed_token_contrasts.npz']),
    (V,['protocol.json','build_results.json','results.json','study.py','vendor_fa_finite_p1_bf16_d256_conditional_diag_20260908.cu','vendor_fa_conditional_diag_20260908.py'])]:
    protected.extend({'path':'/tmp/'+directory.name+'/'+name,'sha256':sha((directory/name).read_bytes())} for name in names)
library_path='/tmp/'+V.name+'/'+vp['diagnostic_library_name'];protected.extend([
    {'path':library_path,'sha256':lib['sha256']},{'path':vp['production_library_path'],'sha256':vp['production_library_sha256']}])
protected.extend(hp['argument_provenance']['sources'])
p.update({
    'purpose':'Current MH2 FA19 routing-at-V0 diagnostic on exact saved coordinates. Reuse already compiled conditional library and wrappers without recompilation; one publicB2FA LSE plus three diagnostic calls. Preserve the larger already measured PV-reference interaction separately. No candidate or baseline replacement.',
    'fixed_steps':['3','10','20'],'case':'morehopqa_2','layer':19,'total_length':710,
    'source_results_path':'/tmp/'+S.name+'/results.json','hybrid_results_path':'/tmp/'+H.name+'/results.json',
    'hybrid_vectors_path':'/tmp/'+H.name+'/signed_token_contrasts.npz',
    'private_artifact_path':'/tmp/'+S.name+'/'+private['file'],'private_artifact_sha256':private['sha256'],'private_artifact_bytes':private['bytes'],
    'input_sha256':sr['input']['input_sha256'],
    'frozen_input_receipts':{step:hr['points'][step]['input_receipt'] for step in ['3','10','20']},
    'original_route_error':{step:hr['points'][step]['fields']['routing_at_baseline_values_prediction_error']['net'] for step in ['3','10','20']},
    'PV_reference_interaction_preserved':{step:hr['points'][step]['fields']['routing_content_interaction_contrast']['net'] for step in ['3','10','20']},
    'source_hybrid_independent_audit':{'file':summary.name,'sha256':sha(summary.read_bytes()),'scope':'Three-term/seed/group/current-source-core closure audited; PV reference interaction exceeds routing error at both early3 and mid10. This diagnostic cannot ignore that larger term.'},
    'reused_build':{'build_results_path':'/tmp/'+V.name+'/build_results.json','build_results_sha256':sha((V/'build_results.json').read_bytes()),
        'protocol_path':'/tmp/'+V.name+'/protocol.json','protocol_sha256':sha((V/'protocol.json').read_bytes()),
        'library_path':library_path,'library_sha256':lib['sha256'],'library_bytes':lib['bytes'],
        'successful_previous_run_results_sha256':sha((V/'results.json').read_bytes()),
        'scope':'The existing T-parameterized BF16/D256 diagnostic ABI and three phases; no build script executed and no new compiler invocation. Historical build cost is provenance, not a new job cost.'},
    'protected_sources':protected,'native_FA_kwargs':kwargs,'native_scaling_provenance':hp['scaling_verification'],
    'build_budget':{'compiler_attempts':0,'compiler_seconds':0,'GPU_kernel_calls':0,'model_calls':0},
    'run_budget':{'wall_seconds':180,'native_FA_calls':1,'batch_size_for_native_LSE':2,'diagnostic_finite_calls':3,
        'diagnostic_phase_launches_on_success':9,'extra_score_MMA_tiles_per_Phase1_tile':2,'extra_warmup':0,
        'compiler_attempts':0,'model_calls':0,'DT_calls':0,'scorer_calls':0,'FT_calls':0,'backward_calls':0,'generation_calls':0,'new_samples':0},
    'source_adaptation':'Current private[19][coeff], private[19][native][B2] and private[19][native][0/3/10/20] replace prior MH1 nested schema. Shape T710, exact source masks, actual native scale and post-H2D contiguous layouts. Wrapper/extension byte-identical; prediction function and four-term algebra unchanged.',
    'acceptance':'All source/library/ABI/artifact/shape/nativeargs/mask identities hold. Record current B2 output and saved qk coefficient replay drift; coefficients independent of actualA. Four-term and group closure below1e-7 against current frozen hybrid R0. Do not replace baseline with regenerated coefficients or force their equality to source.',
    'stop':'Zero compiler invocations. First runtime/source/layout/nonfinite/invariant failure stops and preserves entered/returned/unknown partial phase counts; no automatic retry, new kernel, precision change, extra full-model/scorer point or candidate sweep.',
    'next_action':'Report observed QK interaction, tile-cast, coefficient replay and finite-softmax conditional terms beside the larger PV-reference interaction. Root separately assesses existing endpoint allocation choices; do not repeat the stopped Euclidean/supported-softmax candidates or infer MAS benefit.',
    'artifact_contract':'Current private2.626GB capture stays remote with SHA guard. New diagnostic_coefficients_private.pt retains actual dq/dk/dv/tau/center/LSE/rows privately; public binary NPZ retains contractions. Existing library is read at original remote path, never overwritten or included as new build.',
    'files_sha256':{name:sha(raw) for name,raw in files.items()}})
for name,data in files.items():
    if name.endswith('.py'):ast.parse(data,filename=name)
files['protocol.json']=json.dumps(p,indent=2).encode();pp=A/'dt_MH2_FA19_qk_softmax_diagnostic_protocol_20260909.json';lp=A/'launch_dt_MH2_FA19_qk_softmax_diagnostic_20260909.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen artifacts.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(data).decode() for name,data in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'payload':str(lp),'remote_directory':remote,'run_budget':p['run_budget'],'build_budget':p['build_budget']}))
