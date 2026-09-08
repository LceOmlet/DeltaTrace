"""Freeze shared-LSE current/reversed existing finite FA; no compile or launch."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
S=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_internal_20260909_v1';sp=json.loads((S/'protocol.json').read_bytes());sr=json.loads((S/'results.json').read_bytes())
H=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_native_hybrid_20260909_v1';hp=json.loads((H/'protocol.json').read_bytes());hr=json.loads((H/'results.json').read_bytes())
C=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_fa_build_20260908_v2';cp=json.loads((C/'protocol.json').read_bytes())
assert sr['status']=='MH0_current_FA19_1native_kwargs5replay1finite_internal_complete' and sr['protocol']==sp
assert hr['status']=='MH0_eleven_native_FA_hybrid_contrasts_complete' and hr['protocol']==hp
assert hr['native_FA_calls_entered']==hr['native_FA_calls_returned']==11
for directory in (S,H):
    receipt=json.loads((directory/'terminal_receipt.json').read_bytes());assert receipt.get('proc_exists',receipt.get('pid_alive')) is False
    for name,row in receipt['files'].items():
        assert sha(directory/name)==(row if isinstance(row,str) else row['sha256']),name
        if isinstance(row,dict):assert (directory/name).stat().st_size==row['bytes']
for key in ('source_results_sha256','source_protocol_sha256'):
    assert hp[key]==sha(S/('results.json' if key=='source_results_sha256' else 'protocol.json'))
boundary=A/'snapshot'/sp['source_boundary']['results_path'].lstrip('/');br=json.loads(boundary.read_bytes())
assert sha(boundary)==sp['source_boundary']['results_sha256'] and sr['input']==br['cases']['morehopqa_0']['input']
artifact=sr['private_artifact'];assert artifact['sha256']=='1637779cd62c6ffd05641600fc83eda87d9488ff99db1f78f4baaae0d7a93e0d' and artifact['bytes']==1272488075
cu='vendor_fa_finite_p1_bf16_d256.cu';wrapper='vendor_fa_finite_bf16_d256.py'
assert (R/'research/prototypes'/cu).read_bytes()==(C/cu).read_bytes() and sha(C/cu)==cp['extension_sha256']
assert (R/'research/runtime'/wrapper).read_bytes()==(S/wrapper).read_bytes() and sha(S/wrapper)==sp['files_sha256'][wrapper]
library=Path(sp['finite_FA_library']).name;assert sha(C/library)==sp['finite_FA_library_sha256']
review=A/'MH0_PV_reference_content_candidate_review_20260909.md'
files={'study.py':(R/'research/reproduction_templates/dt_MH0_FA19_PV_orders_20260909.py').read_bytes(),
    wrapper:(S/wrapper).read_bytes(),cu:(C/cu).read_bytes(),review.name:review.read_bytes()}
for name,raw in files.items():
    if name.endswith('.py'):ast.parse(raw,filename=name)
remote='${ARTIFACT_ROOT}/codex_dt_MH0_FA19_PV_orders_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
protected=[{'path':sp['finite_FA_library'],'sha256':sp['finite_FA_library_sha256']}]
for directory,names in [(S,['results.json','protocol.json']),(H,['results.json','protocol.json']),(C,['protocol.json',cu])]:
    protected += [{'path':'/tmp/'+directory.name+'/'+name,'sha256':sha(directory/name)} for name in names]
protected.append({'path':sp['source_boundary']['results_path'],'sha256':sp['source_boundary']['results_sha256']})
p={'scope':'One native B2 auxiliary FA obtains shared fresh LSE; two unchanged finite BF16 FA calls compute current and reverse PV order. Actual partial-deletion operands are CPU evaluation only; no production or native model change.',
    'case':'morehopqa_0','decoder_index':19,'input_sha256':sr['input']['input_sha256'],'fixed_steps':['3','10','20','B2'],
    'source_results_path':'/tmp/'+S.name+'/results.json','source_results_sha256':sha(S/'results.json'),
    'source_protocol_path':'/tmp/'+S.name+'/protocol.json','source_protocol_sha256':sha(S/'protocol.json'),
    'boundary_results_path':sp['source_boundary']['results_path'],'boundary_results_sha256':sp['source_boundary']['results_sha256'],
    'private_artifact_path':'/tmp/'+S.name+'/'+artifact['file'],'private_artifact_sha256':artifact['sha256'],'private_artifact_bytes':artifact['bytes'],
    'hybrid_provenance':{'results_path':'/tmp/'+H.name+'/results.json','results_sha256':sha(H/'results.json'),
        'fields':{step:{name:hr['points'][step]['fields'][name]['net'] for name in ('routing_content_interaction_contrast','R0','RA','qk_prediction','v_prediction','core_error_at_stored_seed')} for step in ('3','10','20')},
        'limit':'Opposite reference-routing contrasts motivate evaluating the complete changed PV operator; the measured interaction component alone does not guarantee either reverse or average improves complete core or whole-input attribution.'},
    'frozen_input_receipts':{step:br['cases']['morehopqa_0']['points'][step]['input_receipt'] for step in ('0','3','10','20')},
    'finite_FA_library':sp['finite_FA_library'],'finite_FA_library_sha256':sp['finite_FA_library_sha256'],
    'installed_FA_interface_sha256':sp['installed_FA_interface_sha256'],'native_FA_kwargs':hp['native_FA_kwargs'],
    'scale':hp['native_FA_kwargs']['softmax_scale'],
    'finite_source_provenance':{'compiled_build_protocol_sha256':sha(C/'protocol.json'),'compiled_extension_sha256':cp['extension_sha256'],
        'vendor_source_version':cp['vendor_source_version'],'installed_model_FA_version':cp['installed_model_FA_version'],
        'same_version_source_claim':cp['same_version_source_claim'],'scope':'Exact existing library and its archived FA-framework-derived source, no compilation or custom replacement of the native model attention.'},
    'rule':{'current':'Delta(PV)=DeltaP*V0+P1*DeltaV',
        'reverse':'Swap real q0/q1,k0/k1,lse0/lse1,place actual V1 in v0 slot,unchanged upstream u and scale,no sign flip:Delta(PV)=DeltaP*V1+P0*DeltaV.',
        'diagnostic_average':'FP64 arithmetic mean of the two returned BF16 dq/dk/dv tensors. Ideal symmetric PV allocation uses meanV/meanP, but this diagnostic is not claimed bitwise equal to a future single-pass implementation or to use one backend call.',
        'QK_softmax_scope':'Existing QK midpoint and logarithmic-mean softmax factors are endpoint-symmetric in the source. Routing seed center changes through V0 to V1 as required; QK and V predictions must both be compared.',
        'candidate_inputs':'Only source actual B2 endpoint q/k/v, one fresh native B2 LSE and saved upstream mcontent. No actual deletion A, gold, scores or continuous fitted weight enters candidate construction.'},
    'mathematical_review':{'file':review.name,'sha256':sha(review),'scope':'Independent source review of QK midpoint, symmetric logarithmic mean, route seed center and original endpoint1 dv; no new execution.'},
    'budget':{'native_public_B2_FA_calls':1,'existing_finite_FA_calls':2,'existing_finite_kernel_phases':6,'model_calls':0,
        'complete_DT_calls':0,'scorer_calls':0,'FT_calls':0,'generation_calls':0,'compile_calls':0,'native_backward_calls':0,
        'new_samples':0,'extra_warmups':0,'wall_time_seconds':180},
    'outputs':'Private actual current/reverse coefficients,FP64diagnostic average and shared LSE are hash-saved. Public signed token/group contractions for actual3/10/20 and fullB2 endpoints include Q/K/V/core,seed cast,current replay transfer,positive/negative masses and average closure. Raw1.272GB activations and coefficient .pt excluded from review ZIP.',
    'numerical_policy':'Default existing BF16 FA and finite outputs,CPUFP64 audit. Retain publicB2 output drift and same-LSE control minus saved coefficient/projection drift separately. No arbitrary bitwise admission test, no silent correction,actual unused return numel must0 to exclude explicit probability allocation.',
    'decision':'This is a bounded operator candidate evaluation, not new input attribution, MAS/RISE, native-model counterfactual or a production speed/memory claim. Mixed local signs must be interpreted through whole-input propagation before any promotion.',
    'stop':'First identity/layout/count/nonfinite/closure failure or180seconds;retain entered/returned and unknown unreturned phase counts. No extra call,precision retry,candidate scan or new kernel.',
    'protected_sources':protected,'files_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}}
assert p['scale']==256**-.5 and p['native_FA_kwargs']['dropout_p']==0
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_MH0_FA19_PV_orders_protocol_20260909.json';lp=A/'launch_dt_MH0_FA19_PV_orders_20260909.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite a frozen operator protocol.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(pp),'study_sha256':p['files_sha256']['study.py'],'payload':str(lp),'remote_directory':remote,'budget':p['budget']}))
