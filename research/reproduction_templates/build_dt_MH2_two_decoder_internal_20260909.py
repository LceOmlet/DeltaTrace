"""Freeze MH2 FA19 internal replay diagnostic; never launch a model job."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_current_conditional_boundaries_20260909_v1'
p0=json.loads((D/'protocol.json').read_bytes());r0=json.loads((D/'results.json').read_bytes())
assert r0['status']=='MH2_current_conditional_boundaries_1DT4score_complete' and r0['protocol']==p0
receipt=json.loads((D/'terminal_receipt.json').read_bytes());assert receipt.get('pid_alive',receipt.get('proc_exists',False)) is False
for name,item in receipt['files'].items():assert sha((D/name).read_bytes())==(item if isinstance(item,str) else item['sha256']),name
assert r0['sources_before']==r0['sources_after'] and r0['weight_stats_before']==r0['weight_stats_after']
summary=A/'dt_MH2_current_conditional_boundaries_summary_20260909.json';sr=json.loads(summary.read_bytes())
assert sr['status']=='MH2_current_conditional_boundaries_independent_CPU_audit_passed'
assert sr['results_sha256']==sha((D/'results.json').read_bytes())
assert sr['steps']['3']['largest_overprediction_layer']==sr['steps']['10']['largest_overprediction_layer']==19
files={'study.py':(R/'research/reproduction_templates/dt_MH2_two_decoder_internal_20260909.py').read_bytes()}
for name,want in p0['files_sha256'].items():
    if name=='study.py':continue
    raw=(D/name).read_bytes();assert sha(raw)==want,name
    matches=[d/name for d in [R/'core',R/'research/runtime',R/'research/reproduction_templates'] if (d/name).is_file()]
    assert len(matches)==1 and matches[0].read_bytes()==raw,('Current runtime changed',name)
    files[name]=raw
helper='decoder19_conditional_decomposition_20260908.py';raw=(R/'research/reproduction_templates'/helper).read_bytes()
old=A/'snapshot${ARTIFACT_ROOT}/codex_dt_decoder19_6_conditional_20260908_v1';oldr=json.loads((old/'results.json').read_bytes())
assert sha(raw)=='32608e57d958c7b2bc3aa657ac240902d49505c3c3d93080ba11de67091ef982'
assert raw==(old/helper).read_bytes() and sha(raw)==oldr['protocol']['files_sha256'][helper];files[helper]=raw
helper2='existing_two_decoder_ledger_20260909.py';data2=(R/'research/reproduction_templates'/helper2).read_bytes()
assert sha(data2)=='28d943dfd6c8bf8d42ee6883b2ca55e87d3031ad916b9a4619ffe6c802e2df29'
archived=(R/'research/reproduction_templates/dt_decoder19_6_conditional_20260908.py').read_bytes()
original={n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(archived)) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for node in ast.parse(data2).body:
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)):assert ast.dump(node,include_attributes=False)==original[node.name]
files[helper2]=data2
case=r0['cases']['morehopqa_2'];private=case['private_artifact'];info=case['input']
assert info['total_length']==710 and info['input_sha256']=='c378ebde9024118211213c6bdbf9e2c2fc18a65f2bb7a91d3e1d0707bc430a76'
assert private['sha256']=='3bf0471df88faac2f068ecda771c9ee4f97904c85ef5f9cba4321c847e4fd413'
remote='${ARTIFACT_ROOT}/codex_dt_MH2_two_decoder_internal_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p=copy.deepcopy(p0)
for key in ['case_indices','call_schedule','fixed_records','source_standalone','source_mask_CPU_freeze','frozen_capture_receipts','frozen_source_scores','evaluation','signed_outputs','endpoint_rule','diagnostic_contract_provenance','utility_provenance','source_mask_CPU_freeze']:
    p.pop(key,None)
p.update({
    'scope':'MH2 FA19 and GDN1 internal replay ledger using saved actual upstream and inputs. One original B2 forward obtains native kwargs; ten native selected-layer replays and two unchanged finite decoder pullbacks. No complete DT/scorer/FT/generation/new candidate.',
    'selected_layers':[19,1],'capture_steps':['B2','0','3','10','20'],'input':info,
    'source_boundary':{'results_path':'/tmp/'+D.name+'/results.json','results_sha256':sha((D/'results.json').read_bytes()),
        'protocol_path':'/tmp/'+D.name+'/protocol.json','protocol_sha256':sha((D/'protocol.json').read_bytes()),
        'vectors_path':'/tmp/'+D.name+'/vectors.npz','vectors_sha256':sha((D/'vectors.npz').read_bytes()),
        'private_path':'/tmp/'+D.name+'/'+private['file'],'private_sha256':private['sha256'],'private_bytes':private['bytes'],
        'scope':'Existing same-run m1/m2/m19/m20 and actual B2 plus B1 clean/step3/step10/allEOS hidden boundaries; source current-control deletion masks remain fixed. No reattribution or rescoring to obtain them.'},
    'independent_boundary_audit':{'file':summary.name,'sha256':sha(summary.read_bytes()),
        'selection':'FA19 is largest at both early/mid points; GDN1 is the largest GDN mid jump. Inspect both rather than assuming all error shares the prior MH0 attention cause.'},
    'existing_MH1_helper_provenance':{'study_results_sha256':sha((old/'results.json').read_bytes()),'helper':helper,'helper_sha256':sha(raw),
        'scope':'Reuse only the frozen nine-term algebra and exact native GQA layout; MH1 measured values do not establish current MH2 causes.'},
    'GDN_ledger_provenance':{'helper':helper2,'helper_sha256':sha(data2),'archived_study_sha256':sha(archived),'AST_identical_all_function_and_class_definitions':True,'scope':'Pure 16-term contractions and passive fresh-cache observer only. No native forward replacement.'},
    'required_decoder_kwargs':['position_embeddings','position_ids','attention_mask','past_key_values','use_cache'],
    'kwargs_contract':'Capture actual arguments at layer1/19 in one official B2 FA/BF16 forward, original logits_to_keep1, use_cacheFalse. Assert paired text positions/RoPE rows equal and actual mask None or all-ones [2,T]. B1 takes the original clean row and native fresh DynamicCache, with no prior state. No invented rotary/mask/positions/cache implementation.',
    'rule_scope':'Current unmodified contentP1 FA and content1 GDN1 with saved upstream from current DT. Existing layer0 repairs retained only through saved coefficients, not repropagated.',
    'backend_schedule':'Official BF16 model load with eager loading flag then native FA setter. No extra NI0/eager initialization. One native B2 kwargs capture, five FA19 and five GDN1 native replays, one unchanged finite decoder per selected layer, one public FA LSE.',
    'budget':{'model_loads':1,'whole_original_B2_forward_for_kwargs':1,'native_head_rows_per_endpoint_in_kwargs_forward':1,
        'native_FA_calls_in_kwargs_forward':8,'native_GDN_FLA_calls_in_kwargs_forward':24,'native_GDN_conv_calls_in_kwargs_forward':24,
        'selected_native_decoder_replays':10,'selected_native_FA_calls_in_replays':5,'selected_native_GDN_FLA_and_conv_calls_in_replays':5,'selected_finite_decoder':2,
        'public_FA_LSE_auxiliary_calls':1,'finite_FA_calls':1,'finite_FA_native_kernel_launches':3,'finite_FLA_calls':1,'finite_FLA_adjoint_stages':2,
        'extra_finite_GDN_public_conv_preactivation_and_autograd_each':1,'complete_DT_calls':0,'original_scorer_calls':0,'FT_calls':0,'generation_calls':0,'new_candidates':0,'wall_time_seconds':600},
    'diagnostic_sign':'Prediction minus actual. Nine FA or sixteen GDN internal terms plus saved-minus-replayed coefficient, input replay transfer and output replay transfer equal the original saved boundary mismatch.',
    'drift_policy':'Report native root input/output and selected replay output drift, regenerated coefficient drift and all three transfer contractions. BF16 numerical drift is retained, no bitwise or precision-polishing requirement.',
    'outputs':'Public scalar/group internal ledgers and numeric per-token transfer contractions. Remote private PT keeps selected actual native features, coefficients, original boundaries and official kwargs. No model weights or private PT in Git/ZIP.',
    'cost_scope':'Full/native/finite/aux entered and returned calls, capture, CPU reductions and private saving included. Diagnostic timing is not production speed. Root native op counts inferred from verified official graph/layer count; selected op counts passively observed.',
    'source_reuse':'All17 non-study runtime files byte-identical to successful MH2 boundary job; two preexisting pure helper definitions and new diagnostic orchestration only.',
    'decision':'Separate FA PV/softmax combined core, input/gate/MLP/RMS, and GDN FLA recurrence, normgate, conv/QK/scalars. Only promote an internal cause hypothesis if actual terms and replay transfer support it; no MAS/RISE/needle or multi-case improvement claim.',
    'stop':'First source/input/nativekwargs/cache/shape/call/closure failure or600s; retain failure and no automatic rerun, alternate masks, native patches, full DT/scorer or candidate search.',
    'protected_sources':[{'path':p0['finite_FA_library'],'sha256':p0['finite_FA_library_sha256']}]+[
        {'path':'/tmp/'+D.name+'/'+name,'sha256':sha((D/name).read_bytes())} for name in ['results.json','protocol.json','vectors.npz']],
    'files_sha256':{name:sha(data) for name,data in files.items()}})
for name,data in files.items():ast.parse(data,filename=name)
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_MH2_two_decoder_internal_protocol_20260909.json';lp=A/'launch_dt_MH2_two_decoder_internal_20260909.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen artifacts.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(data).decode() for name,data in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget']}))
