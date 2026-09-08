"""Freeze MH0 FA19 internal replay diagnostic; never launch a model job."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_current_conditional_boundaries_20260909_v1'
p0=json.loads((D/'protocol.json').read_bytes());r0=json.loads((D/'results.json').read_bytes())
assert r0['status']=='MH0_current_conditional_boundaries_1DT4score_complete' and r0['protocol']==p0
receipt=json.loads((D/'terminal_receipt.json').read_bytes());assert receipt.get('pid_alive',receipt.get('proc_exists',False)) is False
for name,item in receipt['files'].items():assert sha((D/name).read_bytes())==(item if isinstance(item,str) else item['sha256']),name
assert r0['sources_before']==r0['sources_after'] and r0['weight_stats_before']==r0['weight_stats_after']
summary=A/'dt_MH0_current_conditional_boundaries_summary_20260909.json';sr=json.loads(summary.read_bytes())
assert sr['status']=='MH0_current_conditional_boundaries_independent_CPU_audit_passed'
assert sr['results_sha256']==sha((D/'results.json').read_bytes())
assert sr['steps']['3']['largest_overprediction_layer']==sr['steps']['10']['largest_overprediction_layer']==19
files={'study.py':(R/'research/reproduction_templates/dt_MH0_FA19_internal_20260909.py').read_bytes()}
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
case=r0['cases']['morehopqa_0'];private=case['private_artifact'];info=case['input']
assert info['total_length']==853 and info['input_sha256']=='6e1df21d426c5eb28217976f63b7c7932e502bde4fb064f77867df8205ebd618'
assert private['sha256']=='69f7a134e9fbb3be164efe8e8a41005690fcfe4bf1dfa02247c7f0c3fcd177e2'
remote='${ARTIFACT_ROOT}/codex_dt_MH0_FA19_internal_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p=copy.deepcopy(p0)
for key in ['case_indices','call_schedule','fixed_records','source_standalone','source_mask_CPU_freeze','frozen_capture_receipts','frozen_source_scores','evaluation','signed_outputs','endpoint_rule','diagnostic_contract_provenance','utility_provenance','source_mask_CPU_freeze']:
    p.pop(key,None)
p.update({
    'scope':'Current MH0 FA decoder19 internal diagnostic on saved actual boundary coefficients and inputs. One original B2 model forward captures official positions/RoPE/mask kwargs; five selected native decoder19 replays plus one unchanged finite decoder19; no whole DT/scorer/FT/generation/new candidate.',
    'layer':19,'capture_steps':['B2','0','3','10','20'],'input':info,
    'source_boundary':{'results_path':'/tmp/'+D.name+'/results.json','results_sha256':sha((D/'results.json').read_bytes()),
        'protocol_path':'/tmp/'+D.name+'/protocol.json','protocol_sha256':sha((D/'protocol.json').read_bytes()),
        'vectors_path':'/tmp/'+D.name+'/vectors.npz','vectors_sha256':sha((D/'vectors.npz').read_bytes()),
        'private_path':'/tmp/'+D.name+'/'+private['file'],'private_sha256':private['sha256'],'private_bytes':private['bytes'],
        'scope':'Existing same-run m19/m20 and actual B2 plus B1 clean/step3/step10/allEOS hidden boundaries; source standalone deletion masks remain fixed. No reattribution or rescoring to obtain them.'},
    'independent_boundary_audit':{'file':summary.name,'sha256':sha(summary.read_bytes()),
        'selection':'Largest negative decoder jump at both frozen partial-delete points is layer19; other layers also contribute materially, so layer19 is not all remaining error.'},
    'existing_MH1_helper_provenance':{'study_results_sha256':sha((old/'results.json').read_bytes()),'helper':helper,'helper_sha256':sha(raw),
        'scope':'Reuse only the frozen nine-term algebra and exact native GQA layout; MH1 measured values do not establish current MH0 causes.'},
    'required_decoder_kwargs':['position_embeddings','position_ids','attention_mask','past_key_values','use_cache'],
    'kwargs_contract':'Obtain native kwargs at actual decoder19 entry in one complete original FA/BF16 B2 forward over original EOS/input paired tokens, use_cache=False and native logits_to_keep=1. Verify actual FA mask=None and identical text position/cos/sin endpoint rows. Replay B2 with full actual kwargs; B1 takes the unchanged actual clean endpoint row and original DynamicCache fresh per replay. No arange/rotary/mask/shadow-model implementation; no continued cache.',
    'rule_scope':'Unchanged existing FA finite content_P1 and symmetric native decoder boundary rules; saved upstream came from current full DT with layer0 normgate symmetric + layer0 FLA endpoint average. Neither layer0 nor other layers are propagated again.',
    'backend_schedule':'Load official BF16 model with eager loading setting, official setter to FA, then one full B2 native kwargs capture (no extra eager initialization). Five actual native FA19 replays using source saved boundary inputs; one B2 public FA LSE and one current finite decoder19. Full-forward guard prevents repeats.',
    'budget':{'model_loads':1,'whole_original_B2_forward_for_kwargs':1,'native_head_rows_per_endpoint_in_kwargs_forward':1,
        'native_FA_calls_in_kwargs_forward':8,'native_GDN_FLA_calls_in_kwargs_forward':24,'native_GDN_conv_calls_in_kwargs_forward':24,
        'selected_native_decoder_replays':5,'selected_native_FA_calls_in_replays':5,'selected_finite_decoder':1,
        'public_FA_LSE_auxiliary_calls':1,'finite_FA_calls':1,'finite_FA_native_kernel_launches':3,
        'finite_FLA_calls':0,'extra_finite_GDN_conv_calls':0,'complete_DT_calls':0,'original_scorer_calls':0,
        'FT_calls':0,'generation_calls':0,'new_candidates':0,'wall_time_seconds':900},
    'diagnostic_sign':'Prediction minus actual for nine internal terms. Add saved-minus-replayed m19, input replay transfer and output replay transfer, yielding the exact negative of previous actual-minus-predicted decoder19 jump per token.',
    'drift_policy':'Report captured current native B2 layer19 input/output versus source, every replay output versus source, regenerated m19 versus saved m19, plus all three token transfer terms. No bitwise requirement or precision repair, and never discard drift from the boundary mismatch.',
    'outputs':'Public original nine-term/per-token/group transfer ledger for early3,mid10,allEOS20 and paired B2. Public NPZ contains numeric contractions only. Remote private .pt retains selected source boundaries, actual native features including MLP operands, regenerated actual coefficients and official captured kwargs, no checkpoint weights; exclude .pt from reviewZIP/publicGit.',
    'cost_scope':'All full-forward/selected-replay/finite/FAaux entered and returned work plus source validation, passive capture, CPU singleton group reductions and private save are included. The one official kwargs forward also executes8nativeFA and24nativeGDN-FLA/conv. No diagnostic cost is a production speed estimate.',
    'source_reuse':'All current runtime/core/vendor finite sources remain byte-identical to the successful current MH0 boundary job; only a new standalone diagnostic study and existing frozen nine-term helper are assembled.',
    'decision':'Determine whether current MH0 FA19 error lies mainly in MLP, combined output-gate projection, FA core including seed cast, attention input or norm/residual terms. Do not claim routing versus PV decomposition within the core, MH1 cause transfer, precision failure or candidate effectiveness.',
    'stop':'First source/input/native kwargs/cache/shape/finite/call/closure failure or900seconds; preserve failure receipts and unknown unfinished work. No alternate masks, native patches, extra full forward, DT/score, precision polishing or extra FA hybrids.',
    'protected_sources':[{'path':p0['finite_FA_library'],'sha256':p0['finite_FA_library_sha256']}]+[
        {'path':'/tmp/'+D.name+'/'+name,'sha256':sha((D/name).read_bytes())} for name in ['results.json','protocol.json','vectors.npz']],
    'files_sha256':{name:sha(data) for name,data in files.items()}})
for name,data in files.items():ast.parse(data,filename=name)
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_MH0_FA19_internal_protocol_20260909.json';lp=A/'launch_dt_MH0_FA19_internal_20260909.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen artifacts.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(data).decode() for name,data in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget']}))
