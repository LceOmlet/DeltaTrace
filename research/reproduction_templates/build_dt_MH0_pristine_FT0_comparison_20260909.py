"""Freeze one same-case MH0 original FT0/current DT comparison; no execution."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_layer0_fla_endpoint_average_stability_cost_20260909_v1'
dp=json.loads((donor/'protocol.json').read_bytes());dr=json.loads((donor/'results.json').read_bytes())
assert dr['status']=='layer0_FLA_endpoint_average_stability_cost_10DT245FLA84score_complete'
ft_donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v2'
fp=json.loads((ft_donor/'protocol.json').read_bytes());fr=json.loads((ft_donor/'results.json').read_bytes())
assert fr['status']=='fixed_DT_FT_three_author_cases_complete'
native_ft=A/'snapshot${ARTIFACT_ROOT}/codex_official_ft_exp2_default_20260908_v1'
nr=json.loads((native_ft/'results.json').read_bytes());assert nr['status']=='unchanged_official_FT_NI0_trace_completed'
for key in ('checkpoint','checkpoint_config_tokenizer_sha256','native_model_sha256','runtime_source_sha256','official_source_blob_sha1','cache_paths','cache_hashes'):
    assert dp[key]==fp[key],key
files={'study.py':(R/'research/reproduction_templates/dt_MH0_pristine_FT0_comparison_20260909.py').read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(donor/name).read_bytes();assert sha(raw)==want
    paths=[root/name for root in (R/'core',R/'research/runtime',R/'research/reproduction_templates') if (root/name).is_file()]
    assert len(paths)==1 and paths[0].read_bytes()==raw,('Current method changed',name)
    files[name]=raw
records={key:copy.deepcopy(dp['fixed_records'][key]) for key in ('niah_mq_q2_0','morehopqa_0')}
for key in records:
    records[key]['expected_input']=dr['cases'][key]['input'];records[key]['expected_gold']=dr['cases'][key]['gold']
    records[key]['provenance']='Exact same original cached record, actual input and remapped gold from successful20260909 stability study; no previous scores substituted.'
assert records['morehopqa_0']['expected_input']['total_length']==853
assert records['morehopqa_0']['expected_input']['input_sha256']=='6e1df21d426c5eb28217976f63b7c7932e502bde4fb064f77867df8205ebd618'
assert records['morehopqa_0']['metadata_id']=='5ae457a95542996836b02c85_4'
assert not records['morehopqa_0']['expected_gold']
remote='${ARTIFACT_ROOT}/codex_dt_MH0_pristine_FT0_comparison_20260909_v1'
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p=copy.deepcopy(dp)
for key in ('prior_wholepilot','NI0_old_input_provenance','fixed_control_masks','quality_cases','cost','source_reuse','decision','evaluation_scope'):
    p.pop(key,None)
p.update({
    'scope':'One same-case MH0 comparison against predeclared fixed original FT0. Entire unchanged e81b3be Both trace(hops3), retain all hop vectors, score only FT0 and fixed current DT. No new candidate, framework patch, saved-output FT substitute, hop sweep or execution from this builder.',
    'case_indices':[['niah_mq_q2',0],['morehopqa',0]],'fixed_records':records,
    'call_schedule':[['morehopqa_0','DT']],'quality_schedule':[['morehopqa_0','FT0'],['morehopqa_0','DT']],
    'prior_stability':{'results_path':'/tmp/'+donor.name+'/results.json','results_sha256':sha((donor/'results.json').read_bytes()),'scope':'Input and unchanged DT/source identity only; current comparison obtains new real scores for both methods.'},
    'FT_commit':'e81b3be50a48dcfc652fbf1b530069b552736e66',
    'official_package_blob_sha1':fp['official_package_blob_sha1'],
    'FT_public_call':"FlashTrace(model,tokenizer).trace(prompt=original_prompt,target=original_target,output_span=tuple(author_remapped_sink),reasoning_span=tuple(author_remapped_thinking),hops=3,method='flashtrace')",
    'FT0_definition':"Returned metadata['ifr']['observation_projected']['base']; full public trace internally calls original LLMIFRAttributionBoth.calculate_ifr_multi_hop_both. FT1-3 are cumulative returned base+per_hop vectors, saved but never scored or selected by quality.",
    'successful_FT_provenance':[
        {'path':'/tmp/'+ft_donor.name+'/study.py','sha256':sha((ft_donor/'study.py').read_bytes()),'results_sha256':sha((ft_donor/'results.json').read_bytes()),'reuse':'Same successful full public trace/extraction on Qwen3.5 NI1,NI2,MH1.'},
        {'path':'/tmp/'+native_ft.name+'/study.py','sha256':sha((native_ft/'study.py').read_bytes()),'results_sha256':sha((native_ft/'results.json').read_bytes()),'reuse':'Same passive original Both/native call counting from successful NI0 reference.'}],
    'method_target_semantics':'Shared exact raw853-token input, original author fixed response+tokenizerEOS, eligible set and original full-response evaluator. Preserve original FT hop0 answer/sink representation and recursive all-generation policy versus existing DT full fixed-response logprob including EOS. These internal seeds/targets are not identical; do not change either method to pretend otherwise.',
    'source_reuse':'Every non-study finite runtime/core/normal utility/span helper byte-identical to successful stability study. All16official FT package sources and original evaluator pinned before/after; only passive model hooks/Python counters, no author function replacement or custom recurrence.',
    'backend_schedule':'Load original Qwen3.5 BF16 eager; one originalNI0 eager initialization; official setter toFA for one currentDT; setter toeager for one unmodifiedFT public trace;finally restoreFA for42common original scorer forwards. Same model remains resident. Native module identities checked after FT.',
    'budget':{'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_attributions':1,
        'DT_B2_roots':1,'native_decoder_replays_DT':32,'finite_decoder_calls':32,'public_FA_LSE_calls':8,'finite_FA_calls':8,
        'finite_GDN_callback_sites':24,'finite_FLA_backend_calls':25,'native_FLA_adjoint_stage_calls':50,
        'candidate_layer0_average_wrapper_calls':1,'finite_GDN_conv_preactivation_calls':24,'finite_GDN_conv_autograd_calls':24,
        'complete_official_FT_Both_calls':1,'FT_public_hops':3,'FT_native_B1_roots':1,
        'FT_native_eager_attention_calls':8,'FT_hybrid_linear_layer_inputs':24,'FT_hybrid_full_layer_inputs':8,
        'FT_native_FLA_forwards':72,'FT_identity_valued_FLA_probes_included':24,
        'original_metric_curves':2,'native_B1_scorer_forwards':42,'FT_hops_scored':[0],'FT_hops_saved_not_scored':[1,2,3],
        'generation_calls':0,'new_samples':0,'wall_time_seconds':600},
    'cost':'One cold/single complete DT and complete original Both trace. Record all model-resident peaks/time and source verification;FT full trace includes all identity-valued probes/hops and passive Python counter overhead. This is quality comparison with actual cost receipts,not a fair warmed FT0-only speed benchmark. No FT acceleration or output reuse.',
    'evaluation_scope':'Fresh42actual defaultFA/BF16 common original scorer forwards;unchanged faithfulness_test_skip_tokens k20. Preserve all21 raw scores, masks, density, monotone normalized response, penalties and original three returned metrics. RISE/MAS lowerbetter. No fabricated MH needle score. Existing FT1-3 quality remains untested on MH0.',
    'decision':'Use predeclared FT0 to assess the current MH0 residual objectively, not merely its improvement over previousDT. Report every metric/cost difference and target-semantic limit. Root separately decides after read-only review; no FT source repair, morehop scoring, new method/target ablation or followup authorized here.',
    'stop':'First official/source/input/span/count/nonfinite/metric invariant failure or600seconds. Preserve original exception and entered/returned counts. No repeated FT, precision polishing, clipping, altered bounds/framework, sample swap or extra curves. An original FT numerical exception is evidence to inspect,not permission to modify FT.',
    'protected_sources':[{'path':dp['finite_FA_library'],'sha256':dp['finite_FA_library_sha256']},
        {'path':'/tmp/'+donor.name+'/results.json','sha256':sha((donor/'results.json').read_bytes())},
        {'path':'/tmp/'+donor.name+'/protocol.json','sha256':sha((donor/'protocol.json').read_bytes())},
        {'path':'/tmp/'+ft_donor.name+'/results.json','sha256':sha((ft_donor/'results.json').read_bytes())},
        {'path':'/tmp/'+native_ft.name+'/results.json','sha256':sha((native_ft/'results.json').read_bytes())}],
    'files_sha256':{name:sha(raw) for name,raw in files.items()}})
for name,raw in files.items():ast.parse(raw,filename=name)
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_MH0_pristine_FT0_comparison_protocol_20260909.json';lp=A/'launch_dt_MH0_pristine_FT0_comparison_20260909.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen artifacts.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget']}))
