"""Freeze a fixed-candidate developer stability and ABBA-cost study; do not launch."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_layer0_fla_endpoint_average_whole_pilot_20260908_v1'
dp=json.loads((donor/'protocol.json').read_bytes());prior=json.loads((donor/'results.json').read_bytes())
assert prior['status']=='layer0_FLA_endpoint_average_4DT98FLA84score_pilot_complete'
old=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v1'
op=json.loads((old/'protocol.json').read_bytes());oi=json.loads((old/'results.json').read_bytes())['control']['input']
for key in ('cache_paths','cache_hashes','checkpoint','checkpoint_config_tokenizer_sha256','native_model_sha256'):
    assert dp[key]==op[key],key
files={'study.py':(R/'research/reproduction_templates/dt_layer0_fla_endpoint_average_stability_cost_20260909.py').read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(donor/name).read_bytes();assert sha(raw)==want
    locations=[root/name for root in (R/'core',R/'research/runtime',R/'research/reproduction_templates') if (root/name).is_file()]
    assert len(locations)==1 and locations[0].read_bytes()==raw,('Production source differs from completed pilot',name)
    files[name]=raw
helper=R/'research/runtime/official_span_mapping.py'
assert sha(helper.read_bytes())==op['files_sha256']['official_span_mapping.py']
files[helper.name]=helper.read_bytes()
records={}
for dataset,index in [('niah_mq_q2',0),('morehopqa',0),('niah_mq_q2',1)]:
    raw=(A/'snapshot'/Path(dp['cache_paths'][dataset].lstrip('/'))).read_bytes()
    assert sha(raw)==dp['cache_hashes'][dataset]
    line=raw.decode().splitlines()[index];rec=json.loads(line);key=f'{dataset}_{index}'
    records[key]={'source_record_sha256':sha(line.encode()),'cached_sink_span':rec['sink_span'],
        'cached_thinking_span':rec['thinking_span'],'metadata_id':rec['metadata'].get('id'),
        'fixed_target_text_sha256':sha(rec['target'].encode()),'prompt_text_sha256':sha(rec['prompt'].encode()),
        'expected_input':None,'expected_gold':None}
    if key=='niah_mq_q2_0':
        assert records[key]['source_record_sha256']==oi['source_record_sha256']
        records[key]['expected_input']={k:oi[k] for k in ('input_sha256','prompt_length','target_length','total_length','keep')}
        records[key]['expected_gold']=oi['gold']
        records[key]['provenance']='Prior failed three-case study CPU-prepared control only; valid raw-cache/input/span identity, not a successful quality result. Raw input is588 tokens; do not substitute the older605-token chat-template input.'
    elif key=='niah_mq_q2_1':
        records[key]['expected_input']=prior['cases'][key]['input'];records[key]['expected_gold']=prior['cases'][key]['gold']
    else:
        records[key]['provenance']='Exact author processed record0. No historical Qwen3.5 input hash asserted. Original old-tokenizer spans are checked, then original author functions retokenize gold/spans before model load; full actual input ids are saved and frozen then.'
for name,want in op['span_source_sha256'].items():
    raw=(A/'snapshot${FLASHTRACE_ROOT}'/name).read_bytes().replace(b'\r\n',b'\n')
    assert sha(raw)==want,name
remote='${ARTIFACT_ROOT}/codex_dt_layer0_fla_endpoint_average_stability_cost_20260909_v1'
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p=copy.deepcopy(dp)
for key in ('paired_references','existing_runtime_change','parent_control_anchor','fixed_control_masks','risks','decision'):
    p.pop(key,None)
schedule=[['niah_mq_q2_1',m,phase] for m,phase in [('control','warm'),('candidate','warm'),('control','measured'),('candidate','measured'),('candidate','measured'),('control','measured')]]
quality=[['morehopqa_0','control'],['morehopqa_0','candidate'],['niah_mq_q2_0','candidate'],['niah_mq_q2_0','control']]
schedule += [x+['quality'] for x in quality]
p.update({
    'scope':'Fixed existing layer0 FLA endpoint-average candidate. Two author-processed developer stability cases NI0/MH0, plus fixed NI1 C/S warmups and C/S/S/C cost. No new rule, layer selection, data generation, historical score substitution, FT call or launch here.',
    'case_indices':[['niah_mq_q2',0],['morehopqa',0],['niah_mq_q2',1]],
    'quality_cases':['morehopqa_0','niah_mq_q2_0'],'call_schedule':schedule,'quality_schedule':quality,
    'fixed_records':records,'old_checkpoint':op['old_checkpoint'],'author_data_root':op['author_data_root'],
    'span_source_sha256':op['span_source_sha256'],'max_total_tokens':2048,
    'input_pipeline':'Exact previous wholepilot: tokenize author rec.prompt without added special tokens; append tokenized unchanged rec.target+EOS; EOS replaces only author keep_token_indices. No chat-template insertion. Original author attach_spans_from_answer reproduces cached old-tokenizer indices; then remaps with current tokenizer. Original ruler_gold_prompt_token_indices returns current gold; never copy old indices blindly.',
    'prior_wholepilot':{'results_path':'/tmp/'+donor.name+'/results.json','results_sha256':sha((donor/'results.json').read_bytes()),
        'scope':'Input/source/method provenance only; no prior scores consumed for current metrics.'},
    'NI0_old_input_provenance':{'results_path':'/tmp/'+old.name+'/results.json','results_sha256':sha((old/'results.json').read_bytes()),
        'scope':'CPU prepared control input/gold only; prior study failed later. This is not a source of measured quality.'},
    'protected_sources':[{'path':dp['finite_FA_library'],'sha256':dp['finite_FA_library_sha256']},
        {'path':'/tmp/'+donor.name+'/results.json','sha256':sha((donor/'results.json').read_bytes())},
        {'path':'/tmp/'+donor.name+'/protocol.json','sha256':sha((donor/'protocol.json').read_bytes())},
        {'path':'/tmp/'+old.name+'/results.json','sha256':sha((old/'results.json').read_bytes())}],
    'budget':{'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_attributions':10,
        'control_DT':5,'candidate_DT':5,'cost_warm_DT':2,'cost_measured_DT':4,'quality_DT':4,
        'DT_B2_roots':10,'native_decoder_replays':320,'finite_decoder_calls':320,'public_FA_LSE_calls':80,
        'finite_FA_calls':80,'finite_GDN_callback_sites':240,'finite_FLA_backend_calls':245,
        'native_FLA_adjoint_stage_calls':490,'candidate_layer0_average_wrapper_calls':5,
        'candidate_additional_FLA_backend_calls':5,'finite_GDN_conv_preactivation_calls':240,
        'finite_GDN_conv_autograd_calls':240,'original_metric_curves':4,'native_B1_scorer_forwards':84,
        'FT_calls':0,'generation_calls':0,'new_samples':0,'wall_time_seconds':900},
    'source_reuse':'All normal runner, candidate wrapper, finite/native dependencies, utility and evaluator adapter exactly byte-identical to completed wholepilot. Only study orchestration and old author span helper inclusion change.',
    'cost':'Same resident full model and fixed NI1 input. Run C/S warmups, then C/S/S/C, then NI0/MH0 quality and scoring. Record every outer/normal-runner stage time and full peak allocated/reserved; resident memory before B2 setup and before/after attribute cleanup; initialization/compile/warm separately. No empty_cache or per-method model unloading. Endpoint swaps, six means, two backend calls at layer0 and normal CPU checkpoint copies remain timed. Two measured observations per method, not a precise population speed claim.',
    'evaluation_scope':'Unchanged author faithfulness_test_skip_tokens and evaluator, k20,84actual B1 scores. All21 actual scores, normalized response, density, alignment penalty, corrected scores, sorted keep, full input hashes and signed vectors retained. RISE and MAS lower is better. Needle only when actual author gold exists; MH0 has no invented needle labels.',
    'fixed_control_masks':'After fresh own control21-point scores, contract both newly obtained signed vectors on that same actual control deletion set;0extra scorers. This is a conditional magnitude ledger, not a second MAS curve.',
    'decision':'Developer stability and cost only, not independent heldout confirmation. Report each case and both RISE/MAS/needle directions without hiding tradeoffs; no automatic new candidate, weight change, layer scan or sample expansion. Existing NI1/MH1 results remain separate. Root reviews frozen payload before any execution.',
    'stop':'First source/input/span/layout/count/nonfinite/metric invariant failure or900seconds; preserve partial evidence and actual entered/returned work. No retry, substitute example, changed precision, added scores, native modification or FT execution.',
    'files_sha256':{name:sha(raw) for name,raw in files.items()}})
assert sum(x[1]=='candidate' for x in schedule)==5
assert p['budget']['finite_FLA_backend_calls']==24*len(schedule)+5
for name,raw in files.items():ast.parse(raw,filename=name)
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_layer0_fla_endpoint_average_stability_cost_protocol_20260909.json'
lp=A/'launch_dt_layer0_fla_endpoint_average_stability_cost_20260909.json'
assert not pp.exists() and not lp.exists(),'Never overwrite a frozen protocol/payload.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget']}))
