"""Freeze four fixed two-case shards; never submit or run a model here."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
base=json.loads((A/'dt_MH0_pristine_FT0_comparison_protocol_20260909.json').read_bytes())
donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_layer0_fla_endpoint_average_stability_cost_20260909_v1'
dp=json.loads((donor/'protocol.json').read_bytes());dr=json.loads((donor/'results.json').read_bytes())
assert dr['status']=='layer0_FLA_endpoint_average_stability_cost_10DT245FLA84score_complete'
old=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v2'
oldr=json.loads((old/'results.json').read_bytes());assert oldr['status']=='fixed_DT_FT_three_author_cases_complete'
files={'study.py':(R/'research/reproduction_templates/dt_fixed_eight_case_regression_20260909.py').read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(donor/name).read_bytes();assert sha(raw)==want
    paths=[root/name for root in (R/'core',R/'research/runtime',R/'research/reproduction_templates') if (root/name).is_file()]
    assert len(paths)==1 and paths[0].read_bytes()==raw,('Unchanged production method required',name)
    files[name]=raw
records={};lengths={}
for dataset in ('niah_mq_q2','morehopqa'):
    raw=(A/'snapshot'/Path(base['cache_paths'][dataset].lstrip('/'))).read_bytes();assert sha(raw)==base['cache_hashes'][dataset]
    lines=raw.decode().splitlines()
    for index in range(4):
        line=lines[index];rec=json.loads(line);key=f'{dataset}_{index}'
        records[key]={'source_record_sha256':sha(line.encode()),'metadata_id':rec['metadata'].get('id'),
            'prompt_text_sha256':sha(rec['prompt'].encode()),'fixed_target_text_sha256':sha(rec['target'].encode()),
            'cached_sink_span':rec['sink_span'],'cached_thinking_span':rec['thinking_span'],
            'expected_input':None,'expected_gold':None,'prompt_characters':len(rec['prompt']),'target_characters':len(rec['target']),
            'provenance':'Original processed author cache, exact contiguous indices0-3, fixed existing target; original author helper rechecks old tokenizer spans and remaps gold before model load.'}
        source=dr['cases'].get(key) or oldr['cases'].get(key)
        if source:
            records[key]['expected_input']={k:source['input'][k] for k in ('input_sha256','prompt_length','target_length','total_length','keep')}
            records[key]['expected_gold']=source.get('gold',source['input'].get('gold'))
            lengths[key]={'total_length':source['input']['total_length'],'prompt_length':source['input']['prompt_length'],
                'target_length':source['input']['target_length'],'source':'Prior actual Qwen3.5 raw-input study; no historical scoring reused.'}
        else:lengths[key]={'total_length':None,'source':'Exact token count is frozen by original tokenizer CPU preparation before any model load; no estimate is asserted as measurement.'}
# Read only the fixed first8 entries of root's actual tokenizer receipt.
preflight=A/'snapshot${ARTIFACT_ROOT}/codex_dt_original_cache_lengths_20260909.json'
pf=json.loads(preflight.read_bytes());assert pf['checkpoint']==base['checkpoint'] and set(pf['first8'])==set(records)
for key,row in pf['first8'].items():
    assert row['source_record_sha256']==records[key]['source_record_sha256']
    expected={k:row[k] for k in ('prompt_length','target_length','total_length')}
    if records[key]['expected_input'] is not None:assert expected=={k:records[key]['expected_input'][k] for k in expected}
    records[key]['expected_lengths']=expected
    lengths[key]=dict(expected,source='Actual root CPU-only tokenizer receipt; first8 only, no sample substitution.')
preflight_info={'file':'${ARTIFACT_ROOT}/codex_dt_original_cache_lengths_20260909.json','sha256':sha(preflight.read_bytes())}
schedule_cases=[[['niah_mq_q2',i],['morehopqa',i]] for i in range(4)]
manifest={'scope':'Fixed eight original developer regression cases, not independent heldout or full benchmark. Run all four predetermined segments; no result-based selection, new generation, replacement, method or weight/layer sweep.',
    'datasets':{'niah_mq_q2':[0,1,2,3],'morehopqa':[0,1,2,3]},'segment_cases':schedule_cases,
    'methods':['DT_control','DT_candidate','FT0','FT3'],
    'method_definitions':{'DT_control':'Unchanged original finiteFA/FLA with norm_gate_rules={0:symmetric}.',
        'DT_candidate':'Same control plus finite_fla_by_layer={0:Layer0FLAEndpointAverage(shared_original_FLA)}.',
        'FT0':'Original e81b3be completeBoth trace(hops3), returned observation_projected.base.',
        'FT3':'Same completeBoth trace, original cumulative base+threehop contributions; author public defaulthops3. FT1/2 retained,not scored.'},
    'fixed_records':records,'length_inventory':lengths,'optional_tokenizer_preflight':preflight_info,
    'global_budget':{'model_loads':4,'native_eager_NI0_initializations':4,'complete_DT_attributions':16,
        'DT_control':8,'DT_candidate':8,'native_DT_roots':16,'native_DT_decoder_replays':512,'finite_decoder_calls':512,
        'finite_GDN_callback_sites':384,'finite_FLA_backend_calls':392,'native_FLA_adjoint_stage_calls':784,
        'finite_FA_calls':128,'public_FA_LSE_calls':128,'complete_original_Both_traces':8,
        'FT_native_roots':8,'FT_native_FLA_forwards':576,'FT_identity_valued_probes_included':192,
        'original_quality_curves':32,'original_B1_scorer_forwards':672,'generation_calls':0,
        'new_methods':0,'new_generated_samples':0,'wall_seconds_per_segment':600},
    'reporting':'Per-case all four methods: NI needle recovery and original RISE/MAS; dataset macro means and candidate deltas versus control,FT0,FT3. Include every regression, all signed vectors,all21curves,case identity,internal target differences,actual costs and incomplete/failure counts. No success-only averaging. A partially run protocol is not an eight-case result.',
    'source_files_sha256':{name:sha(raw) for name,raw in files.items()}}
manifest_raw=json.dumps(manifest,indent=2).encode();manifest_hash=sha(manifest_raw)
mf=A/'dt_fixed_eight_case_regression_manifest_20260909.json';assert not mf.exists(),'Do not overwrite frozen manifest.'
for name,raw in files.items():ast.parse(raw,filename=name)
output=[]
for i,case_indices in enumerate(schedule_cases):
    keys=[f'{d}_{j}' for d,j in case_indices]
    order=['DT_control','DT_candidate'] if i%2==0 else ['DT_candidate','DT_control']
    calls=[[key,method] for key in keys for method in order]
    quality_order=['FT0','DT_control','DT_candidate','FT3'] if i%2==0 else ['FT3','DT_candidate','DT_control','FT0']
    quality=[[key,method] for key in keys for method in quality_order]
    p=copy.deepcopy(base)
    p.update(scope=manifest['scope'],segment_index=i,case_indices=case_indices,quality_cases=keys,call_schedule=calls,quality_schedule=quality,
        fixed_records=records,max_total_tokens=2048,regression_manifest_sha256=manifest_hash,
        method_definitions=manifest['method_definitions'],
        method_target_semantics='For each case all methods share the exact author raw prompt, fixed response+EOS, eligible set and original commonFA evaluator. Preserve original FT sink-representation/recursive-generation targets versus existing DT full-response logprob including EOS;internal seeds differ and neither method is changed.',
        backend_schedule='One eager model load+NI0initialization persegment;FA for4normal DT;each of2unchanged Both traces uses original eager and restoresFA with officialsetter in finally;168actual originalFA scorers follow. No native/framework modifications.',
        FT0_definition=manifest['method_definitions']['FT0'],FT3_definition=manifest['method_definitions']['FT3'],
        budget={'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_attributions':4,
            'DT_control':2,'DT_candidate':2,'DT_B2_roots':4,'native_decoder_replays_DT':128,'finite_decoder_calls':128,
            'public_FA_LSE_calls':32,'finite_FA_calls':32,'finite_GDN_callback_sites':96,'finite_FLA_backend_calls':98,
            'native_FLA_adjoint_stage_calls':196,'candidate_layer0_average_wrapper_calls':2,
            'finite_GDN_conv_preactivation_calls':96,'finite_GDN_conv_autograd_calls':96,
            'complete_official_FT_Both_calls':2,'FT_public_hops':3,'FT_native_B1_roots':2,
            'FT_native_eager_attention_calls':16,'FT_hybrid_linear_layer_inputs':48,'FT_hybrid_full_layer_inputs':16,
            'FT_native_FLA_forwards':144,'FT_identity_valued_FLA_probes_included':48,
            'original_metric_curves':8,'native_B1_scorer_forwards':168,'FT_hops_scored':[0,3],'FT_hops_saved_not_scored':[1,2],
            'generation_calls':0,'new_samples':0,'wall_time_seconds':600},
        evaluation_scope=manifest['reporting'],
        decision='Fixed four segments in order0,1,2,3; do not select later cases from observed improvements or hide regressions. Root schedules sequentially within the frozen total budget. No automatic retries, sample replacement, extrahops, reweighting or new candidate. All8 completion is required for aggregate8claim.',
        stop='First source/input/span/layout/count/nonfinite/original-framework exception or600seconds stops this segment and saves partialwork. Metric regression alone does not select or stop remaining fixed cases. An execution failure is reviewed without altering FT/native or replacing samples; unrun cases remain explicitly pending.',
        cost='Same per-segment resident full model. All4normal DT,2completeBoth traces,168original scorers,endpoint copies,extra finiteFLA,identity-valuedFTprobes,load/init/compile and passive diagnostics included. Descriptive per-case cost only; no matched warmed FT0-only speed claim.',
        files_sha256=manifest['source_files_sha256'])
    p['protected_sources'].append({'path':preflight_info['file'],'sha256':preflight_info['sha256']})
    remote=f'${ARTIFACT_ROOT}/codex_dt_fixed_eight_case_regression_20260909_s{i}_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
    runfiles=dict(files,**{'regression_manifest.json':manifest_raw,'protocol.json':json.dumps(p,indent=2).encode()})
    pp=A/f'dt_fixed_eight_case_regression_protocol_20260909_s{i}.json';lp=A/f'launch_dt_fixed_eight_case_regression_20260909_s{i}.json'
    assert not pp.exists() and not lp.exists(),'Do not overwrite frozen segment.'
    blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in runfiles.items()}).encode())).decode()
    loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
        'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
        'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
        '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
    pp.write_bytes(runfiles['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
    output.append({'segment':i,'cases':keys,'protocol_sha256':sha(runfiles['protocol.json']),'payload':str(lp),'remote_directory':remote})
mf.write_bytes(manifest_raw)
print(json.dumps({'manifest_sha256':manifest_hash,'study_sha256':manifest['source_files_sha256']['study.py'],'segments':output,'global_budget':manifest['global_budget']}))
