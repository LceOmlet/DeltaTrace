"""Freeze the9-call distinct-sample batch study; never execute it remotely."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_layer0_fla_endpoint_average_stability_cost_20260909_v1'
dp=json.loads((donor/'protocol.json').read_bytes());dr=json.loads((donor/'results.json').read_bytes())
assert dr['status']=='layer0_FLA_endpoint_average_stability_cost_10DT245FLA84score_complete'
lengthpath=A/'snapshot${ARTIFACT_ROOT}/codex_dt_original_cache_lengths_20260909.json'
lengths=json.loads(lengthpath.read_bytes());assert lengths['checkpoint']==dp['checkpoint']
matched=lengths['equal_prompt_target_pairs']['niah_mq_q2']['(937, 264)']
assert [x['index'] for x in matched]==[1,46]
files={'study.py':(R/'research/reproduction_templates/dt_true_minibatch_NI1_46_20260909.py').read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(donor/name).read_bytes();assert sha(raw)==want
    current=[root/name for root in (R/'core',R/'research/runtime',R/'research/reproduction_templates') if (root/name).is_file()]
    assert len(current)==1 and current[0].read_bytes()==raw,('Changed production source',name)
    files[name]=raw
cache=(A/'snapshot'/dp['cache_paths']['niah_mq_q2'].lstrip('/')).read_bytes()
assert sha(cache)==dp['cache_hashes']['niah_mq_q2']
records={k:copy.deepcopy(dp['fixed_records'][k]) for k in ['niah_mq_q2_0','niah_mq_q2_1']}
line=cache.decode().splitlines()[46];rec=json.loads(line);item=matched[1]
assert sha(line.encode())==item['source_record_sha256']
records['niah_mq_q2_46']={'source_record_sha256':sha(line.encode()),
    'cached_sink_span':rec['sink_span'],'cached_thinking_span':rec['thinking_span'],'metadata_id':rec['metadata'].get('id'),
    'fixed_target_text_sha256':sha(rec['target'].encode()),'prompt_text_sha256':sha(rec['prompt'].encode()),
    'expected_input':None,'expected_gold':None,
    'expected_lengths':{k:item[k] for k in ['total_length','prompt_length','target_length']},
    'provenance':'Distinct author processed record46;CPU-only current tokenizer found the same1201/937/264lengths asNI1. Exact raw ids/EOS baseline and current author gold remapping are frozen before model load, not guessed from old token indices.'}
assert records['niah_mq_q2_1']['source_record_sha256']!=records['niah_mq_q2_46']['source_record_sha256']
remote='${ARTIFACT_ROOT}/codex_dt_true_minibatch_NI1_46_20260909_v1'
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p=copy.deepcopy(dp)
for key in ['prior_wholepilot','NI0_old_input_provenance','quality_cases','call_schedule','quality_schedule',
    'evaluation_scope','evaluation','fixed_control_masks','decision','cost','cost_summary','risks']:
    p.pop(key,None)
p.update({
    'scope':'True two-distinct-example batching of the fixed current DT. NI1 andNI46 share exact full length without padding;current originalFA/FLA,layer0 symmetric normgate and endpoint-order average are unchanged. Shape acceptance alone is not claimed as a successful batch validation.',
    'case_indices':[['niah_mq_q2',0],['niah_mq_q2',1],['niah_mq_q2',46]],
    'batch_cases':['niah_mq_q2_1','niah_mq_q2_46'],'fixed_records':records,
    'group_schedule':[['single_pair','warm'],['batch2','warm'],['single_pair','measured'],
        ['batch2','measured'],['batch2','measured'],['single_pair','measured']],
    'group_definition':'S=NI1 singleton thenNI46 singleton(two attribute calls);B=both distinct samples in one attribute call,actual endpoint order[NI1EOS,NI1input,NI46EOS,NI46input]. FirstS/B three calls simultaneously warm and validate. Only nextS/B/B/S six calls form the cost comparison;each group processes two examples.',
    'prior_stability':{'results_path':'/tmp/'+donor.name+'/results.json','results_sha256':sha((donor/'results.json').read_bytes()),
        'scope':'Source/input identity only;no saved attribution or scorer output replaces this run.'},
    'CPU_length_query':{'path':'${ARTIFACT_ROOT}/codex_dt_original_cache_lengths_20260909.json','sha256':sha(lengthpath.read_bytes()),
        'selected_records':matched,'scope':'CPU tokenizer metadata only;0model/0GPU. NI1+88 and MH15+31 are alternatives outside this frozen budget.'},
    'budget':{'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_invocations':9,
        'singleton_invocations':6,'two_example_invocations':3,'sample_attributions':12,
        'native_roots':9,'physical_endpoint_rows_across_roots':24,'native_decoder_replays':288,'finite_decoder_calls':288,
        'public_FA_LSE_calls':72,'finite_FA_calls':72,'finite_GDN_callback_sites':216,
        'finite_FLA_backend_calls':225,'native_FLA_adjoint_stage_calls':450,
        'endpoint_average_wrapper_calls':9,'finite_GDN_conv_preactivation_calls':216,'finite_GDN_conv_autograd_calls':216,
        'original_quality_curves':0,'scorer_forwards':0,'FT_calls':0,'generation_calls':0,'new_samples_generated':0,
        'wall_time_seconds':600},
    'batch_contract':'No duplicate sample, truncation, reordering within a trajectory or EOS padding. Dense mask is allones for every actual endpoint;B1means one real sample/2endpoints,B2means two real samples/4endpoints. PackedAnswerTargets labels/sample indices and native selected head rows are checked; per-example actual root logprobs are split by original target counts.',
    'validation':'Compare all real singleton and batch vectors to same-run warm singleton per original sample:FP64full/FP32promptrelativeL2,maxabs,rootlogp0/logp1,root-effectdifference,eligible signchanges+mass,rank displacement,21deletion-input identities and unchanged author needle. Report every discrepancy or needle regression;no arbitrary numerical threshold, precision change, bitwise requirement or native repair. No RISE/MAS can be inferred because no scorers run.',
    'cost':'Same model resident;single-pair versus batch2 process the identical two requests. Primary times sum outerattribute durations per group, including native head/checkpoints,endpoint copies,allFA/FLA/conv,six means,and existingstage synchronizations. Request packing,CPU validation,npz persistence are outside that clock but separately included in group/totalwall. Report warm separately, measured means/medians+examples/sec,full allocated/reserved peaks and per-call resident/incremental memory. Two observations/mode;no population,arbitrarylength or largerbatch claims.',
    'source_reuse':f'All{len(files)-1} existing non-study source files are checked against the completed stability study. No production/runtime/native/FT source changes or shadow implementation. New work is only scheduling and passive evidence.',
    'signed_outputs':'Retain every actual fullFP64 and promptFP32 vector, per-sample root target logprob vectors, signed masses, needle and deletion-order receipts. No original quality curve or scorer output is computed.',
    'rule_scope':{'single_pair':'Current symmetric normgate atlayer0 and complete-endpoint finiteFLA average atlayer0;each sample separately.',
        'batch2':'Identical current method, both distinct samples in the same normal runner call. No mathematical rule change.'},
    'unequal_length_scope':'Current dense runner rejects padded masks. Existing native right-padding/FAvarlen forward and finiteFA valid-length ABI are not claimed as fully integrated variable-length finite propagation. This study intentionally needs no such adaptation.',
    'decision':'Root separately authorizes launch. Preserve true batchquality/numeric and cost outcomes;no automatic retry,extra cases,NI1+88,fullcurves,FTor generation. Full methodquality is evaluated by the separate frozen author-case study.',
    'stop':'First source/input/span/layout/count/nonfinite/invariant exception or600s;preserve entered/returned counts and unknown in-flightnative work. Numerical drift itself is reported rather than forcing altered native precision.',
    'protected_sources':[{'path':dp['finite_FA_library'],'sha256':dp['finite_FA_library_sha256']},
        {'path':'/tmp/'+donor.name+'/results.json','sha256':sha((donor/'results.json').read_bytes())},
        {'path':'/tmp/'+donor.name+'/protocol.json','sha256':sha((donor/'protocol.json').read_bytes())},
        {'path':'${ARTIFACT_ROOT}/codex_dt_original_cache_lengths_20260909.json','sha256':sha(lengthpath.read_bytes())}],
    'files_sha256':{name:sha(raw) for name,raw in files.items()}})
p['endpoint_rule']['scope']='Unchanged paired-endpoint average wrapper. This study checks one or two distinct examples, physical2or4endpoint rows. No artificial duplicated sample, padding or model/finite replacement.'
for name,raw in files.items():ast.parse(raw,filename=name)
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_true_minibatch_NI1_46_protocol_20260909.json';lp=A/'launch_dt_true_minibatch_NI1_46_20260909.json'
assert not pp.exists() and not lp.exists(),'Never overwrite a frozen protocol/payload.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget']}))
