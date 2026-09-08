"""Freeze only: same-shape C/P0 costs and distinct NI1+46 batch, 12 DT/0 scores."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda path:json.loads(path.read_bytes())
pilot=A/'snapshot${ARTIFACT_ROOT}/codex_dt_PV_layer19_whole_pilot_20260909_v1'
batch=A/'snapshot${ARTIFACT_ROOT}/codex_dt_true_minibatch_NI1_46_20260909_v1'
dp=read(pilot/'protocol.json');dr=read(pilot/'results.json')
bp=read(batch/'protocol.json');br=read(batch/'results.json')
assert dr['status']=='layer19_PV_content0_current_layer0_4DT100FLA84score_complete'
assert br['status']=='true_distinct_NI1_46_batch2_9DT225FLA_no_scores_complete'
for folder in (pilot,batch):
    receipt=read(folder/'terminal_receipt.json');assert receipt['proc_exists'] is False
    for name in ('results.json','protocol.json','vectors.npz'):
        assert sha((folder/name).read_bytes())==receipt['files'][name]['sha256']
identity=('checkpoint','cache_paths','cache_hashes','checkpoint_config_tokenizer_sha256','native_model_sha256',
    'installed_FA_interface_sha256','native_stage_source_sha256','official_source_blob_sha1','span_source_sha256',
    'expected_weight_stats','finite_FA_library','finite_FA_library_sha256')
for key in identity:assert dp[key]==bp[key],key
files={'study.py':(R/'research/reproduction_templates/dt_PV_layer19_cost_true_batch_20260909.py').read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(pilot/name).read_bytes();assert sha(raw)==want
    paths=[root/name for root in (R/'research/runtime',R/'core',R/'research/reproduction_templates') if (root/name).is_file()]
    assert len(paths)==1 and paths[0].read_bytes()==raw,('Frozen PV runtime changed',name)
    files[name]=raw
for name,raw in files.items():ast.parse(raw,filename=name)
cache=(A/'snapshot'/dp['cache_paths']['niah_mq_q2'].lstrip('/')).read_bytes()
assert sha(cache)==dp['cache_hashes']['niah_mq_q2']
records={}
for index in (0,1,46):
    key=f'niah_mq_q2_{index}';record=copy.deepcopy(bp['fixed_records'][key]);case=br['cases'][key]
    line=cache.decode().splitlines()[index];raw=json.loads(line)
    assert sha(line.encode())==record['source_record_sha256']==case['mapping']['source_record_sha256']
    assert sha(raw['prompt'].encode())==record['prompt_text_sha256']
    assert sha(raw['target'].encode())==record['fixed_target_text_sha256']
    assert raw['sink_span']==record['cached_sink_span'] and raw['thinking_span']==record['cached_thinking_span']
    frozen=br['input_freeze_before_model_load'][key]
    assert frozen['input']==case['input'] and frozen['mapping']['gold']==case['gold']
    ids=np.asarray(frozen['input_ids'],dtype=np.int64);target=np.asarray(frozen['target_ids'],dtype=np.int64)
    assert sha(ids.tobytes())==case['input']['input_sha256']
    assert np.array_equal(ids[case['input']['prompt_length']:],target)
    base=ids.copy();base[case['input']['keep']]=ids[-1]
    assert sha(base.tobytes())==frozen['baseline_sha256']==case['baseline_sha256']
    record.update(expected_input=case['input'],expected_gold=case['gold'],
        expected_lengths={k:case['input'][k] for k in ('prompt_length','target_length','total_length')},
        exact_input_source={'results_path':'/tmp/'+batch.name+'/results.json','results_sha256':sha((batch/'results.json').read_bytes()),
            'input_ids_sha256':sha(ids.tobytes()),'target_ids_sha256':sha(target.tobytes()),'baseline_sha256':sha(base.tobytes())})
    records[key]=record
assert records['niah_mq_q2_1']['expected_input']==dp['fixed_records']['niah_mq_q2_1']['expected_input']
assert records['niah_mq_q2_1']['expected_gold']==dp['fixed_records']['niah_mq_q2_1']['expected_gold']
assert records['niah_mq_q2_1']['source_record_sha256']!=records['niah_mq_q2_46']['source_record_sha256']
assert {records[k]['expected_input']['total_length'] for k in ('niah_mq_q2_1','niah_mq_q2_46')}=={1201}
basekeys=('checkpoint','checkpoint_config_tokenizer_sha256','native_model_sha256','isolated_site','compiler_cache',
    'boundary_compiler_cache','installed_FA_interface_sha256','native_stage_source_sha256','official_root',
    'official_source_blob_sha1','dependency_overlays','runtime_source_sha256','expected_weight_stats','cache_paths',
    'cache_hashes','finite_FA_library','finite_FA_library_sha256','expected_FA_layers','expected_GDN_reverse_order',
    'utility_provenance','old_checkpoint','author_data_root','span_source_sha256','max_total_tokens','input_pipeline')
p={k:copy.deepcopy(dp[k]) for k in basekeys}
p.update({
    'scope':'Prepared necessary warm-cost and true batch interface check for fixed C/P0. No new method, scorer curve, FT, generation, kernel or native change.',
    'case_indices':[['niah_mq_q2',0],['niah_mq_q2',1],['niah_mq_q2',46]],
    'batch_cases':['niah_mq_q2_1','niah_mq_q2_46'],'fixed_records':records,
    'run_schedule':[[mode,method,phase] for mode in ('single_NI1','batch2_NI1_46')
        for phase,methods in [('warm',('control','candidate')),('measured',('control','candidate','candidate','control'))]
        for method in methods],
    'attention_pv_rules':{'control':{},'candidate':{'19':'content0'}},
    'rule_scope':'Both methods use original FA/FLA and current layer0 symmetric normgate plus complete FLA endpoint average. P0 only selects the existing layer19 content0 option; no other layer, weight or default changes.',
    'budget':{'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_invocations':12,
        'warm_DT':4,'measured_DT':8,'control_DT':6,'candidate_DT':6,
        'singleton_invocations':6,'two_example_invocations':6,'sample_attributions':18,
        'native_roots':12,'physical_endpoint_rows_across_roots':36,'native_decoder_replays':384,'finite_decoder_calls':384,
        'public_FA_LSE_calls':96,'finite_FA_calls':96,'finite_FA_phases':288,'content0_FA_calls':6,'content1_FA_calls':90,
        'finite_GDN_callback_sites':288,'finite_FLA_backend_calls':300,'native_FLA_adjoint_stage_calls':600,
        'endpoint_average_wrapper_calls':12,'finite_GDN_conv_preactivation_calls':288,'finite_GDN_conv_autograd_calls':288,
        'original_quality_curves':0,'scorer_forwards':0,'FT_calls':0,'generation_calls':0,'new_FA_extension_compiles':0,
        'new_candidate_kernels':0,'wall_time_seconds':600},
    'batch_contract':'Single_NI1 is one real example and 2 endpoint rows. Batch2_NI1_46 is two distinct existing author records and 4 rows [NI1EOS,NI1input,NI46EOS,NI46input]. No padding or duplicate sample; all masks are ones. Assert native root inputs, packed target sample indices/labels/counts, actual head shape and layer0 pair permutation. Preserve per-example target logp0/logp1 and full signed vectors.',
    'validation':'Report same-shape same-method repeat drift from its own warm reference, C/P0 vector/root/needle/sign/rank differences within each shape, and same-process NI1 singleton/batch differences per method. NI46 has no new singleton; structural row checks and repeats do not establish arbitrary batch isolation or equality. Report all differences without bitexact requirement, new tolerance, precision alteration or retry.',
    'cost':'Each shape independently receives C/P0 warm, then C/P0/P0/C measured. Two measured observations per method; only same-shape same-sample C/P0 cost ratios. Retain outer attribute and native runner timers, actual allocated/reserved peaks, resident and incremental memory for all warm/measured/failed calls, plus model initialization and total wall. Do not compare batch2 with NI1 singleton as matched batch speed; NI1x2 is not the serial NI1+46 workload. No general speed guarantee from two observations.',
    'quality_boundary':'Zero original scorer calls: no new batch RISE/MAS, no reuse of old 4 curves as P0 batch quality. Needle remains original author top-fraction recovery only. This does not replace the separate eight-case development quality evidence or validate unequal-length batches.',
    'source_reuse':f'All {len(files)-1} non-study package files byte-identical to the completed PV19 pilot; scheduling/input/row/cost evidence adapted from completed NI1+46 truebatch study. Native model/scorer/tokenizer/data/weights/FA library identities equal across donors.',
    'source_references':{
        'PV19_pilot':{'results_path':'/tmp/'+pilot.name+'/results.json','results_sha256':sha((pilot/'results.json').read_bytes()),
            'protocol_sha256':sha((pilot/'protocol.json').read_bytes())},
        'original_NI1_46_batch':{'results_path':'/tmp/'+batch.name+'/results.json','results_sha256':sha((batch/'results.json').read_bytes()),
            'protocol_sha256':sha((batch/'protocol.json').read_bytes()),'scope':'Source input/gold/span identity only, no saved output substitutes for any new cost or P0 batch quality.'}},
    'protected_sources':[{'path':dp['finite_FA_library'],'sha256':dp['finite_FA_library_sha256']}]+
        [{'path':'/tmp/'+folder.name+'/'+name,'sha256':sha((folder/name).read_bytes())}
            for folder in (pilot,batch) for name in ('results.json','protocol.json')],
    'decision':'PREPARED ONLY. Root dispatches only after reviewing remaining fixed-case quality. Never automatically launch or retry; no new scoring, FT, samples, generation or method selection.',
    'stop':'First source/input/target/layout/count/nonfinite/invariant failure or 600 seconds. Preserve elapsed/entered/returned/peak memory and explicitly unknown native stages when a backend does not return.',
    'files_sha256':{name:sha(raw) for name,raw in files.items()}})
remote='${ARTIFACT_ROOT}/codex_dt_PV_layer19_cost_true_batch_20260909_v1'
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_PV_layer19_cost_true_batch_protocol_20260909.json';lp=A/'launch_dt_PV_layer19_cost_true_batch_20260909.json'
assert not pp.exists() and not lp.exists(),'Never overwrite frozen protocol/payload.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'status':'prepared_not_launched','protocol_sha256':sha(files['protocol.json']),
    'study_sha256':p['files_sha256']['study.py'],'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget']}))
