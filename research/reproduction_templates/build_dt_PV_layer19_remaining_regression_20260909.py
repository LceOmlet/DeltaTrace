"""Freeze three separately dispatchable segments with one identical study."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_PV_layer19_whole_pilot_20260909_v1'
dp=json.loads((donor/'protocol.json').read_bytes());prior=json.loads((donor/'results.json').read_bytes())
assert prior['status']=='layer19_PV_content0_current_layer0_4DT100FLA84score_complete'
assert prior['DT_returned']==4 and prior['scorer_returned']==84
assert sha((donor/'results.json').read_bytes())==json.loads((donor/'terminal_receipt.json').read_bytes())['files']['results.json']['sha256']
common={'study.py':(R/'research/reproduction_templates/dt_PV_layer19_remaining_regression_20260909.py').read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(donor/name).read_bytes();assert sha(raw)==want
    matches=[root/name for root in (R/'research/runtime',R/'core',R/'research/reproduction_templates') if (root/name).is_file()]
    assert len(matches)==1 and matches[0].read_bytes()==raw,('Successful runtime/source changed',name)
    common[name]=raw
for name,raw in common.items():ast.parse(raw,filename=name)
old=(donor/'study.py').read_text(encoding='utf-8');new=common['study.py'].decode()
begin='    # Original scorer forwards, with own actual vectors and masks for every curve.'
end="    r['finite_counts']={method:{'entered':op.entered,'returned':op.returned} for method,op in fas.items()}"
assert old[old.index(begin):old.index(end)]==new[new.index(begin):new.index(end)]
records={};refs={};old_jobs={}
for index in range(4):
    folder=A/f'snapshot${ARTIFACT_ROOT}/codex_dt_fixed_eight_case_regression_20260909_s{index}_v1'
    op=json.loads((folder/'protocol.json').read_bytes());result=json.loads((folder/'results.json').read_bytes())
    assert result['scorer_returned']==168 and result['DT_returned']==4
    receipt=json.loads((folder/'terminal_receipt.json').read_bytes())
    assert sha((folder/'results.json').read_bytes())==receipt['files']['results.json']['sha256']
    for key in ('checkpoint','cache_paths','cache_hashes','checkpoint_config_tokenizer_sha256','native_model_sha256','official_source_blob_sha1','span_source_sha256','finite_FA_library','finite_FA_library_sha256'):
        assert dp[key]==op[key],key
    source={'results_path':'/tmp/'+folder.name+'/results.json','results_sha256':sha((folder/'results.json').read_bytes()),
            'protocol_path':'/tmp/'+folder.name+'/protocol.json','protocol_sha256':sha((folder/'protocol.json').read_bytes())}
    old_jobs[index]=source
    for dataset in ('niah_mq_q2','morehopqa'):
        key=f'{dataset}_{index}';case=result['cases'][key]
        rec=copy.deepcopy(op['fixed_records'][key])
        rec.update(expected_input=case['input'],expected_gold=case['gold'])
        raw=(A/'snapshot'/Path(dp['cache_paths'][dataset].lstrip('/'))).read_bytes()
        assert sha(raw)==dp['cache_hashes'][dataset]
        assert sha(raw.decode().splitlines()[index].encode())==rec['source_record_sha256']
        assert {k:case['input'][k] for k in ('prompt_length','target_length','total_length')}==rec['expected_lengths']
        records[key]=rec
        refs[key]=dict(source,input_sha256=case['input']['input_sha256'],source_record_sha256=rec['source_record_sha256'],
            gold=case['gold'],FT_commit=op['FT_commit'],
            curves={method:{'original_return_metrics':case['curves'][method]['return_metrics'],'needle':case['curves'][method]['needle']} for method in ('FT0','FT3')},
            scope='Historical original e81 FT0/FT3 from the completed eight-case study, different process. Input/source/gold identity checked; no contemporaneous FT execution, no historical raw score substitution for current DT curves, and no claim of absent process drift.',
            internal_target_semantics=op['method_target_semantics'])
segments=[['niah_mq_q2_0','morehopqa_2'],['niah_mq_q2_2','morehopqa_1'],['niah_mq_q2_3','morehopqa_3']]
assert len(set(sum(segments,[])))==6
assert set(sum(segments,[]))|{'morehopqa_0','niah_mq_q2_1'}==set(records)
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';prepared=[]
for segment,pair in enumerate(segments):
    p=copy.deepcopy(dp);first,second=pair
    case_keys=list(dict.fromkeys(['niah_mq_q2_0',first,second]))
    quality=[[first,'control'],[first,'candidate'],[second,'candidate'],[second,'control']]
    protected=[row for row in dp['protected_sources'] if 'fixed_eight_case_regression' not in row['path']]
    protected += [{'path':'/tmp/'+donor.name+'/'+name,'sha256':sha((donor/name).read_bytes())} for name in ('results.json','protocol.json')]
    for index in sorted({int(key.rsplit('_',1)[1]) for key in case_keys}):
        source=old_jobs[index]
        protected += [{'path':source['results_path'],'sha256':source['results_sha256']},
                      {'path':source['protocol_path'],'sha256':source['protocol_sha256']}]
    p.update({
      'scope':'One fixed two-case segment of the remaining six original author cases for the exact completed layer19 P0 development candidate. Identical method/profile/math/library/scorer; only case/segment metadata changes. No new FT, generation or compile.',
      'segment_index':segment,'fixed_segment_plan':segments,'case_indices':[[key.rsplit('_',1)[0],int(key.rsplit('_',1)[1])] for key in case_keys],
      'quality_cases':pair,'quality_schedule':quality,'call_schedule':[row+['quality'] for row in quality],
      'fixed_records':{key:records[key] for key in case_keys},
      'historical_FT_references':{key:refs[key] for key in pair},
      'protected_sources':protected,
      'completed_initial_two_case_pilot':{'results_path':'/tmp/'+donor.name+'/results.json','results_sha256':sha((donor/'results.json').read_bytes()),
         'cases':['morehopqa_0','niah_mq_q2_1'],'scope':'Completed initial two cases only. Their measured outcomes and tradeoffs are not replaced or silently rescored by the remaining-case segments.'},
      'selection_scope':'Layer19 was selected from actual current MH0 evidence. All eight original cases have been used in development, including these six remaining candidate regressions; this is not heldout confirmation. Fixed remaining order is NI0/MH2, NI2/MH1, NI3/MH3 with no data generation or case substitution.',
      'source_reuse':'All18 non-study package files exactly match the completed two-case PV19 pilot. The single shared study only parameterizes fixed segment/case validation and attaches explicit historical FT reference metadata. The entire original21-point scorer and same-control-mask ledger source block is byte-identical to the completed pilot.',
      'cost':'Per segment one model load, one fixed NI0 eager initialization, four full DT calls in first-case C->P0 and second-case P0->C order,84fresh original B1 scores. Same current profiles and original finite FA/FLA libraries,100FLA32FA32LSE128replays. Single observation per method/case only; no repeated speed benchmark.',
      'evaluation_scope':'Each method uses its new full signed vector and own fresh original21-point curve. Fixed control masks use this same segment actual control scores. Historical FT0/FT3 references retain their original process and source labels, never substitute for current scoring or simulate same-run FT. Preserve NI needle and all per-case RISE/MAS gains/regressions.',
      'launch_prerequisite':'PREPARED ONLY, separate payload per segment. Root reviews the first segment actual method benefit/harm before deciding whether to dispatch another. A payload starts only its own segment; it never schedules or launches the next. No claim that all8 candidate cases are complete while remaining segments are unexecuted.',
      'decision':'Within a started segment complete the fixed two cases unless an actual invariant/timeout fails; retain all unfavorable outcomes. Root alone decides stopping subsequent segments for materially adverse method evidence. No posthoc favorable-case selection, layer/weight adjustment, combined softmax change, extra sample, or automatic retry. Final reporting distinguishes completed from unexecuted planned cases.',
      'files_sha256':{name:sha(raw) for name,raw in common.items()}})
    remote=f'${ARTIFACT_ROOT}/codex_dt_PV_layer19_remaining_regression_20260909_s{segment}_v1'
    files=dict(common);files['protocol.json']=json.dumps(p,indent=2).encode()
    pp=A/f'dt_PV_layer19_remaining_regression_protocol_20260909_s{segment}.json'
    lp=A/f'launch_dt_PV_layer19_remaining_regression_20260909_s{segment}.json'
    assert not pp.exists() and not lp.exists(),'Never overwrite frozen payload.'
    blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
    loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
      'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
      'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
      '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
    pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
    prepared.append({'segment_index':segment,'cases':pair,'lengths':[records[key]['expected_input']['total_length'] for key in pair],
        'payload':str(lp),'protocol':str(pp),'protocol_sha256':sha(files['protocol.json']),'remote_directory':remote})
summary={'status':'three_payloads_prepared_not_launched','study_sha256':sha(common['study.py']),'segments':prepared,
   'total_budget_if_all_three_dispatched':{'model_loads':3,'NI0_eager_initializations':3,'DT':12,'finite_FLA':300,
       'native_FLA_adjoint_stages':600,'finite_FA':96,'public_FA_LSE':96,'native_decoder_replays':384,'original_scorers':252,'FT':0,'generation':0,'new_FA_extension_compile':0},
   'same_original_metric_source_block':True,'all_nonstudy_files_same_as_completed_pilot':True}
manifest=A/'dt_PV_layer19_remaining_regression_prepared_20260909.json'
assert not manifest.exists();manifest.write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
