"""Freeze the remaining six original cases as three independently dispatched jobs."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
pilot=A/'snapshot${ARTIFACT_ROOT}/codex_dt_GDN1_K_whole_pilot_20260909_v1';dp=json.loads((pilot/'protocol.json').read_bytes())
aud=json.loads((A/'dt_GDN1_K_whole_pilot_summary_20260909.json').read_bytes());assert aud['status']=='independent_GDN1_K_whole_pilot_audit_passed' and aud['results_sha256']==sha((pilot/'results.json').read_bytes())
pairs=[['niah_mq_q2_2','morehopqa_1'],['niah_mq_q2_1','morehopqa_0'],['niah_mq_q2_3','morehopqa_3']]
text=(pilot/'study.py').read_text();text=text.replace("remaining_pairs=[['niah_mq_q2_0','morehopqa_2']]",'remaining_pairs='+repr(pairs))
text=text.replace("type(p['segment_index']) is int and p['segment_index']==0","type(p['segment_index']) is int and 0<=p['segment_index']<3")
text=text.replace('Only input-supported GDN1 K NI0/MH2 pilot','Only the selected fixed two-case input-supported GDN1 K segment')
study=R/'research/reproduction_templates/dt_GDN1_K_remaining_20260909.py';assert not study.exists();study.write_text(text);ast.parse(text)
common={'study.py':study.read_bytes()}
for name,h in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(pilot/name).read_bytes();assert sha(raw)==h;common[name]=raw
records={};refs={};sources=[]
for index in range(4):
    olddir=A/f'snapshot${ARTIFACT_ROOT}/codex_dt_fixed_eight_case_regression_20260909_s{index}_v1'
    op=json.loads((olddir/'protocol.json').read_bytes());orr=json.loads((olddir/'results.json').read_bytes());rec=json.loads((olddir/'terminal_receipt.json').read_bytes())
    assert rec['proc_exists'] is False and sha((olddir/'results.json').read_bytes())==rec['files']['results.json']['sha256']
    for k in ['checkpoint','cache_paths','cache_hashes','checkpoint_config_tokenizer_sha256','native_model_sha256','official_source_blob_sha1','span_source_sha256','finite_FA_library','finite_FA_library_sha256']:assert op[k]==dp[k]
    source={'results_path':'/tmp/'+olddir.name+'/results.json','results_sha256':sha((olddir/'results.json').read_bytes()),'protocol_path':'/tmp/'+olddir.name+'/protocol.json','protocol_sha256':sha((olddir/'protocol.json').read_bytes())};sources.append(source)
    for dataset in ['niah_mq_q2','morehopqa']:
        key=f'{dataset}_{index}';case=orr['cases'][key];record=copy.deepcopy(op['fixed_records'][key]);record.update(expected_input=case['input'],expected_gold=case['gold']);records[key]=record
        refs[key]=dict(source,input_sha256=case['input']['input_sha256'],source_record_sha256=record['source_record_sha256'],gold=case['gold'],FT_commit=op['FT_commit'],curves={method:{'original_return_metrics':case['curves'][method]['return_metrics'],'needle':case['curves'][method]['needle']} for method in ['FT0','FT3']},scope='Historical unchanged FT, not contemporaneous controls. Same input/source/gold; do not substitute raw curves for newly scored DT.',internal_target_semantics=op['method_target_semantics'])
prepared=[];python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
for index,pair in enumerate(pairs):
    p=copy.deepcopy(dp);keys=list(dict.fromkeys(['niah_mq_q2_0',*pair]));first,second=pair;schedule=[[first,'control'],[first,'candidate'],[second,'candidate'],[second,'control']]
    protected=[row for row in p['protected_sources'] if 'fixed_eight_case_regression' not in row['path']]
    for ix in sorted({int(k.rsplit('_',1)[1]) for k in keys}):
        source=sources[ix];protected.extend([{'path':source['results_path'],'sha256':source['results_sha256']},{'path':source['protocol_path'],'sha256':source['protocol_sha256']}])
    p.update(segment_index=index,quality_cases=pair,case_indices=[[k.rsplit('_',1)[0],int(k.rsplit('_',1)[1])] for k in keys],quality_schedule=schedule,call_schedule=[row+['quality'] for row in schedule],fixed_records={k:records[k] for k in keys},historical_FT_references={k:refs[k] for k in pair},protected_sources=protected,
        fixed_segment_plan=pairs,executed_subset=pair,scope='One segment of fixed remaining six original NI/MH development cases for unchanged input-supported GDN1 K candidate. Same native model, current C, original scorer and helper; no automatic next-job dispatch.',coverage='Only the dispatched segment plus previously completed original NI0/MH2 pilot. Other frozen segments remain unexecuted until separately dispatched; no heldout/true batch claim.',
        completed_pilot={'path':'/tmp/'+pilot.name+'/results.json','sha256':sha((pilot/'results.json').read_bytes())},
        decision='Finish its two cases unless invariant/timeout failure. Stop later expansion on materially adverse quality evidence; preserve all needle/RISE/MAS regressions. All fixed cases reported as complete or unexecuted, no favorable-case substitution or rule tuning.',files_sha256={name:sha(raw) for name,raw in common.items()})
    remote=f'${ARTIFACT_ROOT}/codex_dt_GDN1_K_remaining_20260909_s{index}_v1';files=dict(common);files['protocol.json']=json.dumps(p,indent=2).encode()
    pp=A/f'dt_GDN1_K_remaining_protocol_20260909_s{index}.json';lp=A/f'launch_dt_GDN1_K_remaining_20260909_s{index}.json';assert not pp.exists() and not lp.exists()
    blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
    loader='import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))'
    pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -B -c '+shlex.quote(loader),'timeout':10}));prepared.append({'index':index,'cases':pair,'protocol_sha256':sha(files['protocol.json']),'status':'prepared_not_launched','name':Path(remote).name.removeprefix('codex_')})
manifest=A/'dt_GDN1_K_remaining_prepared_20260909.json';assert not manifest.exists();manifest.write_text(json.dumps({'status':'prepared_not_launched','segments':prepared,'all_nonstudy_files_identical_to_pilot':True},indent=2))
print(json.dumps(prepared))
