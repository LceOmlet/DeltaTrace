"""Freeze existing actual-boundary diagnostic on MH2's already scored C masks."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_PV_layer19_remaining_regression_20260909_s0_v1'
dp=json.loads((donor/'protocol.json').read_bytes());dr=json.loads((donor/'results.json').read_bytes())
receipt=json.loads((donor/'terminal_receipt.json').read_bytes());assert receipt['proc_exists'] is False
for name,row in receipt['files'].items():assert sha((donor/name).read_bytes())==row['sha256']
assert dr['protocol']==dp and dr['status']=='layer19_PV_content0_remaining_segment_4DT100FLA84score_complete'
assert dr['sources_before']==dr['sources_after'] and dr['DT_returned']==4 and dr['scorer_returned']==84
p=copy.deepcopy(json.loads((A/'dt_MH0_current_conditional_boundaries_protocol_20260909.json').read_bytes()))
for name in ('checkpoint','native_model_sha256','runtime_source_sha256','checkpoint_config_tokenizer_sha256','expected_weight_stats','cache_hashes','native_stage_source_sha256','span_source_sha256'):
    assert p[name]==dp[name],name
files={'study.py':(R/'research/reproduction_templates/dt_MH2_current_conditional_boundaries_20260909.py').read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(donor/name).read_bytes();assert sha(raw)==want
    locs=[base/name for base in (R/'core',R/'research/runtime',R/'research/reproduction_templates') if (base/name).is_file()]
    assert len(locs)==1 and locs[0].read_bytes()==raw,name
    files[name]=raw
key='morehopqa_2';method='control';run_index=3;case=dr['cases'][key];f=dr['input_freeze_before_model_load'][key]
assert dr['runs'][run_index]['case']==key and dr['runs'][run_index]['method']==method
assert case['input']['total_length']==710 and case['input']['prompt_length']==480 and not case['gold']
ids=np.asarray(f['input_ids'],dtype=np.int64);assert sha(ids.tobytes())==case['input']['input_sha256']
z=np.load(donor/'vectors.npz',allow_pickle=False);w=z[key+'_'+method+'_evaluated'];full=z[key+'_'+method+'_full']
assert np.array_equal(full[:len(w)].astype(np.float32),w) and w.dtype==np.float32
curve=case['curves'][method];order=curve['sorted_keep'];assert sorted(order)==case['input']['keep']
assert np.all(np.diff(w[order].astype(np.float64))<=0)
assert curve['input_receipts']==dr['runs'][run_index]['deletion_audit']['input_receipts']
n,extra=divmod(len(order),20);offset=0;deleted=set();x=ids.copy();receipts=[{'input_sha256':sha(x.tobytes()),'deleted_positions':[]}]
for i in range(20):
    group=order[offset:offset+n+(i<extra)];offset+=len(group);deleted.update(group);x[group]=ids[-1]
    receipts.append({'input_sha256':sha(x.tobytes()),'deleted_positions':sorted(deleted)})
assert receipts==curve['input_receipts'] and receipts[-1]['input_sha256']==f['baseline_sha256']
p['case_indices']=dp['case_indices'];p['fixed_records']=copy.deepcopy(dp['fixed_records'])
p['source_method']=method;p['source_run_index']=run_index;p['call_schedule']=[[key,'DT']]
p['scope']='One current C MH2 attribution and four original B1 scores on its already measured control masks. Existing actual-boundary diagnostic and default FA/FLA; no candidate or complete new metric curve.'
p['source_standalone']={'results_path':'/tmp/'+donor.name+'/results.json','results_sha256':sha((donor/'results.json').read_bytes()),
    'vectors_path':'/tmp/'+donor.name+'/vectors.npz','vectors_sha256':sha((donor/'vectors.npz').read_bytes()),
    'protocol_path':'/tmp/'+donor.name+'/protocol.json','protocol_sha256':sha((donor/'protocol.json').read_bytes()),
    'study_sha256':dp['files_sha256']['study.py'],'vector_key':key+'_'+method,
    'scope':'Historical field name retained for the common observer. Source is explicitly the current C method of completed PV fixed-eight segment0, not a standalone FT or altered source object.'}
p['frozen_capture_receipts']={str(s):receipts[s] for s in p['capture_steps']}
p['frozen_source_scores']={str(s):curve['scores'][s] for s in p['capture_steps']}
p['source_reuse']='All18 non-study source files match the completed PV segment and current local executable sources. Study changes only MH0 toMH2 and explicit source method/run selectors; native/finite kernels and observer contractions unchanged.'
p['budget']['wall_time_seconds']=600
p['decision']='Locate largest actual MH2 conditional jumps with signed compensation preserved, after failed supported-softmax pilot. OneDT/four original score points is diagnosis, not new MAS or validation of a repair. Freeze a minimal operator modification only after dominant terms are observed.'
p['outputs']=p['outputs'].replace('MH0','MH2')
p['stop']=p['stop'].replace('900','600')
p['protected_sources']=[{'path':p['finite_FA_library'],'sha256':p['finite_FA_library_sha256']}]+[{'path':'/tmp/'+donor.name+'/'+n,'sha256':sha((donor/n).read_bytes())} for n in ('results.json','vectors.npz','protocol.json')]
p['files_sha256']={n:sha(raw) for n,raw in files.items()}
for n,raw in files.items():ast.parse(raw,filename=n)
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_MH2_current_conditional_boundaries_protocol_20260909.json';lp=A/'launch_dt_MH2_current_conditional_boundaries_20260909.json';assert not pp.exists() and not lp.exists()
remote='${ARTIFACT_ROOT}/codex_dt_MH2_current_conditional_boundaries_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'runtime_sources':len(files)-2,'budget':p['budget']}))
