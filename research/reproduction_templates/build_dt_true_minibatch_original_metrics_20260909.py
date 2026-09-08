"""Freeze four original B1 curves on saved real-batch/singleton vectors; no launch."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_true_minibatch_NI1_46_20260909_v1'
dp=json.loads((D/'protocol.json').read_bytes());dr=json.loads((D/'results.json').read_bytes())
assert dr['status']=='true_distinct_NI1_46_batch2_9DT225FLA_no_scores_complete' and dr['protocol']==dp
assert dr['sources_before']==dr['sources_after'] and dr['weight_stats_before']==dr['weight_stats_after']
assert dr['DT_entered']==dr['DT_returned']==9 and dr['scorer_entered']==dr['scorer_returned']==0
receipt=json.loads((D/'terminal_receipt.json').read_bytes())
assert any(name in receipt for name in ('pid_alive','proc_exists'))
for name in ('pid_alive','proc_exists'):
    if name in receipt:assert receipt[name] is False
for name,row in receipt['files'].items():
    assert sha((D/name).read_bytes())==(row if isinstance(row,str) else row['sha256']),name
assert sha((D/'vectors.npz').read_bytes())==dr['vectors_sha256']
files={'study.py':(R/'research/reproduction_templates/dt_true_minibatch_original_metrics_20260909.py').read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(D/name).read_bytes();assert sha(raw)==want
    choices=[root/name for root in (R/'core',R/'research/runtime',R/'research/reproduction_templates') if (root/name).is_file()]
    assert len(choices)==1 and choices[0].read_bytes()==raw,name
    files[name]=raw
keys=dp['batch_cases'];assert keys==['niah_mq_q2_1','niah_mq_q2_46']
warm={keys[0]:{'singleton':0,'batch2':2},keys[1]:{'singleton':1,'batch2':2}}
schedule=[[keys[0],'singleton'],[keys[0],'batch2'],[keys[1],'batch2'],[keys[1],'singleton']]
z=np.load(D/'vectors.npz',allow_pickle=False);vector_hashes={k:{} for k in keys}
for key,method in schedule:
    n=warm[key][method];run=dr['runs'][n]
    assert run['number']==n and run['phase']=='warm' and run['status']=='complete'
    assert run['mode']==('batch2' if method=='batch2' else 'single_pair')
    assert run['example_batch_size']==(2 if method=='batch2' else 1)
    s=next(x for x in run['sample_records'] if x['case']==key);name=s['vector_key']
    full=z[name+'_full'];score=z[name+'_evaluated'];f=dr['input_freeze_before_model_load'][key];info=f['input']
    assert full.dtype==np.float64 and score.dtype==np.float32 and full.shape==(1201,) and score.shape==(937,)
    assert np.array_equal(full[:937].astype(np.float32),score) and np.isfinite(full).all()
    vector_hashes[key][method]={'full':sha(full.tobytes()),'evaluated':sha(score.tobytes())}
    ids=np.asarray(f['input_ids'],dtype=np.int64);assert sha(ids.tobytes())==info['input_sha256']
    assert np.array_equal(ids[937:],f['target_ids'])
    order=s['deletion_audit']['sorted_keep'];assert sorted(order)==info['keep'] and np.all(np.diff(score[order].astype(np.float64))<=0)
    count,extra=divmod(len(order),20);cursor=0;deleted=set();changed=ids.copy();groups=[]
    receipts=[{'input_sha256':sha(changed.tobytes()),'deleted_positions':[]}]
    for step in range(20):
        group=order[cursor:cursor+count+(step<extra)];cursor+=len(group);groups.append(group)
        deleted.update(group);changed[group]=f['target_ids'][-1]
        receipts.append({'input_sha256':sha(changed.tobytes()),'deleted_positions':sorted(deleted)})
    assert receipts==s['deletion_audit']['input_receipts'] and groups==s['deletion_audit']['groups']
    assert receipts[-1]['input_sha256']==f['baseline_sha256']
    assert dr['cases'][key]['gold']==f['mapping']['gold']
same=['checkpoint','checkpoint_config_tokenizer_sha256','native_model_sha256','isolated_site','installed_FA_interface_sha256',
    'native_stage_source_sha256','official_root','official_source_blob_sha1','dependency_overlays','runtime_source_sha256',
    'expected_weight_stats','cache_paths','cache_hashes','old_checkpoint','author_data_root','span_source_sha256']
remote='${ARTIFACT_ROOT}/codex_dt_true_minibatch_original_metrics_20260909_v1'
p=copy.deepcopy(dp)
for name in ('group_schedule','group_definition','prior_stability','CPU_length_query','batch_contract','validation','cost',
             'unequal_length_scope','decision','signed_outputs','endpoint_rule','diagnostic_contract_provenance','utility_provenance'):
    p.pop(name,None)
p.update(scope='Required original-metric followup for the actual changed deletion ordering of two distinct NI samples in real minibatch. Exactly saved warm vectors, four original B1 curves; no attribution execution or method change.',
    quality_cases=keys,quality_schedule=schedule,saved_warm_runs=warm,saved_vector_sha256=vector_hashes,
    source_batch={kind+'_path':'/tmp/'+D.name+'/'+name for kind,name in [('results','results.json'),('vectors','vectors.npz'),('protocol','protocol.json')]},
    same_native_identity_keys=same,
    rule_scope='Both saved vectors are the identical current method with layer0 symmetric normgate and complete-endpoint FLA average. Source singleton batchB1/endpointsB2 versus source true samplebatchB2/endpointsB4; no method is rerun here.',
    budget={'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_attributions':0,'finite_FA_calls':0,'finite_FLA_calls':0,
        'original_metric_curves':4,'native_B1_scorer_forwards':84,'FT_calls':0,'generation_calls':0,'new_samples_generated':0,'wall_time_seconds':300},
    evaluation='Unchanged original faithfulness_test_skip_tokens(k20), original compute_logprob_response_given_prompt, and unchanged author needle function. FixedInputMetricView only preserves the raw prompt input already used for attribution. Every full deletion input is checked against that saved vector own21receipts. Original signed density is not clamped/renormalized/replaced by another vector; no cached model score.',
    source_reuse='Every non-study bundled file byte-identical to the completed real-minibatch study; native/official sources and checkpoint identities pinned before/after. Native model untouched, one same NI0 eager initialization then official FA setter.',
    output_scope='All4full original curves and actual native endpoints, exact copied source full/evaluated vectors, per-case batch-minus-singleton RISE/MAS/needle; no new FT comparison or quality claim from needle alone. Both vector sources evaluated by the same original B1 scorer, not batched scoring.',
    decision='Record both case-specific gains/regressions and endpoint drift. This measures batch-induced attribution/ranking quality on the two fixed real samples only; not varlen, larger batch or general-dataset validation.',
    stop='First frozen identity/input/count/metric failure or300seconds; preserve entered/returned native scores and first error. No additional attribution/curve, model precision adjustment, mask substitution, candidate or retry.',
    protected_sources=[{'path':'/tmp/'+D.name+'/'+name,'sha256':sha((D/name).read_bytes())} for name in ('results.json','vectors.npz','protocol.json')],
    files_sha256={name:sha(raw) for name,raw in files.items()})
for kind,name in [('results','results.json'),('vectors','vectors.npz'),('protocol','protocol.json')]:p['source_batch'][kind+'_sha256']=sha((D/name).read_bytes())
for name,raw in files.items():ast.parse(raw,filename=name)
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_true_minibatch_original_metrics_protocol_20260909.json';lp=A/'launch_dt_true_minibatch_original_metrics_20260909.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen protocol or payload.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget']}))
