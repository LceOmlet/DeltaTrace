"""Independently review true example batching and original coroutine scoring."""
import ast,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_batch_20260907_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete' and d['all_native_vectors_frozen_before_any_quality']
assert p==json.loads((A/'vendor_fa_batch_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'vendor_fa_batch_probe_20260907.py')
for name,h in p['sources'].items():assert sha(F/name)==sha(A/name)==h
parent_file=A/'snapshot'/p['required_parent'].lstrip('/');assert sha(parent_file)==p['required_parent_sha256']
parent=json.loads(parent_file.read_text())
assert d['checkpoint_before']==d['checkpoint_after']==parent['checkpoint_before']==parent['checkpoint_after']
assert d['native_sources_before']==d['native_sources_after']==parent['native_sources_before']==parent['native_sources_after']
for key,value in p['budget'].items():assert d[key]==value,(key,d[key],value)
assert len(d['native_attempt_costs'])==d['manual_attempts']==29 and all(x['completed'] for x in d['native_attempt_costs'])
counts={'finite':16,'batch2':8,'batch4':5}
assert {mode:sum(x['capture_mode']==mode for x in d['native_attempt_costs']) for mode in counts}==counts
for item in d['native_attempt_costs']:
    batch={'finite':1,'batch2':2,'batch4':4}[item['capture_mode']];c=item['cost']
    assert c['native_forwards']==1 and c['native_forward_trajectories']==batch*2 and c['vjps']==0
    assert c['extra_replay_calls']==36 and c['extra_replay_trajectories']==72*batch
    assert c['native_decoder_layer_calls']==72 and c['native_decoder_layer_trajectories']==144*batch
    assert c['public_FA_activity']['auxiliary_completed']==c['public_FA_activity']['auxiliary_attempts']==36
    assert len(c['finite_FA_activity'])==36
    for call in c['finite_FA_activity']:
        assert call['calls_attempted']==call['calls_enqueued']==1
        for buf in call['buffer_contract']:assert buf['shape'][0]==batch and buf['shape'][1]==32 and len(buf['shape']) in [3,4]
assert sum(x['cost']['native_forward_trajectories'] for x in d['native_attempt_costs'])==104
assert sum(len(x['cost']['finite_FA_activity']) for x in d['native_attempt_costs'])==1044
out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'protocol_sha256':sha(F/'protocol.json'),
 'scope':p['purpose'],'budget':p['budget'],'costs':{},'cases':[],'curves_verified':0,'max_metric_reconstruction_error':0.}
rows=d['records'];assert [(r['dataset'],r['idx']) for r in rows]==[tuple(x) for x in p['selection']]
for row in rows:
    old=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(row['dataset'],row['idx']))
    for key in ['input_ids','input_ids_sha256','prompt_len','user_positions','keep_local_indices','eligible_positions','gold_full_local','gold_eligible_local']:assert row[key]==old[key]
    assert hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()==row['input_ids_sha256']
    assert len(row['single_runs'])==4
    assert [r['repeat'] for r in row['single_runs']]==list(range(4))
assert len(d['batch_runs'])==12
for mode,batch in [('batch2',2),('batch4',4)]:
    for repeat in range(4):
        runs=[r for r in d['batch_runs'] if r['repeat']==repeat and r['mode']==mode]
        assert len(runs)==4//batch and sorted(i for r in runs for i in r['indices'])==list(range(4))
        assert all(len(r['indices'])==batch for r in runs)
for mode in ['single','batch2','batch4']:
    runs=[r for row in rows for r in row['single_runs']] if mode=='single' else [r for r in d['batch_runs'] if r['mode']==mode]
    costs=[sum(r['result']['seconds'] for r in runs if r['repeat']==repeat) for repeat in range(4)]
    out['costs'][mode]={'seconds_for_all_four_by_repeat':costs,'median_seconds_for_all_four':statistics.median(costs[1:]),
        'seconds_per_example':statistics.median(costs[1:])/4,
        'peak_bytes':max(r['result']['peak_allocated_bytes'] for r in runs if not r['warmup']),
        'physical_calls_per_measured_repeat':4 if mode=='single' else 2 if mode=='batch2' else 1}
for mode in ['batch2','batch4']:out['costs'][mode]['throughput_speedup_vs_four_singles']=out['costs']['single']['median_seconds_for_all_four']/out['costs'][mode]['median_seconds_for_all_four']
def check_value(value,indices):
    batch=len(indices);n=max(len(rows[i]['input_ids']) for i in indices)
    assert value['example_batch_size']==batch and value['physical_endpoint_batch_size']==2*batch and value['padded_length']==n
    assert value['actual_lengths']==[len(rows[i]['input_ids']) for i in indices]
    vectors=np.array(value['signed_full_sequence']);assert vectors.shape==(batch,n) and np.isfinite(vectors).all()
    assert len(value['finite_attention_activity'])==len(value['public_FA_capture_checks'])==len(value['native_layer_boundary_checks'])==36
    for c in value['public_FA_capture_checks']:
        assert c['auxiliary_batch_size']==batch*2 and c['public_output_exact_to_actual_model_FA']
        assert c['testing_matrix_numel']==0 and not c['private_FA_slots_read'] and not c['model_output_replaced']
    for c in value['native_layer_boundary_checks']:assert c['native_input_exact'] and c['native_output_exact']
    for sample,i in enumerate(indices):
        allowed=set(rows[i]['eligible_positions']);x=vectors[sample]
        assert all(v==0 for j,v in enumerate(x) if j not in allowed)
        assert abs(x.sum()-value['signed_sum'][sample])<1e-9
        assert abs(value['target_delta_score32_sum64'][sample]-value['signed_sum'][sample]-value['unassigned_total'][sample])<1e-9
    return vectors
for run in d['batch_runs']:check_value(run['result'],run['indices'])
profile=d['batch_profiles']['batch4'];check_value(profile['result'],list(range(4)))
path=F/profile['trace'];assert sha(path)==profile['sha256'];events=json.loads(path.read_text())['traceEvents']
names=[e['name'] for e in events if e.get('cat')=='kernel']
assert sum('flash_fwd_kernel' in n for n in names)==profile['native_FA_forward_kernels']==108
assert sum('deltatrace_fa_finite_p1_kernel' in n for n in names)==profile['finite_FA_kernels']==108
assert not any('flash_bwd' in n for n in names)
out['profile']={'actual_default_FA_forwards':108,'actual_finite_kernels':108,'physical_endpoint_batch':8,'example_batch':4}
del events,names
# Independently reconstruct original metrics, only remove old B1 cost assertion.
helper=ast.parse((A/'verify_native_output_contrast_development16_20260906.py').read_text())
functions=[n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name in ['area','metrics']]
fn=next(n for n in functions if n.name=='metrics');assert isinstance(fn.body[-2],ast.Assign) and isinstance(fn.body[-1],ast.Assert)
fn.body=fn.body[:-2]
ns={'np':np,'math':math,'out':out};exec(compile(ast.Module(body=functions,type_ignores=[]),'independent_original_metrics','exec'),ns)
activity=d['original_batched_evaluation_activity'];cost=d['batched_evaluation_cost']
assert activity['physical_evaluation_forwards']==cost['native_forwards']==84
assert activity['evaluation_trajectories']==cost['native_forward_trajectories']==336
assert len(activity['actual_batch_sizes'])==84 and set(activity['actual_batch_sizes'])=={4}
assert cost['native_decoder_layer_calls']==84*36 and cost['native_decoder_layer_trajectories']==336*36
assert activity['source_proof']['restored_AST_identical'] and not activity['source_proof']['placeholder_scores']
source_path=A/'snapshot${FLASHTRACE_ROOT}/ft_ifr_improve.py'
source=source_path.read_text();assert hashlib.sha256(source.encode()).hexdigest()==p['official_normalized_sources']['ft_ifr_improve.py']
function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='faithfulness_test_skip_tokens')
original_source=''.join(source.splitlines(keepends=True)[function.lineno-1:function.end_lineno])
assert hashlib.sha256(original_source.encode()).hexdigest()==activity['source_proof']['original_function_source_sha256']
assert activity['source_proof']['suspended_native_calls_in_source']==2
out['original_batched_evaluation']={'physical_forwards':84,'trajectories':336,'batch_sizes':[4],
 'original_function_source_sha256':activity['source_proof']['original_function_source_sha256'],
 'original_metric_and_deletion_AST_preserved':True,'placeholders_used':False,'seconds':cost['seconds'],'peak_bytes':cost['peak_allocated_bytes']}
for i,row in enumerate(rows):
    baseline=next(x['result'] for x in row['single_runs'] if x['repeat']==1);ref=np.array(baseline['signed_full_sequence']);eligible=row['eligible_positions']
    assert np.isfinite(ref).all() and all(v==0 for j,v in enumerate(ref) if j not in set(eligible))
    case={'dataset':row['dataset'],'idx':row['idx'],'N':len(row['input_ids']),'batch_numerics':{},'metrics':{}}
    vectors={'single':ref}
    for mode in ['batch2','batch4']:
        run=next(r for r in d['batch_runs'] if r['repeat']==1 and r['mode']==mode and i in r['indices'])
        sample=run['indices'].index(i);value=run['result'];x=np.asarray(value['signed_full_sequence'][sample][:len(ref)]);vectors[mode]=x
        flips=np.sign(x[eligible])!=np.sign(ref[eligible])
        case['batch_numerics'][mode]={'relative_l2_to_single':float(np.linalg.norm(x-ref)/max(np.linalg.norm(ref),1e-30)),
            'maximum_absolute':float(abs(x-ref).max()),'eligible_sign_changes':int(flips.sum()),
            'single_absolute_credit_mass_on_sign_changes':float(abs(ref[eligible][flips]).sum()/max(abs(ref[eligible]).sum(),1e-30)),
            'native_endpoint_score_differences':{side:value['endpoint_scores32'][side][sample]-baseline['endpoint_scores32'][side] for side in ['before','after']},
            'relative_unassigned':abs(value['unassigned_total'][sample])/max(1,abs(value['target_delta_score32_sum64'][sample]))}
    for mode,x in vectors.items():assert np.array_equal(np.maximum(x[row['user_positions']],0).astype(np.float32),np.array(row['scores'][mode],dtype=np.float32))
    endpoints=[row['metrics']['single']['raw_curve'][0],row['metrics']['single']['raw_curve'][-1]]
    metric_row=dict(row);metric_row['common_eager_evaluation_endpoints16']=endpoints
    old=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(row['dataset'],row['idx']))
    ft=row['historical_FT_source']['name'];assert row['scores']['historical_FT']==old['scores'][ft]
    assert row['historical_FT_source']['parent_sha256']==p['required_parent_sha256'] and not row['historical_FT_source']['fresh_attribution']
    for mode in ['single','batch2','batch4','historical_FT']:
        assert row['metrics'][mode]['raw_curve']==activity['curves'][str(i)+'_'+mode]
        # A B4 native evaluator may differ across rows even for identical
        # endpoint inputs. Preserve each actually returned endpoint instead of
        # substituting the single-mode curve's endpoints. Do not repair curves.
        metric_row['common_eager_evaluation_endpoints16']=[row['metrics'][mode]['raw_curve'][0],row['metrics'][mode]['raw_curve'][-1]]
        ns['metrics'](metric_row,mode)
        case['metrics'][mode]={k:row['metrics'][mode][k] for k in ['rise','mas','recovery']}
    case['B4_minus_historical_B1_evaluation_endpoints']=(np.array(endpoints)-np.array(old['common_eager_evaluation_endpoints16'])).tolist()
    case['evaluation_endpoints_by_mode']={mode:[m['raw_curve'][0],m['raw_curve'][-1]] for mode,m in row['metrics'].items()}
    out['cases'].append(case)
assert out['curves_verified']==16
diagnostic=json.loads((A/'batch_endpoint_diagnostic_summary_20260907.json').read_text())
assert diagnostic['status']=='verified_complete' and all(r['directly_matches_original_batch_run'] for r in diagnostic['cases'])
diagnostic_protocol=json.loads((A/'batch_endpoint_diagnostic_protocol_20260907.json').read_text())
assert diagnostic_protocol['batch_parent_sha256']==out['raw_sha256']
out['endpoint_identity_review']={'initial_cross_mode_bitwise_guard':'failed_and_preserved',
    'failure_receipt':'vendor_fa_batch_first_verification_failure_20260907.json',
    'direct_native_diagnostic_raw_sha256':diagnostic['raw_sha256'],
    'metric_reconstruction':'Each unchanged original curve uses its actual native endpoints; no common-endpoint substitution or curve edits.',
    'underlying_backend_arithmetic_cause_identified':False}
out['limits']='Four original development examples, original lengths367-607, actual B1/B2/B4. No arbitrary batch or long-input extrapolation, independent-quality or token-causal-sign confirmation.'
(A/'vendor_fa_batch_summary_20260907.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
numeric={'raw_sha256':out['raw_sha256'],'scope':'Original numeric scores/vectors/curves/masks and full actual costs; no original text/token IDs or activations.',
 'records':[{k:v for k,v in row.items() if k!='input_ids'} for row in rows],
 'batch_runs':d['batch_runs'],'batch_profiles':d['batch_profiles'],'batched_evaluation_activity':activity,'batched_evaluation_cost':cost}
(A/'vendor_fa_batch_numeric_20260907.json').write_text(json.dumps(numeric,separators=(',',':')),encoding='utf-8')
print(json.dumps(out,indent=2))
