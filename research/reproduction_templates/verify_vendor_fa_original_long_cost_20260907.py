"""Independent long-input source, full-vector, memory, cost and trace review."""
import ast,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_original_long_cost_20260907_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete' and d['quality_evaluated'] is False
assert p==json.loads((A/'vendor_fa_original_long_cost_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'vendor_fa_original_long_cost_20260907.py')
for name,h in p['sources'].items():assert sha(F/name)==sha(A/name)==h
parent_path=A/'snapshot'/p['required_parent'].lstrip('/');assert sha(parent_path)==p['required_parent_sha256']
parent=json.loads(parent_path.read_text())
assert d['checkpoint_before']==d['checkpoint_after']==parent['checkpoint_before']==parent['checkpoint_after']
assert d['native_sources_before']==d['native_sources_after']==parent['native_sources_before']==parent['native_sources_after']
assert d['dtype']==parent['dtype']=='torch.float16'
cap_path=A/'snapshot'/p['capacity_file'].lstrip('/');assert sha(cap_path)==p['capacity_sha256'];capacity=json.loads(cap_path.read_text())
for k,v in p['budget'].items():assert d[k]==v,(k,d[k],v)
assert len(d['native_attempt_costs'])==d['manual_attempts']==25 and all(x['completed'] for x in d['native_attempt_costs'])
for attempt in d['native_attempt_costs']:
    c=attempt['cost'];assert c['native_forwards']==1 and c['native_forward_trajectories']==2 and c['vjps']==0
    assert c['native_decoder_layer_calls']==72 and c['native_decoder_layer_trajectories']==144
    assert c['extra_replay_calls']==36 and c['extra_replay_trajectories']==72
    assert c['public_FA_activity']['auxiliary_attempts']==c['public_FA_activity']['auxiliary_completed']==36
    assert len(c['finite_FA_activity'])==(36 if attempt['capture_mode']=='finite' else 0)
assert sum(len(x['cost']['finite_FA_activity']) for x in d['native_attempt_costs'])==468
# Require exact already-reviewed production sources, not merely similar math.
verified=json.loads((A/'vendor_fa_development16_summary_20260907.json').read_text());assert verified['status']=='verified_complete'
original_protocol=json.loads((A/'vendor_fa_development16_protocol_20260907.json').read_text())
assert p['sources']==original_protocol['sources'] and p['library_sha256']==verified['library_sha256']
helper=ast.parse((A/'verify_vendor_fa_end_to_end_20260907.py').read_text())
compare=next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='compare')
ns={'np':np};exec(compile(ast.Module(body=[compare],type_ignores=[]),'independent_vector_comparison','exec'),ns)
out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'protocol_sha256':sha(F/'protocol.json'),
     'scope':p['purpose'],'budget':p['budget'],'cases':[],'quality_evaluated':False}
assert [(r['dataset'],r['idx']) for r in d['records']]==[tuple(x) for x in p['selection']]
for row in d['records']:
    cap=next(x for x in capacity['datasets'][row['dataset']]['records'] if x['idx']==row['idx'])
    assert row['N']==len(row['input_ids'])==cap['full_tokens'] and row['prompt_len']==cap['formatted_prompt_tokens']
    assert hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()==row['input_ids_sha256']==cap['input_ids_sha256']
    assert row['eligible_positions']==[row['user_positions'][j] for j in row['keep_local_indices']]
    assert len(row['runs'])==8 and {(r['mode'],r['repeat']) for r in row['runs']}=={(m,i) for m in ['dense','finite'] for i in range(4)}
    dense=next(r['result'] for r in row['runs'] if r['mode']=='dense' and r['repeat']==1)
    detail={'dataset':row['dataset'],'idx':row['idx'],'N':row['N'],'response_length':row['N']-row['prompt_len'],'modes':{}}
    for mode in ['dense','finite']:
        runs=[r for r in row['runs'] if r['mode']==mode];values=[r['result'] for r in runs]
        if mode in row['profiles']:values.append(row['profiles'][mode]['result'])
        for v in values:
            x=np.asarray(v['signed_full_sequence']);assert x.shape==(row['N'],) and np.isfinite(x).all()
            assert all(c==0 for i,c in enumerate(x) if i not in set(row['eligible_positions']))
            assert abs(x.sum()-v['signed_sum'])<1e-9 and abs(v['target_delta_score32_sum64']-v['signed_sum']-v['unassigned_total'])<1e-9
            assert v['native_layer_replay_calls']==v['extra_native_fa_attention_calls']==36
            for c in v['public_FA_capture_checks']:
                assert c['public_output_exact_to_actual_model_FA'] and not c['private_FA_slots_read'] and not c['model_output_replaced'] and c['testing_matrix_numel']==0
            for c in v['native_layer_boundary_checks']['paired_batch']:assert c['native_input_exact'] and c['native_output_exact']
            assert all(c['actual_native_qkv_and_output_exact'] and not c['auxiliary_output_used_by_model'] for c in v['native_fa_operand_audits'])
            assert v['peak_allocated_bytes']>=max(v['capture_peak_before_propagation'],v['propagation_peak_before_full_max'])
            if mode=='finite':
                assert len(v['finite_attention_activity'])==36
                for call in v['finite_attention_activity']:
                    assert call['calls_attempted']==call['calls_enqueued']==1 and len(call['buffer_contract'])==15
                    for buf in call['buffer_contract']:assert buf['shape'] in [[1,32,row['N'],128],[1,32,row['N']]]
        selected=runs[1]['result'];measured=[r['result'] for r in runs if not r['warmup']]
        detail['modes'][mode]={'median_seconds':statistics.median(v['seconds'] for v in measured),'peak_bytes':max(v['peak_allocated_bytes'] for v in measured),
           'capture_peak_bytes':max(v['capture_peak_before_propagation'] for v in measured),'propagation_peak_bytes':max(v['propagation_peak_before_full_max'] for v in measured),
           'all_vectors_identical':all(v['signed_full_sequence']==selected['signed_full_sequence'] for v in values),
           'numerics_to_same_job_dense':ns['compare'](selected['signed_full_sequence'],dense['signed_full_sequence'],row['eligible_positions']),
           'endpoint_score_differences_to_same_job_dense':{k:selected['endpoint_scores32'][k]-dense['endpoint_scores32'][k] for k in ['before','after']},
           'max_relative_unassigned':max(abs(v['unassigned_total'])/max(1,abs(v['target_delta_score32_sum64'])) for v in values),
           'signed_sum':selected['signed_sum'],'target_delta':selected['target_delta_score32_sum64']}
    ordinary=row['ordinary_backward'];assert ordinary['native_root_forwards']==ordinary['native_vjps']==1 and ordinary['backend']=='flash_attention_2' and not ordinary['parameter_gradients_enabled']
    a=detail['modes']['dense'];b=detail['modes']['finite']
    detail.update(finite_to_dense_time_ratio=b['median_seconds']/a['median_seconds'],peak_saved_bytes=a['peak_bytes']-b['peak_bytes'],
       finite_peak_minus_ordinary=b['peak_bytes']-ordinary['peak_allocated_bytes'],ordinary_backward=ordinary,
       ordinary_B1_minus_attribution_B2_score=ordinary['score32_sum64']-dense['endpoint_scores32']['after'])
    if row['profiles']:
        profile=row['profiles']['finite'];path=F/profile['trace'];assert sha(path)==profile['sha256']
        events=json.loads(path.read_text())['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel'];names=[e['name'] for e in kernels]
        assert sum('flash_fwd_kernel' in n for n in names)==profile['actual_default_FA_forward_kernels']==108
        assert sum('deltatrace_fa_finite_p1_kernel' in n for n in names)==profile['actual_finite_kernels']==108
        assert not any('flash_bwd' in n for n in names)
        square=[];shape_count=0
        def walk(value):
            if isinstance(value,list):
                if len(value)>=2 and all(isinstance(v,int) for v in value) and value[-2:]==[row['N'],row['N']]:square.append(value)
                for v in value:walk(v)
        for event in events:
            if event.get('cat')=='cpu_op' and 'Input Dims' in event.get('args',{}):shape_count+=1;walk(event['args']['Input Dims'])
        assert shape_count>100 and not square
        detail['profile']={'native_FA_forwards':108,'finite_kernels':108,'observed_NxN_inputs':0,'CPU_operations_with_shapes':shape_count,
            'finite_kernel_microseconds':sum(e['dur'] for e in kernels if 'deltatrace_fa_finite_p1_kernel' in e['name'])}
        del events,kernels,names
    out['cases'].append(detail)
out['limits']='Real original length/rollout resource evidence and finite-vector comparisons. Not new long-input quality or causal-sign confirmation. Shape traces plus fixed source/buffer contracts do not prove future arbitrary input behavior.'
(A/'vendor_fa_original_long_cost_summary_20260907.json').write_text(json.dumps(out,indent=2))
numeric={'raw_sha256':out['raw_sha256'],'scope':'Full numeric signed vectors and actual costs; no input token IDs or model activations.',
    'records':[{k:v for k,v in r.items() if k!='input_ids'} for r in d['records']], 'native_attempt_costs':d['native_attempt_costs']}
(A/'vendor_fa_original_long_cost_numeric_20260907.json').write_text(json.dumps(numeric,separators=(',',':')))
print(json.dumps(out,indent=2))
