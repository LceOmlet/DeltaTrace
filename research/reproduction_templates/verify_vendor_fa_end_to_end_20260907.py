"""Independent source, runtime, original-curve, signed-vector and cost review."""
import ast,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_end_to_end_20260907_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete' and d['all_native_vectors_frozen_before_any_quality']
assert p==json.loads((A/'vendor_fa_end_to_end_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'vendor_fa_end_to_end_probe_20260907.py')
for name,h in p['sources'].items():assert sha(F/name)==sha(A/name)==h
parent_path=A/'snapshot'/p['required_parent'].lstrip('/');assert sha(parent_path)==p['required_parent_sha256']
parent=json.loads(parent_path.read_text())
assert d['checkpoint_before']==d['checkpoint_after']==parent['checkpoint_before']==parent['checkpoint_after']
assert d['native_sources_before']==d['native_sources_after']==parent['native_sources_before']==parent['native_sources_after']
assert d['dtype']==parent['dtype']=='torch.float16' and d['torch_version']==parent['torch_version']
for name,h in p['official_normalized_sources'].items():
    assert hashlib.sha256((A/'snapshot${FLASHTRACE_ROOT}'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h
for name,h in parent['protocol']['sources'].items():assert p['sources'][name]==h
build_path=A/'snapshot'/p['build_result'].lstrip('/');assert sha(build_path)==p['build_result_sha256']
build=json.loads(build_path.read_text());assert build['library']['sha256']==p['library_sha256']
local=json.loads((A/'vendor_fa_captured_operator_summary_20260907.json').read_text())
assert local['status']=='verified_complete' and local['local_numerical_guard_pass']
# Old/new attention differs explicitly; unrelated finite math is identical.
base=(F/'qwen_signed_secant_pv_rules.py').read_text();new=(F/'qwen_signed_secant_vendor_fa.py').read_text()
for start,end in [('    with torch.no_grad():','        for li in reversed('),
                  ('            layer = model.model.layers[li]','            q0'),
                  ('            mqp = norm_back','            checks.append('),
                  ('        embedding_delta =','        return {')]:
    assert base[base.index(start):base.index(end,base.index(start))]==new[new.index(start):new.index(end,new.index(start))]
assert not any(isinstance(n,ast.MatMult) for n in ast.walk(ast.parse(new)))
assert 'scaled_probability(' not in new and 'softmax_secant_pullback(' not in new
assert "groups,1.0,k0.shape[1],v0.shape[1],n,h)" in new  # scale already in compiled finite Q/K
assert 'from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair,PairedReplayViews,EndpointReplay' in (F/'qwen_signed_secant_paired_vendor_fa.py').read_text()
for key,value in p['budget'].items():assert d[key]==value,(key,d[key],value)
assert len(d['native_attempt_costs'])==d['manual_attempts']==26 and all(x['completed'] for x in d['native_attempt_costs'])
for attempt in d['native_attempt_costs']:
    c=attempt['cost'];assert c['native_forwards']==1 and c['native_forward_trajectories']==2 and c['vjps']==0
    assert c['native_decoder_layer_calls']==72 and c['native_decoder_layer_trajectories']==144
    assert c['extra_replay_calls']==36 and c['extra_replay_trajectories']==72
    a=c['public_FA_activity'];assert a['auxiliary_attempts']==a['auxiliary_completed']==len(a['metadata'])==36
    assert len(c['finite_FA_activity'])==(36 if attempt['capture_mode']=='finite' else 0)
assert sum(len(x['cost']['finite_FA_activity']) for x in d['native_attempt_costs'])==468
out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'protocol_sha256':sha(F/'protocol.json'),
 'scope':p['purpose'],'budget':p['budget'],'library_sha256':p['library_sha256'],
 'cases':[],'curves_verified':0,'max_metric_reconstruction_error':0.,'source_non_attention_math_unchanged':True,
 'source_has_no_explicit_NxN_attribution_operations':True}
helper=ast.parse((A/'verify_native_output_contrast_development16_20260906.py').read_text())
functions=[n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name in ['area','metrics']]
ns={'np':np,'math':math,'out':out};exec(compile(ast.Module(body=functions,type_ignores=[]),'independent_original_metrics','exec'),ns)
numeric={'raw_sha256':out['raw_sha256'],'scope':'Full signed vectors, original six curves, score vectors, masks and costs; original text/token IDs and activations excluded.','records':[]}
assert [(r['dataset'],r['idx']) for r in d['records']]==[tuple(x) for x in p['selection']]
def compare(x,y,eligible):
    x=np.asarray(x);y=np.asarray(y);a=x[eligible];b=y[eligible];flips=np.sign(a)!=np.sign(b)
    return {'relative_l2':float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30)),
            'maximum_absolute':float(np.max(abs(x-y))),'eligible_sign_changes':int(flips.sum()),
            'reference_absolute_mass_on_changed_signs_fraction':float(abs(b[flips]).sum()/max(abs(b).sum(),1e-30))}
for row in d['records']:
    prev=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(row['dataset'],row['idx']))
    assert row['complete'] and row['N']==len(row['input_ids'])
    for key in ['input_ids','input_ids_sha256','prompt_len','user_positions','keep_local_indices','eligible_positions','gold_full_local','gold_eligible_local']:assert row[key]==prev[key]
    assert hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()==row['input_ids_sha256']
    eligible=row['eligible_positions'];detail={'dataset':row['dataset'],'idx':row['idx'],'N':row['N'],'modes':{},'profiles':{}}
    assert {(r['repeat'],r['mode']) for r in row['runs']}=={(i,m) for i in range(4) for m in ['dense','finite']}
    reference=next(r['result'] for r in row['runs'] if r['mode']=='dense' and r['repeat']==1)
    for mode in ['dense','finite']:
        runs=[r for r in row['runs'] if r['mode']==mode];measured=[r['result'] for r in runs if not r['warmup']]
        selected=next(r['result'] for r in runs if r['repeat']==1);all_values=[r['result'] for r in runs]
        if mode in row['profiles']:
            profile=row['profiles'][mode];path=F/profile['trace'];assert sha(path)==profile['sha256']
            events=json.loads(path.read_text())['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel'];names=[e['name'] for e in kernels]
            assert sum('flash_fwd_kernel' in n for n in names)==profile['actual_flash_forward_kernels']==108
            assert sum('deltatrace_fa_finite_p1_kernel' in n for n in names)==profile['finite_kernels']==(108 if mode=='finite' else 0)
            assert not any('flash_bwd' in n for n in names)
            annotations=[e for e in events if e.get('cat')=='user_annotation' and e.get('name')=='ATTR_VENDOR_FA_FINITE_P1']
            assert len(annotations)==(36 if mode=='finite' else 0)
            square=[];observed_shapes=0
            def inspect_shapes(value,event):
                if isinstance(value,list):
                    if len(value)>=2 and all(isinstance(v,int) for v in value) and value[-2:]==[row['N'],row['N']]:square.append({'op':event['name'],'shape':value})
                    for v in value:inspect_shapes(v,event)
            for event in events:
                if event.get('cat')=='cpu_op' and 'Input Dims' in event.get('args',{}):
                    observed_shapes+=1;inspect_shapes(event['args']['Input Dims'],event)
            assert observed_shapes>100
            assert (len(square)==0) if mode=='finite' else (len(square)>0)
            detail['profiles'][mode]={'native_FA_forwards':108,'finite_kernels':profile['finite_kernels'],
                'finite_operator_scopes':len(annotations),'CPU_ops_with_recorded_shapes':observed_shapes,
                'observed_NxN_inputs_count':len(square),'observed_NxN_examples':square[:4],
                'finite_GPU_kernel_microseconds':sum(e['dur'] for e in kernels if 'deltatrace_fa_finite_p1_kernel' in e['name'])}
            all_values.append(profile['result'])
            del events,kernels,names
        for value in all_values:
            x=np.asarray(value['signed_full_sequence']);assert x.shape==(row['N'],) and np.isfinite(x).all()
            assert all(v==0 for j,v in enumerate(x) if j not in set(eligible))
            assert abs(x.sum()-value['signed_sum'])<1e-9
            assert abs(value['target_delta_score32_sum64']-value['signed_sum']-value['unassigned_total'])<1e-9
            assert value['endpoint_scores32']==reference['endpoint_scores32']==prev['native']['strong_secant_pv_content_P1']['endpoint_scores32']
            assert value['native_layer_replay_calls']==36 and value['extra_native_fa_attention_calls']==36
            assert len(value['public_FA_capture_checks'])==36 and len(value['native_fa_operand_audits'])==72
            for check in value['public_FA_capture_checks']:
                assert check['public_output_exact_to_actual_model_FA'] and not check['private_FA_slots_read'] and not check['model_output_replaced']
                assert check['testing_matrix_numel']==0 and check['auxiliary_public_FA_calls']==1
            for check in value['native_layer_boundary_checks']['paired_batch']:assert check['native_input_exact'] and check['native_output_exact']
            for audit in value['native_fa_operand_audits']:
                assert audit['actual_native_qkv_and_output_exact'] and not audit['auxiliary_output_used_by_model']
                if mode=='finite':assert audit['dense_probability_audit_performed'] is False
            if mode=='finite':
                assert len(value['finite_attention_activity'])==36
                for c in value['finite_attention_activity']:
                    assert c['calls_attempted']==c['calls_enqueued']==1
                    assert len(c['buffer_contract'])==15
                    for buf in c['buffer_contract']:assert buf['shape'] in [[1,32,row['N'],128],[1,32,row['N']]]
            assert value['per_operator_ledger_collected'] is False
            assert all(value[k] is None for k in ['ledger','ledger_residual_sum','absolute_ledger_residual_sum','unbooked_rounding_residual'])
        assert np.array_equal(np.maximum(np.array(selected['signed_full_sequence'])[row['user_positions']],0).astype(np.float32),np.array(row['scores'][mode],dtype=np.float32))
        detail['modes'][mode]={'median_seconds':statistics.median(v['seconds'] for v in measured),
            'peak_bytes':max(v['peak_allocated_bytes'] for v in measured),
            'warmup_seconds':runs[0]['result']['seconds'],
            'all_vectors_identical':all(v['signed_full_sequence']==selected['signed_full_sequence'] for v in all_values),
            'numerics_to_same_job_dense':compare(selected['signed_full_sequence'],reference['signed_full_sequence'],eligible),
            'numerics_to_historical_parent':compare(selected['signed_full_sequence'],prev['native']['strong_secant_pv_content_P1']['signed_full_sequence'],eligible),
            'max_relative_unassigned':max(abs(v['unassigned_total'])/max(1,abs(v['target_delta_score32_sum64'])) for v in all_values),
            'negative_eligible_tokens':int((np.asarray(selected['signed_full_sequence'])[eligible]<0).sum()),
            'metrics':{k:row['metrics'][mode][k] for k in ['rise','mas','recovery']}}
        metric_row=dict(row);metric_row['common_eager_evaluation_endpoints16']=prev['common_eager_evaluation_endpoints16'];ns['metrics'](metric_row,mode)
    ordinary=row['ordinary_backward'];assert ordinary['native_root_forwards']==ordinary['native_vjps']==1 and ordinary['backend']=='flash_attention_2'
    assert ordinary['parameter_gradients_enabled'] is False
    assert math.isfinite(ordinary['score32_sum64'])  # B1 versus B2 arithmetic difference is reported, not threshold tuned.
    a=detail['modes']['dense'];b=detail['modes']['finite']
    detail.update(finite_to_dense_time_ratio=b['median_seconds']/a['median_seconds'],
        full_peak_saved_bytes=a['peak_bytes']-b['peak_bytes'],finite_peak_above_ordinary_backward=b['peak_bytes']-ordinary['peak_allocated_bytes'],
        ordinary_backward=ordinary,ordinary_B1_minus_actual_B2_score=ordinary['score32_sum64']-reference['endpoint_scores32']['after'],
        historical_FT_controls=row['historical_FT_control'])
    assert row['historical_FT_control']['source_sha256']==p['required_parent_sha256']
    assert row['historical_FT_control']['metrics']=={k:v for k,v in prev['metrics'].items() if k.startswith('flashtrace_')}
    out['cases'].append(detail)
    numeric['records'].append({k:row[k] for k in ['dataset','idx','N','input_ids_sha256','user_positions','keep_local_indices','eligible_positions','gold_full_local','gold_eligible_local','runs','profiles','scores','metrics','evaluation_masks','evaluation_costs','recovery_topk_local','ordinary_backward','historical_FT_control'] if k in row})
assert out['curves_verified']==6
out.update(actual_FA_forwards=26*108+3*36,finite_kernel_calls=468*3,
    observed_finite_shape_limit='No NxN inputs in first actual full finite trace; source and buffer contract independently checked. Not allocator-level proof for arbitrary future code or lengths.',
    conclusion_scope='Three original development cases only. No full16newquality, multi-example, long-input, independent-quality or causal-token-sign confirmation.')
(A/'vendor_fa_end_to_end_summary_20260907.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(A/'vendor_fa_end_to_end_numeric_20260907.json').write_text(json.dumps(numeric,separators=(',',':')),encoding='utf-8')
print(json.dumps(out,indent=2))
