"""Independently verify public FA source, actual calls, vectors and cost guards."""
import ast,hashlib,json,statistics
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_public_fa_capture_probe_20260907_v2'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete'
assert p==json.loads((A/'public_fa_capture_probe_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'public_fa_capture_probe_20260907.py')
for name,digest in p['sources'].items():assert sha(F/name)==sha(A/name)==digest
parent_file=A/'snapshot'/p['required_parent'].lstrip('/')
assert sha(parent_file)==p['required_parent_sha256'];parent=json.loads(parent_file.read_text())
assert d['checkpoint_before']==d['checkpoint_after']==parent['checkpoint_before']==parent['checkpoint_after']
assert d['native_sources_before']==d['native_sources_after']==parent['native_sources_before']==parent['native_sources_after']
assert d['dtype']==parent['dtype']=='torch.float16' and d['torch_version']==parent['torch_version']
for name,digest in p['official_normalized_sources'].items():
    assert hashlib.sha256((A/'snapshot${FLASHTRACE_ROOT}'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest
# Frozen finite arithmetic and endpoint root capture unchanged.
assert p['sources']['qwen_signed_secant_pv_rules.py']==parent['protocol']['sources']['qwen_signed_secant_pv_rules.py']
old=ast.parse((F/'qwen_signed_secant_native_paired_pv_rules.py').read_text())
new=ast.parse((F/'qwen_signed_secant_paired_public_fa.py').read_text())
for name in ['capture_checkpoint_pair_raw','capture_checkpoint_pair']:
    extract=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    assert extract(old)==extract(new)
source=(F/'qwen_public_fa_layer_replay.py').read_text()
assert '_flash_attn_forward' not in source and 'result[5]' not in source
assert 'from flash_attn import flash_attn_func' in source
assert len(d['native_attempt_costs'])==d['manual_attempts']==26
assert all(x['completed'] for x in d['native_attempt_costs'])
for key,value in p['budget'].items():assert d[key]==value,(key,d[key],value)
for src,target in [('native_forwards','native_root_forwards'),('native_forward_trajectories','native_attribution_endpoint_trajectories'),
                   ('extra_replay_calls','extra_layer_replay_calls'),('extra_replay_trajectories','extra_layer_replay_endpoint_trajectories')]:
    assert sum(x['cost'][src] for x in d['native_attempt_costs'])==d[target]
assert sum(x['cost']['public_FA_activity']['auxiliary_completed'] for x in d['native_attempt_costs'])==468
for attempt in d['native_attempt_costs']:
    c=attempt['cost'];a=c['public_FA_activity'];public=attempt['capture_mode']=='public'
    assert c['native_forwards']==1 and c['native_forward_trajectories']==2
    assert c['native_decoder_layer_calls']==72 and c['native_decoder_layer_trajectories']==144
    assert c['extra_replay_calls']==36 and c['extra_replay_trajectories']==72
    assert a['auxiliary_attempts']==a['auxiliary_completed']==(36 if public else 0)
    assert len(a['metadata'])==(36 if public else 0)
    for m in a['metadata']:assert m['testing_return']['type']=='NoneType' or m['testing_return']['numel']==0
    assert c['seconds']>0 and c['peak_allocated_bytes']>0
out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'protocol_sha256':sha(F/'protocol.json'),
 'budget':p['budget'],'cases':[],'scope':p['purpose'],'decision_guard':p['predeclared_decision'],
 'prior_failed_attempt':p['previous_attempt'],'source_math_unchanged':True}
numeric={'scope':'Complete numeric signed vectors and costs from original cases; no input IDs, prompts or model tensors.',
 'raw_sha256':out['raw_sha256'],'records':[]}
assert [(r['dataset'],r['idx']) for r in d['records']]==[tuple(x) for x in p['selection']]
for row in d['records']:
    prev=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(row['dataset'],row['idx']))
    assert row['N']==len(prev['input_ids']) and row['input_ids_sha256']==prev['input_ids_sha256']
    original=prev['native']['strong_secant_pv_content_P1']['signed_full_sequence']
    assert len(row['runs'])==8 and {(r['repeat'],r['mode']) for r in row['runs']}=={(i,m) for i in range(4) for m in ['public','private']}
    detail={'dataset':row['dataset'],'idx':row['idx'],'N':row['N'],'modes':{}}
    baseline=next(r['result'] for r in row['runs'] if r['mode']=='private')
    for mode in ['private','public']:
        runs=[r for r in row['runs'] if r['mode']==mode];measured=[r['result'] for r in runs if not r['warmup']]
        all_values=[r['result'] for r in runs]
        if mode in row['profiles']:
            profile=row['profiles'][mode];trace=F/profile['trace'];assert sha(trace)==profile['sha256']
            events=json.loads(trace.read_text())['traceEvents']
            names=[e['name'] for e in events if e.get('cat')=='kernel']
            actual=sum('flash_fwd_kernel' in n for n in names)
            assert actual==profile['actual_flash_forward_kernels']==(108 if mode=='public' else 72)
            assert not any('flash_bwd_kernel' in n for n in names)
            all_values.append(profile['result'])
            detail.setdefault('profile_counts',{})[mode]=actual
        for result in all_values:
            assert result['capture_mode']==mode and result['pv_rule']=='content_P1'
            vector=np.asarray(result['signed_full_sequence']);assert vector.shape==(row['N'],) and np.isfinite(vector).all()
            assert result['native_layer_replay_calls']==36 and result['extra_native_fa_attention_calls']==(36 if mode=='public' else 0)
            assert len(result['native_layer_boundary_checks']['paired_batch'])==36
            assert all(x['native_input_exact'] and x['native_output_exact'] for x in result['native_layer_boundary_checks']['paired_batch'])
            assert abs(vector.sum()-result['signed_sum'])<1e-9
            assert abs(result['target_delta_score32_sum64']-result['signed_sum']-result['unassigned_total'])<1e-9
            if mode=='public':
                assert len(result['public_FA_capture_checks'])==36
                for check in result['public_FA_capture_checks']:
                    assert check['public_output_exact_to_actual_model_FA'] and check['testing_matrix_numel']==0
                    assert not check['private_FA_slots_read'] and not check['model_output_replaced']
                    assert check['auxiliary_batch_size']==2 and check['auxiliary_public_FA_calls']==1
        detail['modes'][mode]={'median_seconds':statistics.median(r['seconds'] for r in measured),
            'max_peak_bytes':max(r['peak_allocated_bytes'] for r in measured),
            'all_vectors_exact_to_same_job_private':all(r['signed_full_sequence']==baseline['signed_full_sequence'] for r in all_values),
            'all_vectors_exact_to_frozen_parent':all(r['signed_full_sequence']==original for r in all_values),
            'max_absolute_vector_change':max(float(np.max(abs(np.asarray(r['signed_full_sequence'])-np.asarray(original)))) for r in all_values),
            'max_relative_unassigned':max(abs(r['unassigned_total'])/max(1,abs(r['target_delta_score32_sum64'])) for r in all_values)}
    a=detail['modes']['private'];b=detail['modes']['public']
    detail.update(public_to_private_latency_ratio=b['median_seconds']/a['median_seconds'],public_peak_extra_bytes=b['max_peak_bytes']-a['max_peak_bytes'])
    detail['exact_quality_reuse_guard_pass']=all(x['all_vectors_exact_to_frozen_parent'] for x in detail['modes'].values())
    detail['cost_guard_pass']=detail['public_to_private_latency_ratio']<=p['predeclared_decision']['max_per_case_median_latency_ratio_to_same_job_private'] and detail['public_peak_extra_bytes']<=p['predeclared_decision']['max_per_case_peak_extra_bytes_to_same_job_private']
    out['cases'].append(detail)
    numeric['records'].append({k:row[k] for k in ['dataset','idx','N','input_ids_sha256','runs','profiles']})
out['all_exact_quality_reuse_guards_pass']=all(c['exact_quality_reuse_guard_pass'] for c in out['cases'])
out['all_cost_guards_pass']=all(c['cost_guard_pass'] for c in out['cases'])
out['actual_native_FA_calls_in_successful_job']=26*72+468
(A/'public_fa_capture_probe_summary_20260907.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(A/'public_fa_capture_probe_numeric_20260907.json').write_text(json.dumps(numeric,separators=(',',':')),encoding='utf-8')
print(json.dumps(out,indent=2))
