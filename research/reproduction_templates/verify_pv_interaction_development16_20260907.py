"""Independent PV-rule16 source, native runtime, original metric and full-cost review."""
import ast,hashlib,json,math,statistics,re,os,subprocess,sys
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
folder=A/'snapshot${ARTIFACT_ROOT}/codex_pv_interaction_development16_20260907_v1'
if os.name=='nt':folder=Path('\\\\?\\'+str(folder.resolve()))
d=json.loads((folder/'results.json').read_text());p=d['protocol']
assert d['status']=='complete' and d['all_native_vectors_frozen_before_any_quality']
assert d['checkpoint_before']==d['checkpoint_after'] and d['native_sources_before']==d['native_sources_after']
assert p==json.loads((A/'pv_interaction_development16_protocol_20260907.json').read_text())
assert sha(folder/'study.py')==p['study_sha256'] and (folder/'study.py').read_bytes()==(A/'pv_interaction_development16_20260907.py').read_bytes()
for name,digest in p['sources'].items():assert sha(folder/name)==sha(A/name)==digest
for name,meta in json.loads((folder/'artifact_receipt.json').read_text()).items():
    if (folder/name).exists():assert sha(folder/name)==meta['sha256'] and (folder/name).stat().st_size==meta['bytes']
assert d['reused_quality_curves']+d['fresh_quality_curves']==176
assert d['evaluation_forwards']==21*d['fresh_quality_curves']<=p['maximum_evaluation_forwards']
executed=dict(p['budget']);executed['evaluation_forwards']=d['evaluation_forwards'];executed['native_root_forwards']+=d['evaluation_forwards']
assert d['executed_budget']==executed and d['native_root_forwards']<=p['maximum_native_root_forwards']
for key,value in executed.items():assert d[key]==value
assert d['manual_attempts']==d['manual_passes']==len(d['native_attempt_costs'])==198
assert all(x['completed'] for x in d['native_attempt_costs'])
for rule in p['pv_rules'].values():assert sum(x['pv_rule']==rule for x in d['native_attempt_costs'])==66
for x in d['native_attempt_costs']:
    c=x['cost']
    assert c['native_forwards']==1 and c['native_forward_trajectories']==2 and c['vjps']==0
    assert c['native_decoder_layer_calls']==72 and c['native_decoder_layer_trajectories']==144
    assert c['extra_replay_calls']==36 and c['extra_replay_trajectories']==72
parents={}
for source,digest in [(p['required_pilot'],p['required_pilot_sha256']),(p['fallback_quality_parent'],p['fallback_quality_parent_sha256'])]:
    f=A/'snapshot'/source.lstrip('/');assert sha(f)==digest
    h=json.loads(f.read_text());assert h['status']=='complete'
    assert h['checkpoint_before']==h['checkpoint_after']==d['checkpoint_before']
    assert h['native_sources_before']==h['native_sources_after']==d['native_sources_before']
    assert h['dtype']==d['dtype']=='torch.float16' and h['torch_version']==d['torch_version']
    assert h['protocol']['official_normalized_sources']==p['official_normalized_sources']
    assert h['protocol']['official_evaluation_backend']==p['official_evaluation_backend']
    parents[source]=h
expected={(ds,i) for ds,indices in p['selection'].items() for i in indices}
assert len(d['records'])==16 and {(r['dataset'],r['idx']) for r in d['records']}==expected
subprocess.run([sys.executable,str(A/'verify_pv_interaction_source_20260907.py')],check=True,stdout=subprocess.DEVNULL)
out={'status':'verified_complete','raw_sha256':sha(folder/'results.json'),'budget':executed,'base_budget':p['budget'],
 'reused_curves':0,'fresh_curves':0,'curves_verified':0,'max_metric_reconstruction_error':0.,'checked_ledger_rows':0,'cases':[],'profiles':[],
 'scope':'Complete original NI0-7/MH0-7 development. Only PV multipliers differ between3discrete rules. Actual native FA unchanged; mixed endpoints are algebraic terms only. Exact symbolic identity does not erase actual numerical residual. Original176 curves and complete per-call timings. Per-op ledger explicitly None, not measured. Not independent/full248/sign-causality success.'}
helper=ast.parse((A/'verify_native_output_contrast_development16_20260906.py').read_text())
functions=[n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name in ['area','metrics']]
metric_function=next(n for n in functions if n.name=='metrics')
assert isinstance(metric_function.body[-1],ast.Assert) and isinstance(metric_function.body[-2],ast.Assign)
metric_function.body=metric_function.body[:-2]
ns={'np':np,'math':math,'out':out};exec(compile(ast.Module(body=functions,type_ignores=[]),'independent_original_metrics','exec'),ns)
unique_parent_curves=set()
ancestor_cache={}
def load_verified_ancestor(file,digest):
    key=(str(file),digest)
    if key not in ancestor_cache:
        raw=file.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest
        ancestor_cache[key]=json.loads(raw)
    return ancestor_cache[key]
def check_run(r,eligible,ref,branch,diagnostics):
    c = r['end_to_end_cost']
    assert c['native_forwards'] == 1 and c['native_forward_trajectories'] == 2 and (c['vjps'] == 0)
    assert c['native_decoder_layer_calls'] == 72 and c['native_decoder_layer_trajectories'] == 144
    assert c['extra_replay_calls'] == 36 and c['extra_replay_trajectories'] == 72
    assert r['preparation_host_seconds_this_call'] > 0
    assert r['seconds'] >= r['propagation_internal_seconds'] > 0
    assert r['per_operator_ledger_collected'] is False
    assert all((math.isfinite(v) for v in r['endpoint_scores32'].values()))
    assert r['target_delta_score32_sum64'] == r['endpoint_scores32']['after'] - r['endpoint_scores32']['before']
    if True:
        assert r['native_layer_replay_calls'] == 36 and r['native_layer_replay_endpoint_trajectories'] == 72
        assert len(r['native_paired_replay_layouts']) == 36
        for layer in r['native_paired_replay_layouts']:
            for key, meta in layer['tensors'].items():
                assert meta['shape'][0] == 2 and len(meta['endpoint_strides']) == 2
                assert meta['endpoint_storage_offsets'] == [0, meta['stride'][0]]
    assert r['seconds'] == c['seconds'] == r['comparable_attribution_seconds']
    assert r['peak_allocated_bytes'] == c['peak_allocated_bytes']
    for endpoint in ['paired_batch']:
        checks = r['native_layer_boundary_checks'][endpoint]
        assert len(checks) == 36 and all((x['native_input_exact'] and x['native_output_exact'] for x in checks))
    assert len(r['native_fa_operand_audits']) == 72
    for audit in r['native_fa_operand_audits']:
        assert audit['actual_native_qkv_and_output_exact'] and audit['extra_native_attention_calls'] == 0
        assert audit['model_dropout'] == 0.0 and (not audit['auxiliary_output_used_by_model'])
        for k in ['probability_relative_l2_to_explicit_QK', 'probability_row_sum_max_error', 'PV_relative_l2_to_actual_FA']:
            assert 0 <= audit[k] <= 0.001
    assert r['unassigned_total'] == r['target_delta_score32_sum64'] - r['signed_sum']
    scale = max(1.0, abs(r['target_delta_score32_sum64']))
    assert r['numerical_review']['relative_unassigned'] == abs(r['unassigned_total']) / scale
    assert all((r[k] is None for k in ['ledger', 'ledger_residual_sum', 'absolute_ledger_residual_sum', 'unbooked_rounding_residual']))
    assert r['numerical_review']['relative_absolute_ledger'] is r['numerical_review']['relative_unbooked'] is None
    x = np.array(r['signed_full_sequence'])
    assert np.isfinite(x).all() and all((v == 0 for j, v in enumerate(x) if j not in set(eligible)))
    flips = np.sign(x[eligible]) != np.sign(ref[eligible])
    diagnostics.append({'branch': branch, 'repeat': r['repeat'], 'signed_relative_l2_to_same_job_baseline': float(np.linalg.norm(x - ref) / max(np.linalg.norm(ref), 1e-30)), 'sign_changed_tokens': int(flips.sum()), 'reference_credit_mass_on_changed_signs_fraction': float(np.abs(ref[eligible][flips]).sum() / max(np.abs(ref[eligible]).sum(), 1e-30))})
    out['checked_ledger_rows'] += 0

    assert [x['layer'] for x in r['layer_checks']]==list(reversed(range(36)))
    for check in r['layer_checks']:
        assert check['native_attention_qkv_and_output_captured'] and check['native_LSE_probability_PV_approximation_explicit']
        assert check['native_swiglu_product_endpoints_exact'] and math.isfinite(check['max_abs_input_multiplier'])
    assert [(a['layer'],a['endpoint']) for a in r['native_fa_operand_audits']]==[(i,side) for i in reversed(range(36)) for side in ['before','after']]

def check_profile(row,method,profile):
    f = folder / profile['trace']
    assert sha(f) == profile['sha256']
    trace = json.loads(f.read_text())
    names = [e['name'] for e in trace['traceEvents'] if e.get('cat') == 'kernel']
    assert sum(('flash_fwd_kernel' in n for n in names)) == 72 and (not any(('flash' in n.lower() and 'bwd' in n.lower() for n in names)))
    annotations = [e for e in trace['traceEvents'] if e.get('name') == 'ATTR_NATIVE_HALF_LINEAR' and e.get('ph') == 'X' and (e.get('cat') == 'user_annotation')]
    assert len(annotations) == 253
    cpu = {e.get('args', {}).get('External id'): e for e in trace['traceEvents'] if e.get('cat') == 'cpu_op' and e.get('args', {}).get('External id') is not None}
    runtime_by_correlation = {}
    for event in trace['traceEvents']:
        if event.get('cat') == 'cuda_runtime' and 'LaunchKernel' in event.get('name', ''):
            correlation = event.get('args', {}).get('correlation')
            if correlation is not None:
                assert correlation not in runtime_by_correlation
                runtime_by_correlation[correlation] = event
    correlation_link_repairs = []
    projection_kernels = []
    unlinked = []
    coverage = set()
    for e in trace['traceEvents']:
        if e.get('cat') != 'kernel' or 'gemm' not in e['name'].lower():
            continue
        host = cpu.get(e.get('args', {}).get('External id'))
        if host is None:
            launch = runtime_by_correlation.get(e.get('args', {}).get('correlation'))
            if launch is not None:
                hosts = [h for h in cpu.values() if h['name'] == 'aten::mm'
                         and h.get('pid') == launch.get('pid') and h.get('tid') == launch.get('tid')
                         and h['ts'] <= launch['ts']
                         and launch['ts'] + launch.get('dur', 0) <= h['ts'] + h.get('dur', 0) + 0.001]
                assert len(hosts) <= 1, 'Ambiguous runtime-to-aten::mm evidence'
                if hosts:
                    host = hosts[0]
                    assert e['ts'] >= launch['ts']
                    correlation_link_repairs.append({'correlation': e['args']['correlation'],
                        'kernel': e['name'], 'launch': launch, 'cpu_aten_mm': host})
            if host is None:
                unlinked.append(e['name'])
                continue
        for index, scope in enumerate(annotations):
            if scope['ts'] <= host['ts'] and host['ts'] + host.get('dur', 0) <= scope['ts'] + scope['dur'] + 0.001:
                projection_kernels.append(e['name'])
                coverage.add(index)
                break
    assert len(coverage) == 253, 'Every attribution projection must have actual native GEMM evidence'
    assert all(('__half' in n or 'fp16' in n.lower() for n in projection_kernels))
    profile_dispatch = {'dataset': row['dataset'], 'idx': row['idx'], 'method': method, 'pv_rule': p['pv_rules'][method]}
    profile_dispatch['runtime_correlation_links_for_missing_external_id'] = correlation_link_repairs
    profile_dispatch['native_half_projection_dispatch'] = {'annotated_calls': len(annotations), 'covered_calls': len(coverage), 'actual_GEMM_kernel_calls': len(projection_kernels), 'actual_kernel_names': sorted(set(projection_kernels)), 'unlinked_GEMM_events_outside_claim': unlinked}
    generated = list((folder / 'inductor_cache').rglob('*.py'))
    finite_files = [f for f in generated if any((marker in f.read_text() for marker in ['def triton_red_fused__log_softmax', 'def triton_red_fused_add_div_mean_mul_pow_reciprocal_sqrt_sum', 'def triton_red_fused_all_gt', 'def triton_per_fused_all_bitwise_and_gt']))]
    assert finite_files, 'No generated finite-rule source evidence'
    kernel_names = set()
    for f in finite_files:
        kernel_names.update(re.findall('def (triton_[a-zA-Z0-9_]+)', f.read_text()))
    assert any(('_log_softmax' in n for n in kernel_names)) and any(('mean_mul_pow_reciprocal_sqrt_sum' in n for n in kernel_names))
    actual = [n for n in names if any((k in n for k in kernel_names))]
    assert actual, 'No actual compiled finite-rule kernel execution'
    profile_dispatch['compiler_dispatch'] = {'generated_finite_rule_files': [str(f.relative_to(folder)) for f in finite_files], 'actual_kernel_names': sorted(set(actual)), 'actual_kernel_calls': len(actual)}
    seed_files = [f for f in generated if 'scatter' in f.read_text()]
    assert seed_files, 'No generated output-seed source'
    seed_names = set()
    for f in seed_files:
        seed_names.update(re.findall('def (triton_[a-zA-Z0-9_]+)', f.read_text()))
    seed_actual = [n for n in names if any((k in n for k in seed_names))]
    assert seed_actual, 'No actual generated seed kernel dispatch'
    profile_dispatch['compiled_seed_dispatch'] = {'generated_files': [str(f.relative_to(folder)) for f in seed_files], 'actual_kernel_names': sorted(set(seed_actual)), 'actual_kernel_calls': len(seed_actual)}
    probability_files = [f for f in generated if 'masked_fill' in f.read_text() and 'softmax' in f.read_text()]
    pv_files = [f for f in generated if 'linalg_vector_norm' in f.read_text() and 'softmax' not in f.read_text() and ('masked_fill' not in f.read_text())]
    assert probability_files and pv_files, 'Missing generated probability/PV audit source'
    for label, files in [('probability', probability_files), ('pv_audit', pv_files)]:
        definitions = set()
        for f in files:
            definitions.update(re.findall('def (triton_[a-zA-Z0-9_]+)', f.read_text()))
        dispatched = [n for n in names if any((k in n for k in definitions))]
        assert dispatched, 'No actual new probability/audit dispatch'
        profile_dispatch['compiled_' + label + '_dispatch'] = {'generated_files': [str(f.relative_to(folder)) for f in files], 'actual_kernel_names': sorted(set(dispatched)), 'actual_kernel_calls': len(dispatched)}
    difference_files = [f for f in generated if '_to_copy_mul_sub_sum' in f.read_text()]
    definitions = set()
    for f in difference_files:
        definitions.update(re.findall('def (triton_[a-zA-Z0-9_]+)', f.read_text()))
    dispatched = [n for n in names if any((k in n for k in definitions))]
    assert not dispatched, 'Production unexpectedly executes per-operator FP64 difference audit'
    profile_dispatch['compiled_difference_audit_dispatch'] = {'generated_files': [str(f.relative_to(folder)) for f in difference_files], 'actual_kernel_names': sorted(set(dispatched)), 'actual_kernel_calls': len(dispatched)}
    plain_files = [f for f in generated if re.search('def triton_[A-Za-z0-9_]*_to_copy_mul_sum_', f.read_text())]
    plain_defs = set()
    for f in plain_files:
        plain_defs.update(re.findall('def (triton_[a-zA-Z0-9_]+)', f.read_text()))
    plain_actual = [n for n in names if any((k in n for k in plain_defs))]
    assert not plain_actual, 'Production unexpectedly executes per-operator FP64 ordinary audit'
    profile_dispatch['plain_FP64_audit_actual_kernel_calls'] = len(plain_actual)
    profile_dispatch['actual_flash_forward_kernels'] = sum(('flash_fwd_kernel' in n for n in names))
    profile_dispatch['note'] = 'Compiler-cache existence is not execution. Full diagnostic runs generate audit code; production profiles must have0such kernel calls.'
    swiglu_scopes = [e for e in trace['traceEvents'] if e.get('name') == 'ATTR_COMPILED_SWIGLU_SECANT' and e.get('ph') == 'X' and (e.get('cat') == 'user_annotation')]
    assert len(swiglu_scopes) == 36
    swiglu_files = [f for f in generated if re.search('def triton_[A-Za-z0-9_]*sigmoid[A-Za-z0-9_]*where', f.read_text())]
    assert swiglu_files, 'Missing generated original finite SwiGLU expression'
    swiglu_defs = set()
    for f in swiglu_files:
        swiglu_defs.update(re.findall('def (triton_[a-zA-Z0-9_]+)', f.read_text()))
    actual_swiglu = []
    covered_swiglu = set()
    for e in trace['traceEvents']:
        if e.get('cat') != 'kernel' or not any((k in e['name'] for k in swiglu_defs)):
            continue
        host = cpu.get(e.get('args', {}).get('External id'))
        assert host is not None, 'Unlinked alleged compiled SwiGLU kernel'
        indices = [i for i, scope in enumerate(swiglu_scopes) if scope['ts'] <= host['ts'] and host['ts'] + host.get('dur', 0) <= scope['ts'] + scope['dur'] + 0.001]
        assert len(indices) == 1
        covered_swiglu.update(indices)
        actual_swiglu.append(e['name'])
    assert len(covered_swiglu) == 36, 'Not all36 real attribution layers dispatch the compiled finite rule'
    profile_dispatch['compiled_swiglu_dispatch'] = {'generated_files': [str(f.relative_to(folder)) for f in swiglu_files], 'actual_kernel_names': sorted(set(actual_swiglu)), 'actual_kernel_calls': len(actual_swiglu), 'covered_original_layer_calls': len(covered_swiglu)}
    all_generated_definitions = {}
    for f in generated:
        for kernel in re.findall('def (triton_[a-zA-Z0-9_]+)', f.read_text()):
            all_generated_definitions.setdefault(kernel, []).append(str(f.relative_to(folder)))
    boundary_proofs = {}
    for label, expected in [('ATTR_COMPILED_MIDPOINT', 144 if p['pv_rules'][method]=='symmetric' else 72), ('ATTR_COMPILED_SCALED_PROBABILITY', 72), ('ATTR_COMPILED_ATTENTION_LAYOUT', 36), ('ATTR_COMPILED_NORM_RESIDUAL_TWO', 36), ('ATTR_COMPILED_NORM_RESIDUAL_THREE', 36)]:
        scopes = [e for e in trace['traceEvents'] if e.get('name') == label and e.get('ph') == 'X' and (e.get('cat') == 'user_annotation')]
        assert len(scopes) == expected, (label, len(scopes))
        actual = []
        covered = set()
        files = set()
        for e in trace['traceEvents']:
            if e.get('cat') != 'kernel':
                continue
            host = cpu.get(e.get('args', {}).get('External id'))
            if host is None:
                continue
            indices = [i for i, scope in enumerate(scopes) if scope['ts'] <= host['ts'] and host['ts'] + host.get('dur', 0) <= scope['ts'] + scope['dur'] + 0.001]
            if not indices:
                continue
            assert len(indices) == 1
            definitions = [k for k in all_generated_definitions if k in e['name']]
            assert definitions, (label, 'Unexpected noncompiled GPU path', e['name'])
            actual.append(e['name'])
            covered.update(indices)
            for k in definitions:
                files.update(all_generated_definitions[k])
        assert len(covered) == expected, (label, 'Missing actual dispatch', len(covered))
        boundary_proofs[label] = {'annotated_calls': len(scopes), 'covered_calls': len(covered), 'actual_kernel_calls': len(actual), 'actual_kernel_names': sorted(set(actual)), 'matching_generated_files': sorted(files)}
    profile_dispatch['compiled_boundary_dispatch'] = boundary_proofs
    profile_dispatch['all_actual_GEMM_kernel_calls']=sum('gemm' in n.lower() for n in names)
    out['profiles'].append(profile_dispatch)

for row in d['records']:
    assert row['complete'] and row['native_complete']
    payload={k:v for k,v in row.items() if k!='record_digest_sha256'}
    assert hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==row['record_digest_sha256']
    assert hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()==row['input_ids_sha256']
    assert row['memory_reference']['native_root_forwards']==row['memory_reference']['native_vjps']==1
    assert row['memory_reference']['backend']=='flash_attention_2' and row['memory_reference']['parameter_gradients_enabled'] is False
    assert row['eligible_positions']==[row['user_positions'][j] for j in row['keep_local_indices']]
    assert row['gold_eligible_local']==sorted(set(row['gold_full_local'])&set(row['keep_local_indices']))
    primary=parents[p['required_pilot']]
    old=next((x for x in primary['records'] if (x['dataset'],x['idx'])==(row['dataset'],row['idx'])),None)
    if old is not None:
        history=primary;parent_source=p['required_pilot'];parent_digest=p['required_pilot_sha256'];parent_map=p['parent_method_map']
    else:
        history=parents[p['fallback_quality_parent']];parent_source=p['fallback_quality_parent'];parent_digest=p['fallback_quality_parent_sha256'];parent_map=p['fallback_parent_method_map']
        old=next(x for x in history['records'] if (x['dataset'],x['idx'])==(row['dataset'],row['idx']))
    for key in ['input_ids','prompt_len','user_positions','keep_local_indices','eligible_positions']:assert row[key]==old[key]
    assert set(row['scores'])==set(row['metrics'])==set(p['native_methods']+p['baselines'])
    assert set(row['repeats'])==set(p['native_methods'])
    ref=np.array(row['native'][p['native_methods'][0]]['signed_full_sequence']);eligible=row['eligible_positions']
    diagnostics=[];costs={};numerics={}
    ft=statistics.median(x['seconds'] for x in row['ft_both1_repeats']);assert len(row['ft_both1_repeats'])==3
    assert row['ft_costs']['flashtrace_both_hop1']==row['ft_both1_warmup']
    # Avoid double-counting the warmup alias in ft_costs.
    ft_calls=[row['ft_both1_warmup']]+row['ft_both1_repeats']+[v for k,v in row['ft_costs'].items() if k!='flashtrace_both_hop1']+[row['legacy_joint_cost']]
    assert len(ft_calls)==8
    for fc in ft_calls:
        assert fc['native_forwards']==fc['native_forward_trajectories']==1 and fc['vjps']==0
        assert fc['native_decoder_layer_calls']==fc['native_decoder_layer_trajectories']==36
        assert fc['extra_replay_calls']==fc['extra_replay_trajectories']==0
    for method in p['native_methods']:
        runs=row['repeats'][method];selected=row['native'][method]
        assert len(runs)==4
        for i,r in enumerate(runs):
            assert r['repeat']==i and r['warmup']==(i==0) and r['method']==method and r['pv_rule']==p['pv_rules'][method]
            check_run(r,eligible,ref,method,diagnostics)
            assert r['endpoint_scores32']==row['native'][p['native_methods'][0]]['endpoint_scores32']
        assert selected['signed_full_sequence']==runs[-1]['signed_full_sequence']
        value=np.array(selected['signed_full_sequence'])
        assert math.isclose(float(value.sum()),selected['signed_sum'],abs_tol=1e-9,rel_tol=1e-12)
        assert np.array_equal(np.maximum(value[row['user_positions']],0).astype(np.float32),np.array(row['scores'][method],dtype=np.float32))
        elapsed=statistics.median(r['seconds'] for r in runs[1:]);peak=max(r['peak_allocated_bytes'] for r in runs[1:])
        assert selected['comparable_attribution_seconds']==elapsed and selected['peak_allocated_bytes']==peak
        assert selected['time_ratio_to_ft']==elapsed/ft and selected['time_gate_pass']==(elapsed/ft<=1)
        excess=peak-row['memory_reference']['peak_allocated_bytes']
        assert selected['peak_bytes_above_reference']==excess and selected['memory_gate_pass']==(excess<=p['memory_allowance_bytes'])
        costs[method]={'median_seconds':elapsed,'peak_bytes':peak,'FT_ratio':elapsed/ft,'peak_above_ordinary_FA_bytes':excess,
          'first_complete_seconds':runs[0]['seconds'],'individual_complete_seconds':[r['seconds'] for r in runs],
          'paired_round_FT_ratios':[r['seconds']/f['seconds'] for r,f in zip(runs[1:],row['ft_both1_repeats'])],
          'time_gate_pass':selected['time_gate_pass'],'memory_gate_pass':selected['memory_gate_pass']}
        numerics[method]={'signed_relative_l2_to_symmetric':float(np.linalg.norm(value-ref)/max(np.linalg.norm(ref),1e-30)),
          'sign_changed_tokens_to_symmetric':int((np.sign(value[eligible])!=np.sign(ref[eligible])).sum()),
          'negative_tokens':int((value[eligible]<0).sum()),'negative_absolute_mass':float(-value[value<0].sum()),
          'positive_absolute_mass':float(value[value>0].sum()),'signed_sum':selected['signed_sum'],
          'target_delta':selected['target_delta_score32_sum64'],'unassigned_total':selected['unassigned_total'],
          'relative_unassigned':selected['numerical_review']['relative_unassigned'],
          'repeat_vectors_identical':all(r['signed_full_sequence']==selected['signed_full_sequence'] for r in runs),
          'endpoint_sums_identical_to_symmetric':selected['endpoint_scores32']==row['native'][p['native_methods'][0]]['endpoint_scores32']}
    for method in row['scores']:
        provenance = row['evaluation_provenance'][method]
        cost = row['evaluation_costs'][method]
        parent_method = parent_map.get(method, method)
        if provenance['mode'] == 'identical_score_parent_curve':
            assert row['scores'][method] == old['scores'][parent_method]
            assert provenance['source'] == parent_source and provenance['source_sha256'] == parent_digest and (provenance['source_method'] == parent_method)
            assert provenance['historical_evaluation_cost'] == old['evaluation_costs'][parent_method]
            for field in ['metrics', 'evaluation_masks']:
                assert row[field][method] == old[field][parent_method]
            if parent_method in old.get('recovery_topk_local', {}):
                assert row['recovery_topk_local'][method] == old['recovery_topk_local'][parent_method]
            assert cost['native_forwards'] == cost['native_decoder_layer_calls'] == cost['vjps'] == 0
            origin_report, origin_row, origin_method = (history, old, parent_method)
            chain = []
            seen = set()
            while origin_report.get('reused_quality_curves', 0) and origin_row.get('evaluation_provenance', {}).get(origin_method, {}).get('mode') == 'identical_score_parent_curve':
                link = origin_row['evaluation_provenance'][origin_method]
                assert link['source'].startswith('${ARTIFACT_ROOT}/codex_')
                identity = (link['source'], link['source_method'])
                assert identity not in seen
                seen.add(identity)
                file = A / 'snapshot' / link['source'].lstrip('/')
                assert sha(file) == link['source_sha256']
                ancestor = json.loads(file.read_text())
                assert ancestor['status'] == 'complete' and ancestor['checkpoint_before'] == d['checkpoint_before']
                assert ancestor['native_sources_before'] == d['native_sources_before']
                assert ancestor['protocol']['official_normalized_sources'] == p['official_normalized_sources']
                earlier = next((x for x in ancestor['records'] if (x['dataset'], x['idx']) == (row['dataset'], row['idx'])))
                for key in ['input_ids', 'prompt_len', 'user_positions', 'keep_local_indices', 'eligible_positions']:
                    assert earlier[key] == row[key]
                for key in ['scores', 'metrics', 'evaluation_masks']:
                    assert earlier[key][link['source_method']] == origin_row[key][origin_method]
                chain.append({'source': link['source'], 'sha256': link['source_sha256'], 'method': link['source_method']})
                origin_report, origin_row, origin_method = (ancestor, earlier, link['source_method'])
            assert origin_row['evaluation_costs'][origin_method]['native_forwards'] == 21
            row.setdefault('_verified_curve_origins', {})[method] = {'chain': chain, 'original_fresh_cost': origin_row['evaluation_costs'][origin_method]}
            unique_parent_curves.add((row['dataset'], row['idx'], parent_method))
            out['reused_curves'] += 1
        else:
            assert provenance['mode'] == 'fresh_original_curve' and (parent_method not in old['scores'] or row['scores'][method] != old['scores'][parent_method])
            assert cost['native_forwards'] == 21 and cost['native_decoder_layer_calls'] == 756 and (cost['vjps'] == 0)
            out['fresh_curves'] += 1
        ns['metrics'](row, method)
    assert sum((x['native_forwards'] for x in row['evaluation_costs'].values())) == 21 * sum((x['mode'] == 'fresh_original_curve' for x in row['evaluation_provenance'].values()))

    for method,profile in row.get('profiles',{}).items():
        assert profile['pv_rule']==p['pv_rules'][method] and profile['actual_flash_forward_kernels']==72
        assert row['profile_numerical_comparisons'][method]['per_operator_ledger_collected'] is False
        check_profile(row,method,profile)
    out['cases'].append({'dataset':row['dataset'],'idx':row['idx'],'N':len(row['input_ids']),'costs':costs,'fresh_FT_seconds':ft,
        'metrics':{m:{k:row['metrics'][m][k] for k in ['rise','mas','recovery']} for m in row['metrics']},
        'numerics':numerics,'numerical_diagnostics':diagnostics})
assert out['curves_verified']==176 and out['checked_ledger_rows']==0 and len(out['profiles'])==6
assert out['reused_curves']==d['reused_quality_curves'] and out['fresh_curves']==d['fresh_quality_curves']
for idx in [0,2]:
    profiles=[x for x in out['profiles'] if x['idx']==idx];assert len(profiles)==3
    assert len(set(x['all_actual_GEMM_kernel_calls'] for x in profiles))==1,'PV rules changed GEMM count'
out['unique_parent_curves_reused']=len(unique_parent_curves)
out['physical_decoder_calls']=36*executed['native_root_forwards']+executed['extra_layer_replay_calls']
out['endpoint_decoder_trajectories']=36*(executed['native_attribution_endpoint_trajectories']+executed['ordinary_reference_forwards']+executed['ft_attribution_forwards']+executed['evaluation_forwards'])+executed['extra_layer_replay_endpoint_trajectories']
out['production_calls_without_ledger']=198;out['diagnostic_calls_with_ledger']=0
methods=p['native_methods']+p['baselines'];sym=p['native_methods'][0]
out.update(means={},paired_comparisons={},quality_gates={},cost_overview={},improvement_over_symmetric={})
rng=np.random.default_rng(73016)
for dataset in p['selection']:
    rows=[r for r in out['cases'] if r['dataset']==dataset];assert len(rows)==8
    fields=['rise','mas']+(['recovery'] if dataset=='niah_mq_q2' else [])
    out['means'][dataset]={m:{field:float(np.mean([r['metrics'][m][field] for r in rows])) for field in fields} for m in methods}
    out['paired_comparisons'][dataset]={}
    for candidate in p['native_methods']:
        out['paired_comparisons'][dataset][candidate]={}
        for control in p['native_methods']+p['baselines']:
            if control==candidate:continue
            comp={}
            for field in fields:
                differences=np.array([r['metrics'][candidate][field]-r['metrics'][control][field] for r in rows])
                draws=differences[rng.integers(0,8,size=(10000,8))].mean(1)
                benefit=differences if field=='recovery' else -differences
                comp[field]={'mean_candidate_minus_control':float(differences.mean()),
                    'paired_mean_bootstrap95_descriptive_only':np.quantile(draws,[.025,.975]).tolist(),
                    'wins':int((benefit>0).sum()),'ties':int((benefit==0).sum()),'losses':int((benefit<0).sum()),'per_case_differences':differences.tolist()}
            out['paired_comparisons'][dataset][candidate][control]=comp
for candidate in p['native_methods']:
    ni=out['means']['niah_mq_q2'];mh=out['means']['morehopqa']
    gate={'NI_recovery_not_below_best_FT_mean':ni[candidate]['recovery']>=max(ni[m]['recovery'] for m in p['baselines']),
      'MH_RISE_below_best_FT_mean':mh[candidate]['rise']<min(mh[m]['rise'] for m in p['baselines']),
      'MH_MAS_below_best_FT_mean':mh[candidate]['mas']<min(mh[m]['mas'] for m in p['baselines'])}
    gate['joint_quality_gate']=all(gate.values());out['quality_gates'][candidate]=gate
    gains=[ni[candidate]['recovery']-ni[sym]['recovery'],mh[sym]['rise']-mh[candidate]['rise'],mh[sym]['mas']-mh[candidate]['mas']]
    out['improvement_over_symmetric'][candidate]={'goal_metric_gains':gains,'none_worse_and_one_better':all(x>=0 for x in gains) and any(x>0 for x in gains),
        'NI_RISE_change':ni[candidate]['rise']-ni[sym]['rise'],'NI_MAS_change':ni[candidate]['mas']-ni[sym]['mas']}
    ratios=np.array([r['costs'][candidate]['FT_ratio'] for r in out['cases']]);peaks=np.array([r['costs'][candidate]['peak_above_ordinary_FA_bytes'] for r in out['cases']])
    symmetric_ratios=np.array([r['costs'][candidate]['median_seconds']/r['costs'][sym]['median_seconds'] for r in out['cases']])
    out['cost_overview'][candidate]={'median_ratio_to_FT':float(np.median(ratios)),'p95_ratio_to_FT':float(np.quantile(ratios,.95)),
        'max_ratio_to_FT':float(ratios.max()),'FT_time_gate_passed_cases':int((ratios<=1).sum()),'all_cases_FT_time_gate':bool((ratios<=1).all()),
        'max_peak_above_ordinary_FA_bytes':int(peaks.max()),'all_cases_memory_gate':bool((peaks<=p['memory_allowance_bytes']).all()),
        'median_time_ratio_to_symmetric':float(np.median(symmetric_ratios)),'faster_than_symmetric_cases':int((symmetric_ratios<1).sum()),
        'peak_difference_to_symmetric_by_case':[r['costs'][candidate]['peak_bytes']-r['costs'][sym]['peak_bytes'] for r in out['cases']]}
(A/'pv_interaction_development16_summary_20260907.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['cases','profiles','paired_comparisons']},indent=2))
