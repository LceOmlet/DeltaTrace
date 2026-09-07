"""Independent original metrics, numerical drift, cost and actual compiler/native provenance."""
import ast, hashlib, json, math, statistics, re, os
from pathlib import Path
import numpy as np
A = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
folder = A / 'snapshot${ARTIFACT_ROOT}/codex_secant_boundary_pilot_20260907_v1'
if os.name == 'nt':
    folder = Path('\\\\?\\' + str(folder.resolve()))
d = json.loads((folder / 'results.json').read_text())
p = d['protocol']
assert d['status'] == 'complete' and d['all_native_vectors_frozen_before_any_quality']
assert d['checkpoint_before'] == d['checkpoint_after'] and d['native_sources_before'] == d['native_sources_after']
assert p == json.loads((A / 'secant_boundary_pilot_protocol_20260907.json').read_text())
assert sha(folder / 'study.py') == p['study_sha256'] and (folder / 'study.py').read_bytes() == (A / 'secant_boundary_pilot_20260907.py').read_bytes()
for n, h in p['sources'].items():
    assert sha(folder / n) == sha(A / n) == h
for n, m in json.loads((folder / 'artifact_receipt.json').read_text()).items():
    if (folder / n).exists():
        assert sha(folder / n) == m['sha256'] and (folder / n).stat().st_size == m['bytes']
assert d['reused_quality_curves'] + d['fresh_quality_curves'] == 30
assert d['evaluation_forwards'] == 21 * d['fresh_quality_curves'] <= p['maximum_evaluation_forwards']
executed = dict(p['budget'])
executed['evaluation_forwards'] = d['evaluation_forwards']
executed['native_root_forwards'] += d['evaluation_forwards']
assert d['executed_budget'] == executed and d['native_root_forwards'] <= p['maximum_native_root_forwards']
for k, v in executed.items():
    assert d[k] == v
assert d['manual_attempts'] == d['manual_passes'] == 26
history_path = A / 'snapshot${ARTIFACT_ROOT}/codex_compiled_swiglu_secant_pilot_20260907_v1/results.json'
assert sha(history_path) == p['required_pilot_sha256']
history = json.loads(history_path.read_text())
assert history['status'] == 'complete' and d['dtype'] == history['dtype'] == 'torch.float16'
assert d['torch_version'] == history['torch_version']
assert p['official_normalized_sources'] == history['protocol']['official_normalized_sources']
assert p['official_evaluation_backend'] == history['protocol']['official_evaluation_backend']
unique_parent_curves = set()
expected = {(dataset, i) for dataset, indices in p['selection'].items() for i in indices}
assert len(d['records']) == 3 and {(r['dataset'], r['idx']) for r in d['records']} == expected
assert len(d['native_attempt_costs']) == 26 and all((x['completed'] for x in d['native_attempt_costs']))
assert sum((x['production'] for x in d['native_attempt_costs'])) == 14
for x in d['native_attempt_costs']:
    c = x['cost']
    assert c['native_forwards'] == 1 and c['native_forward_trajectories'] == 2 and (c['vjps'] == 0)
    assert c['native_decoder_layer_calls'] == 72 and c['native_decoder_layer_trajectories'] == 144
    assert c['extra_replay_calls'] == 36 and c['extra_replay_trajectories'] == 72
original16_path = A / 'snapshot${ARTIFACT_ROOT}/codex_qwen_secant_checkpoint_development_20260906_v1/results.json'
assert sha(original16_path) == 'a5f659c9cefa79e99e0739beeb0cdc6854cef7eafdd25b2d857ba2a1b688a3a9'
original16 = json.loads(original16_path.read_text())
import subprocess, sys
subprocess.run([sys.executable, str(A / 'verify_secant_boundary_source_20260907.py')], check=True)
out = {'status': 'verified_complete', 'raw_sha256': sha(folder / 'results.json'), 'budget': executed, 'base_budget': p['budget'], 'reused_curves': 0, 'fresh_curves': 0, 'curves_verified': 0, 'max_metric_reconstruction_error': 0.0, 'checked_ledger_rows': 0, 'cases': [], 'scope': 'Original NI0/NI2/MH0. Fuse original attribution expression boundaries using installed compiler: midpoint/scaled probability plus audit/GQA+RoPE/norm residual. No native model/FA/GEMM changes. Source math and actual annotated compiled/native dispatch independently checked, not bitwise gate. Full actual end-to-end costs and original30 curve arithmetic/provenance retained. Per-op ledger explicitly uncomputedNone.3developmentcases not full16/248/sign/independent victory.', 'profiles': []}
helper = ast.parse((A / 'verify_native_output_contrast_development16_20260906.py').read_text())
functions = [n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name in ['area', 'metrics']]
metric_function = next((n for n in functions if n.name == 'metrics'))
assert isinstance(metric_function.body[-2], ast.Assign) and ast.unparse(metric_function.body[-2]).startswith("cost = row['evaluation_costs'][name]")
assert isinstance(metric_function.body[-1], ast.Assert)
metric_function.body = metric_function.body[:-2]
ns = {'np': np, 'math': math, 'out': out}
exec(compile(ast.Module(body=functions, type_ignores=[]), 'independent_original_metrics', 'exec'), ns)
for row in d['records']:
    assert row['memory_reference']['native_root_forwards'] == row['memory_reference']['native_vjps'] == 1
    assert row['memory_reference']['backend'] == 'flash_attention_2' and row['memory_reference']['parameter_gradients_enabled'] is False
    assert row['eligible_positions'] == [row['user_positions'][j] for j in row['keep_local_indices']]
    assert row['complete'] and row['native_complete'] and (len(row['candidate_repeats']) == len(row['baseline_repeats']) == 4)
    payload = {k: v for k, v in row.items() if k != 'record_digest_sha256'}
    assert hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == row['record_digest_sha256']
    old = next((x for x in history['records'] if (x['dataset'], x['idx']) == (row['dataset'], row['idx'])), None)
    identity = old if old is not None else next((x for x in original16['records'] if (x['dataset'], x['idx']) == (row['dataset'], row['idx'])))
    for k in ['input_ids', 'prompt_len', 'user_positions', 'keep_local_indices']:
        assert row[k] == identity[k]
    if old is not None:
        assert row['eligible_positions'] == old['eligible_positions']
    assert set(row['scores']) == set(row['metrics']) == set(p['native_methods'] + p['baselines'])
    name = p['native_methods'][0]
    selected = row['native'][name]
    base = row['baseline_repeats'][-1]
    ref = np.array(base['signed_full_sequence'])
    value = np.array(selected['signed_full_sequence'])
    eligible = row['eligible_positions']
    assert selected['signed_full_sequence'] == row['candidate_repeats'][-1]['signed_full_sequence']
    assert np.array_equal(np.maximum(value[row['user_positions']], 0).astype(np.float32), np.array(row['scores'][name], dtype=np.float32))
    diagnostics = []
    for branch in ['baseline', 'candidate']:
        for r in row[branch + '_repeats']:
            c = r['end_to_end_cost']
            assert c['native_forwards'] == 1 and c['native_forward_trajectories'] == 2 and (c['vjps'] == 0)
            assert c['native_decoder_layer_calls'] == 72 and c['native_decoder_layer_trajectories'] == 144
            assert c['extra_replay_calls'] == 36 and c['extra_replay_trajectories'] == 72
            assert r['preparation_host_seconds_this_call'] > 0
            assert r['seconds'] >= r['propagation_internal_seconds'] > 0
            assert r['per_operator_ledger_collected'] is False
            assert all((math.isfinite(v) for v in r['endpoint_scores32'].values()))
            assert r['target_delta_score32_sum64'] == r['endpoint_scores32']['after'] - r['endpoint_scores32']['before']
            if branch in ['candidate', 'baseline']:
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
    ft = statistics.median((x['seconds'] for x in row['ft_both1_repeats']))
    assert len(row['ft_both1_repeats']) == 3
    for fc in [row['ft_both1_warmup']] + row['ft_both1_repeats'] + list(row['ft_costs'].values()) + [row['legacy_joint_cost']]:
        assert fc['native_forwards'] == fc['native_forward_trajectories'] == 1 and fc['vjps'] == 0
        assert fc['native_decoder_layer_calls'] == fc['native_decoder_layer_trajectories'] == 36
        assert fc['extra_replay_calls'] == fc['extra_replay_trajectories'] == 0
    costs = {}
    for branch in ['baseline', 'candidate']:
        runs = row[branch + '_repeats']
        warm = runs[1:]
        elapsed = statistics.median((r['comparable_attribution_seconds'] for r in warm))
        peak = max((r['peak_allocated_bytes'] for r in warm))
        costs[branch] = {'median_seconds': elapsed, 'peak_bytes': peak, 'FT_ratio': elapsed / ft, 'peak_above_ordinary_FA_bytes': peak - row['memory_reference']['peak_allocated_bytes'], 'first_complete_seconds': runs[0]['comparable_attribution_seconds'], 'individual_complete_seconds': [r['comparable_attribution_seconds'] for r in runs]}
    assert selected['comparable_attribution_seconds'] == costs['candidate']['median_seconds'] and selected['peak_allocated_bytes'] == costs['candidate']['peak_bytes']
    assert old is not None
    assert history['checkpoint_before'] == d['checkpoint_before'] and history['native_sources_before'] == d['native_sources_before']
    for method in row['scores']:
        provenance = row['evaluation_provenance'][method]
        cost = row['evaluation_costs'][method]
        parent_method = p['parent_method_map'].get(method, method)
        if provenance['mode'] == 'identical_score_parent_curve':
            assert row['scores'][method] == old['scores'][parent_method]
            assert provenance['source'] == p['required_pilot'] and provenance['source_sha256'] == p['required_pilot_sha256'] and (provenance['source_method'] == parent_method)
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
            assert provenance['mode'] == 'fresh_original_curve' and row['scores'][method] != old['scores'][parent_method]
            assert cost['native_forwards'] == 21 and cost['native_decoder_layer_calls'] == 756 and (cost['vjps'] == 0)
            out['fresh_curves'] += 1
        ns['metrics'](row, method)
    assert sum((x['native_forwards'] for x in row['evaluation_costs'].values())) == 21 * sum((x['mode'] == 'fresh_original_curve' for x in row['evaluation_provenance'].values()))
    if 'profile' in row:
        f = folder / row['profile']['trace']
        assert sha(f) == row['profile']['sha256']
        trace = json.loads(f.read_text())
        names = [e['name'] for e in trace['traceEvents'] if e.get('cat') == 'kernel']
        assert sum(('flash_fwd_kernel' in n for n in names)) == 72 and (not any(('flash' in n.lower() and 'bwd' in n.lower() for n in names)))
        annotations = [e for e in trace['traceEvents'] if e.get('name') == 'ATTR_NATIVE_HALF_LINEAR' and e.get('ph') == 'X' and (e.get('cat') == 'user_annotation')]
        assert len(annotations) == 253
        cpu = {e.get('args', {}).get('External id'): e for e in trace['traceEvents'] if e.get('cat') == 'cpu_op' and e.get('args', {}).get('External id') is not None}
        projection_kernels = []
        unlinked = []
        coverage = set()
        for e in trace['traceEvents']:
            if e.get('cat') != 'kernel' or 'gemm' not in e['name'].lower():
                continue
            host = cpu.get(e.get('args', {}).get('External id'))
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
        profile_dispatch = {'dataset': row['dataset'], 'idx': row['idx']}
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
        for label, expected in [('ATTR_COMPILED_MIDPOINT', 144), ('ATTR_COMPILED_SCALED_PROBABILITY', 72), ('ATTR_COMPILED_ATTENTION_LAYOUT', 36), ('ATTR_COMPILED_NORM_RESIDUAL_TWO', 36), ('ATTR_COMPILED_NORM_RESIDUAL_THREE', 36)]:
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
        out['profiles'].append(profile_dispatch)
    fields = ['rise', 'mas', 'recovery']
    metrics = {m: {k: v[k] for k in fields} for m, v in row['metrics'].items()}
    out['cases'].append({'dataset': row['dataset'], 'idx': row['idx'], 'costs': costs, 'fresh_FT_seconds': ft, 'warm_time_reduction_fraction': 1 - costs['candidate']['median_seconds'] / costs['baseline']['median_seconds'], 'metrics': metrics, 'previous_accepted_metrics': None if old is None else {k: old['metrics']['strong_secant_compiled_swiglu'][k] for k in fields}, 'full_sequence_length': len(row['input_ids']), 'numerical_review': selected['numerical_review'], 'numerical_diagnostics': diagnostics, 'full_vector_identical_to_same_job_baseline': selected['signed_full_sequence'] == base['signed_full_sequence'], 'current_score_identical_to_parent': row['scores'][name] == old['scores']['strong_secant_compiled_swiglu'], 'endpoint_score_delta_to_baseline': {side: selected['endpoint_scores32'][side] - base['endpoint_scores32'][side] for side in ['before', 'after']}, 'endpoint_logprob_relative_L2_to_baseline': None, 'endpoint_logprob_artifact_scope': 'This timing harness retained actual endpoint sums but did not export per-target-token endpoint logprob arrays; no independent current-array check claimed.', 'max_ledger_absolute_difference': None, 'global_signed_sum_delta_to_baseline': selected['signed_sum'] - base['signed_sum'], 'global_unassigned_delta_to_baseline': selected['unassigned_total'] - base['unassigned_total'], 'time_gate_pass': costs['candidate']['FT_ratio'] <= 1, 'memory_gate_pass': costs['candidate']['peak_above_ordinary_FA_bytes'] <= p['memory_allowance_bytes']})
assert out['curves_verified'] == 30 and out['checked_ledger_rows'] == 0 and (len(out['profiles']) == 2)
assert out['reused_curves'] == d['reused_quality_curves'] and out['fresh_curves'] == d['fresh_quality_curves']
out['unique_parent_curves_reused'] = len(unique_parent_curves)
out['physical_decoder_calls'] = 36 * executed['native_root_forwards'] + executed['extra_layer_replay_calls']
out['endpoint_decoder_trajectories'] = 36 * (executed['native_attribution_endpoint_trajectories'] + executed['ordinary_reference_forwards'] + executed['ft_attribution_forwards'] + executed['evaluation_forwards']) + executed['extra_layer_replay_endpoint_trajectories']
out['all_three_time_gates'] = all((r['time_gate_pass'] for r in out['cases']))
out['all_three_memory_gates'] = all((r['memory_gate_pass'] for r in out['cases']))
out['production_calls_without_ledger'] = 26
out['diagnostic_calls_with_ledger'] = 0
(A / 'secant_boundary_summary_20260907.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k not in ['cases', 'profiles']}, indent=2))
for row in out['cases']:
    print(row['dataset'], row['idx'], row['costs'], 'vectors_equal', row['full_vector_identical_to_same_job_baseline'], 'time_gate', row['time_gate_pass'], 'memory_gate', row['memory_gate_pass'])
