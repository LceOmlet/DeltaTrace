"""Independent JSON/source audit; no raw-tensor, model or GPU execution."""
import ast
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent / 'snapshot${ARTIFACT_ROOT}/codex_dt_normgate_repair_NI1_20260908_v1'
OUTPUT = Path(__file__).with_suffix('.json')
KEY = 'niah_mq_q2_1'
COMPONENTS = ('CPU_reference_baseline_content_mismatch', 'runtime_baseline_content_numerics',
              'B2_beta1_minus_B1_clean_beta_transfer', 'mathematical_write_minus_native_write_delta')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit_path, source_path = BASE / 'fla_beta_content_mismatch_audit.json', BASE / 'results.json'
    script_path = BASE / 'fla_beta_content_mismatch_audit_20260908.py'
    a = json.loads(audit_path.read_text())
    r = json.loads(source_path.read_text())
    receipt = json.loads((BASE / 'terminal_receipt.json').read_text())
    ac, rc = a['cases'][KEY], r['cases'][KEY]
    repo_script = BASE.parents[3] / 'DeltaTrace/research/reproduction_templates' / script_path.name
    assert a['status'] == 'CPU_local_write_audit_complete'
    assert a['self_sha256'] == sha(script_path) == sha(repo_script)
    assert a['source_results_sha256'] == a['source_results_sha256_after'] == sha(source_path)
    assert a['source_results_bytes'] == source_path.stat().st_size
    artifact = ac['input_artifact']
    assert artifact == rc['control_content_artifact']
    if 'private_artifacts' in receipt:
        assert artifact == receipt['private_artifacts']['control_content_artifact']
    else:
        assert {k: artifact[k] for k in ('sha256', 'bytes')} == receipt['files'][artifact['file']]
    assert ac['input_artifact_sha256_before'] == ac['input_artifact_sha256_after'] == artifact['sha256']
    assert ac['input_sha256'] == rc['input']['input_sha256']
    assert ac['scoring_input_receipts'] == {s: rc['scoring_points'][s]['input_receipt'] for s in ('0', '1', '10', '20')}
    verified, missing = [], []
    for name, value in receipt['files'].items():
        path = BASE / name
        if path.is_file():
            assert path.stat().st_size == value['bytes'] and sha(path) == value['sha256'], name
            verified.append(name)
        else:
            missing.append(name)
    max_error, steps = 0., {}
    for step, record in ac['steps'].items():
        sizes = record['token_group_sizes']
        assert sum(sizes.values()) == rc['input']['total_length']
        deleted = len(set(rc['scoring_points'][step]['input_receipt']['deleted_positions']))
        assert sizes['deleted'] == deleted and sizes['kept'] == len(rc['input']['keep']) - deleted
        steps[step] = {}
        for label, projection in record['projections'].items():
            terms = projection['terms']
            assert len(projection['per_head']) == ac['packed_layout']['H']
            checks = [terms['local_write_prediction'] - terms['local_native_write_effect'] - terms['local_prediction_minus_native_effect'],
                      terms['local_prediction_minus_native_effect'] - sum(terms[n] for n in COMPONENTS)]
            for name, value in terms.items():
                checks.extend([sum(g[name] for g in projection['token_groups'].values()) - value,
                               sum(h[name] for h in projection['per_head']) - value])
            for part in list(projection['token_groups'].values()) + projection['per_head']:
                checks.append(part['local_prediction_minus_native_effect'] - sum(part[n] for n in COMPONENTS))
            checks.extend(projection['checks'].values())
            maximum = max(abs(value) for value in checks)
            assert maximum < 1e-7
            max_error = max(max_error, maximum)
            full = -rc['scoring_points'][step][label + '_layer0_decomposition']['terms']['GDN_FLA_including_raw_g_exp']
            head_beta = [h[COMPONENTS[0]] for h in projection['per_head']]
            steps[step][label] = {
                'local_terms': terms, 'complete_FLA_prediction_minus_actual': full,
                'complete_FLA_minus_local_write_arithmetic_remainder': full - terms['local_prediction_minus_native_effect'],
                'theoretical_beta_mismatch_head_positive_sum': sum(max(x, 0) for x in head_beta),
                'theoretical_beta_mismatch_head_negative_sum': sum(min(x, 0) for x in head_beta),
                'theoretical_beta_mismatch_head_max_absolute': max(abs(x) for x in head_beta),
                'max_recomputed_aggregate_error': maximum}
            assert all(projection['token_groups'][g][COMPONENTS[0]] == 0 for g in ('kept', 'other_prompt', 'response'))
            assert terms['B2_beta1_minus_B1_clean_beta_transfer'] == 0
    for stats in ac['native_write_reconstruction_residuals'].values():
        assert abs(math.sqrt(stats['sum_squared_residual'] / stats['sum_squared_native_write']) - stats['relative_L2']) < 1e-15
    layout = ac['packed_layout']
    assert ac['calls']['CPU_chunk_readout_evaluations'] == 5 * layout['H'] * layout['N']
    assert ac['calls']['CPU_bmm_calls'] == 3 * ac['calls']['CPU_chunk_readout_evaluations']
    assert all(a[k] == 0 for k in ('GPU_calls', 'model_calls', 'native_operator_calls', 'parameter_searches'))
    tree = ast.parse(script_path.read_text())
    readout = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'readout_reference')
    bmm_count = sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'bmm' for n in ast.walk(readout))
    assert bmm_count == 3
    summary = {
        'status': 'verified_saved_aggregate_audit_with_raw_tensor_limitation',
        'case': KEY, 'fixed_steps': ['1', '10', '20'], 'sign_convention': 'prediction_minus_actual',
        'source_directory': str(BASE),
        'sha256': {'audit_result': sha(audit_path), 'audit_script': sha(script_path), 'source_results': sha(source_path)},
        'identity_checks': {
            'audit_source_before_after_match': True, 'audit_script_matches_frozen_repo_script': True,
            'scoring_input_receipts_match': True, 'input_artifact_receipts_and_reported_before_after_match': True,
            'local_terminal_receipt_files_verified': verified, 'receipt_files_not_present_locally': missing,
            'raw_input_artifact_reloaded_by_this_independent_review': False,
            'raw_input_artifact_locally_present': (BASE / artifact['file']).is_file()},
        'max_recomputed_aggregate_error': max_error, 'steps': steps,
        'native_write_reconstruction_residuals': ac['native_write_reconstruction_residuals'],
        'current_candidate_packed_checks': ac['current_candidate_packed_checks'],
        'cpu_execution': {'seconds': a['CPU_job_seconds'], **ac['calls'], 'GPU_calls': 0, 'model_calls': 0},
        'no_shadow_state_source_review': {
            'readout_BMM_calls_per_tile': bmm_count, 'uses_saved_chunk_start_h_and_native_v_new': True,
            'chunk_width_limit': 64, 'no_new_state_trajectory_or_per_sequence_T_by_T_matrix': True,
            'readout_is_CPU64_algebra_reference_not_observed_native_internal_r': True},
        'conclusions': [
            'In NI1 step10, beta content mismatch and complete local-write error are negative under both actual projections; they cannot explain the dominant positive complete FLA mismatch.',
            'The nonzero beta interaction is at deleted update locations. These groups are not original source-token attributions.',
            'Current/candidate r0 is reported bitwise equal; L changes. This does not establish a repaired FLA update rule.',
            'Native-write reconstruction residuals and runtime c0 discrepancies are explicitly included; their projected values do not reverse the step10 conclusion.',
            'Complete FLA minus local write is only an arithmetic remainder, not a proven read, forget or inherited-content source contribution.',
            'This review recomputes saved aggregate algebra and checks source/receipts. It does not independently rerun raw-tensor BMMs on the remote 496MB capture.'
        ],
        'next_diagnostic_if_needed': {
            'boundary': 'r = k^T C, with lambda_r = -beta1 * actual_L',
            'reason': 'Adjacent content-read interaction; already saved chunk states and the same adjoint suffice, with no new model run.',
            'primary_projection': 'sum delta_k dot ((C0-CA) lambda_r)',
            'B2_B1_transfer': 'sum (k1-k_clean) dot ((C_clean-CA) lambda_r)',
            'chunk_algebra': 'Cj*lambda_r = exp(Gj)*(lambda_r @ Hj^T) + ((lambda_r @ Uj^T)*E_j_strict_lower) @ Kj; three CPU BMMs per source tile, no state trajectory.',
            'limit': 'Local read-boundary algebra, not the entire inherited beta1*(delta_chat-delta_c). Earlier state/forget errors remain distinct; separately bound any next CPU execution.'
        }
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'saved': str(OUTPUT), 'max_error': max_error, 'sha256': sha(OUTPUT),
                      'step10': {k: {'full': v['complete_FLA_prediction_minus_actual'], 'beta': v['local_terms'][COMPONENTS[0]],
                                     'local': v['local_terms']['local_prediction_minus_native_effect']}
                                 for k, v in steps['10'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
