"""CPU-only local write-gate audit on actual saved finite FLA captures.

This is chunk algebra on native states, never another state trajectory or model
forward. L/r0 are the actual packed production tensors. Reconstructed readouts
are labelled CPU64 references and checked against native v_new = beta*(v-r).
The local write prediction excludes inherited c=v-r propagation errors; it is
neither a complete input attribution nor an extra term to add to a recursive
error total. A changed norm gate changes the target projection L, not this FLA
allocation rule. No metric, repair, timing or native internal-readout claim.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MACA_VISIBLE_DEVICES'] = '-1'

import hashlib
import json
import math
import signal
import sys
import time
import traceback
from pathlib import Path

import torch


CASE = 'niah_mq_q2_1'
STEPS = ('1', '10', '20')
C = 64
CPU_WALL_SECONDS = 180
CLOSURE_LIMIT = 1e-7
FIELDS = ('q', 'k', 'v', 'beta', 'raw_g', 'g', 'h', 'v_new', 'o')
TERM_NAMES = (
    'local_write_prediction', 'local_native_write_effect',
    'local_prediction_minus_native_effect',
    'CPU_reference_baseline_content_mismatch',
    'runtime_baseline_content_numerics',
    'B2_beta1_minus_B1_clean_beta_transfer',
    'mathematical_write_minus_native_write_delta',
)
COMPONENTS = TERM_NAMES[3:]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cpu'
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(child) for child in value.values())
    return 0


def validate_endpoint(e, batch, length, heads, key_dim, value_dim):
    assert all(name in e for name in FIELDS)
    for name in FIELDS:
        value = e[name]
        assert isinstance(value, torch.Tensor) and value.device.type == 'cpu'
        if name == 'h':
            shape = (batch, (length + C - 1) // C, heads, key_dim, value_dim)
        elif name in ('beta', 'raw_g', 'g'):
            shape = (batch, length, heads)
        else:
            shape = (batch, length, heads, key_dim if name in ('q', 'k') else value_dim)
        assert tuple(value.shape) == shape, (name, tuple(value.shape), shape)


def token(e, name, row, head, start, end):
    value = e[name][row, start:end, head].double()
    assert bool(torch.isfinite(value).all()), name
    return value


def readout_reference(e, row, head, chunk, start, end, counts):
    """The existing r0 formula: three CPU64 BMMs, one tile, no state update."""
    k = token(e, 'k', row, head, start, end).unsqueeze(0)
    u = token(e, 'v_new', row, head, start, end).unsqueeze(0)
    g = token(e, 'g', row, head, start, end).unsqueeze(0)
    state = e['h'][row, chunk, head].double().unsqueeze(0)
    assert bool(torch.isfinite(state).all())
    assert bool((g[:, 1:] - g[:, :-1] <= 1e-5).all()), 'Nonmonotone native cumulative decay'
    width = end - start
    lower = torch.ones((width, width), dtype=torch.bool, device='cpu').tril(-1)
    decay = (g[:, :, None] - g[:, None, :]).clamp_max(0).exp() * lower
    result = g.exp()[..., None] * torch.bmm(k, state)
    result = result + torch.bmm(torch.bmm(k, k.transpose(-1, -2)) * decay, u)
    assert bool(torch.isfinite(result).all())
    counts['CPU_chunk_readout_evaluations'] += 1
    counts['CPU_bmm_calls'] += 3
    return result[0]


def new_residual_stats():
    return {'elements': 0, 'sum_squared_residual': 0., 'sum_squared_native_write': 0.,
            'max_absolute_residual': 0.}


def record_write_residual(stats, e, row, head, start, end, content):
    native = token(e, 'v_new', row, head, start, end)
    beta = token(e, 'beta', row, head, start, end)
    residual = native - beta[:, None] * content
    stats['elements'] += residual.numel()
    stats['sum_squared_residual'] += float(residual.square().sum())
    stats['sum_squared_native_write'] += float(native.square().sum())
    stats['max_absolute_residual'] = max(stats['max_absolute_residual'], float(residual.abs().max()))


def finish_residual_stats(stats):
    denominator = stats['sum_squared_native_write']
    stats['relative_L2'] = math.sqrt(stats['sum_squared_residual'] / denominator) if denominator > 0 else None
    stats['zero_native_write_denominator'] = denominator == 0
    return stats


def audit_case(case, captures):
    retained_bytes = tensor_bytes(captures)
    paired, points = captures['paired'], captures['points']
    assert set(points) == {'0', *STEPS}
    B, T, H, V = points['0']['v'].shape
    K = points['0']['k'].shape[-1]
    assert B == 1 and K == V == 128
    N = (T + C - 1) // C
    validate_endpoint(paired, 2, T, H, K, V)
    for value in points.values():
        validate_endpoint(value, 1, T, H, K, V)
    scale = float(captures['scale'])
    assert math.isfinite(scale) and scale > 0
    diag = {}
    for label in ('current', 'candidate'):
        assert set(captures[label]) == {'L', 'r0'}
        diag[label] = {}
        for name in ('L', 'r0'):
            value = captures[label][name]
            assert value.shape == (B * H * N, C, V) and value.dtype == torch.float32
            assert bool(torch.isfinite(value).all())
            diag[label][name] = value.reshape(B, H, N, C, V)
    P = case['input']['prompt_length']
    keep = set(case['input']['keep'])
    assert case['input']['total_length'] == T and 0 < P <= T
    assert all(0 <= position < P for position in keep)
    groups = {}
    for step in STEPS:
        deleted = set(case['scoring_points'][step]['input_receipt']['deleted_positions'])
        assert deleted <= keep
        groups[step] = {'deleted': sorted(deleted), 'kept': sorted(keep - deleted),
                        'other_prompt': sorted(set(range(P)) - keep), 'response': list(range(P, T))}
        assert sum(len(x) for x in groups[step].values()) == T
    totals = {step: {label: {name: torch.zeros((T, H), dtype=torch.float64, device='cpu')
                            for name in TERM_NAMES} for label in diag} for step in STEPS}
    residuals = {name: new_residual_stats() for name in ('B2_EOS', 'B1_clean', *STEPS)}
    counts = {'CPU_chunk_readout_evaluations': 0, 'CPU_bmm_calls': 0}
    max_closure = {step: {label: 0. for label in diag} for step in STEPS}
    for head in range(H):
        for chunk in range(N):
            start, end = chunk * C, min((chunk + 1) * C, T)
            width = end - start
            r0_ref = readout_reference(paired, 0, head, chunk, start, end, counts)
            rc = readout_reference(points['0'], 0, head, chunk, start, end, counts)
            v0 = token(paired, 'v', 0, head, start, end)
            c0_ref = v0 - r0_ref
            cc = token(points['0'], 'v', 0, head, start, end) - rc
            beta1 = token(paired, 'beta', 1, head, start, end)
            bc = token(points['0'], 'beta', 0, head, start, end)
            uc = token(points['0'], 'v_new', 0, head, start, end)
            record_write_residual(residuals['B2_EOS'], paired, 0, head, start, end, c0_ref)
            record_write_residual(residuals['B1_clean'], points['0'], 0, head, start, end, cc)
            for step in STEPS:
                point = points[step]
                ra = readout_reference(point, 0, head, chunk, start, end, counts)
                ca = token(point, 'v', 0, head, start, end) - ra
                ba = token(point, 'beta', 0, head, start, end)
                ua = token(point, 'v_new', 0, head, start, end)
                record_write_residual(residuals[step], point, 0, head, start, end, ca)
                dc, db, du = cc - ca, bc - ba, uc - ua
                mathematical_minus_native = bc[:, None] * cc - ba[:, None] * ca - du
                for label, values in diag.items():
                    L = values['L'][0, head, chunk, :width].double()
                    c0_runtime = v0 - values['r0'][0, head, chunk, :width].double()
                    def project(value):
                        return (L * value).sum(-1)
                    prediction = project(beta1[:, None] * dc + c0_runtime * db[:, None])
                    actual = project(du)
                    terms = {
                        'local_write_prediction': prediction,
                        'local_native_write_effect': actual,
                        'local_prediction_minus_native_effect': prediction - actual,
                        'CPU_reference_baseline_content_mismatch': project((c0_ref - ca) * db[:, None]),
                        'runtime_baseline_content_numerics': project((c0_runtime - c0_ref) * db[:, None]),
                        'B2_beta1_minus_B1_clean_beta_transfer': project((beta1 - bc)[:, None] * dc),
                        'mathematical_write_minus_native_write_delta': project(mathematical_minus_native),
                    }
                    closure = terms['local_prediction_minus_native_effect'] - sum(terms[name] for name in COMPONENTS)
                    maximum = float(closure.abs().max())
                    max_closure[step][label] = max(max_closure[step][label], maximum)
                    assert maximum < CLOSURE_LIMIT, (step, label, head, chunk, maximum)
                    for name, value in terms.items():
                        assert bool(torch.isfinite(value).all()), (step, label, name)
                        totals[step][label][name][start:end, head] = value
    result = {'steps': {}, 'scale_recorded_not_applied_again': scale,
              'packed_layout': {'B': B, 'T': T, 'H': H, 'N': N, 'C': C, 'K': K, 'V': V},
              'capture_tensor_bytes': retained_bytes,
              'term_array_CPU_bytes': tensor_bytes(totals), 'calls': counts,
              'native_write_reconstruction_residuals': {k: finish_residual_stats(v) for k, v in residuals.items()},
              'current_candidate_packed_checks': {}}
    for name in ('L', 'r0'):
        left, right = captures['current'][name], captures['candidate'][name]
        result['current_candidate_packed_checks'][name] = {
            'bitwise_equal': bool(torch.equal(left, right)),
            'max_absolute_difference': float((left.double() - right.double()).abs().max())}
    for step in STEPS:
        result['steps'][step] = {'token_group_sizes': {k: len(v) for k, v in groups[step].items()}, 'projections': {}}
        for label, arrays in totals[step].items():
            values = {name: float(value.sum()) for name, value in arrays.items()}
            head_values = [{name: float(value[:, h].sum()) for name, value in arrays.items()} for h in range(H)]
            grouped = {group: {name: float(value[indices].sum()) for name, value in arrays.items()}
                       for group, indices in groups[step].items()}
            closure = values['local_prediction_minus_native_effect'] - sum(values[name] for name in COMPONENTS)
            head_closure = max(abs(h['local_prediction_minus_native_effect'] - sum(h[k] for k in COMPONENTS)) for h in head_values)
            group_closure = max(abs(g['local_prediction_minus_native_effect'] - sum(g[k] for k in COMPONENTS)) for g in grouped.values())
            group_sum = max(abs(sum(g[name] for g in grouped.values()) - value) for name, value in values.items())
            head_sum = max(abs(sum(h[name] for h in head_values) - value) for name, value in values.items())
            checks = {'total_closure': closure, 'max_head_closure': head_closure, 'max_group_closure': group_closure,
                      'max_group_sum_difference': group_sum, 'max_head_sum_difference': head_sum,
                      'max_token_head_closure': max_closure[step][label]}
            assert all(abs(v) < CLOSURE_LIMIT for v in checks.values()), (step, label, checks)
            result['steps'][step]['projections'][label] = {'terms': values, 'per_head': head_values,
                                                         'token_groups': grouped, 'checks': checks}
    return result


def main():
    assert len(sys.argv) == 2, 'Supply only the experiment directory.'
    directory = Path(sys.argv[1]).resolve()
    source = directory / 'results.json'
    output = directory / 'fla_beta_content_mismatch_audit.json'
    assert source.is_file() and not output.exists(), 'Require original results; refuse to overwrite this audit.'
    started = time.perf_counter()
    r = {'status': 'starting', 'self_sha256': digest(Path(__file__)), 'source_results_sha256': digest(source),
         'source_results_bytes': source.stat().st_size, 'fixed_case': CASE, 'fixed_steps': list(STEPS),
         'model_calls': 0, 'GPU_calls': 0, 'native_operator_calls': 0, 'parameter_searches': 0,
         'CPU_wall_budget_seconds': CPU_WALL_SECONDS, 'sign_convention': 'prediction_minus_actual',
         'cases': {}, 'skipped_cases': {}}
    def save():
        temporary = output.with_suffix('.partial')
        temporary.write_text(json.dumps(r, indent=2, allow_nan=False))
        temporary.replace(output)
    def timeout(*_args):
        raise TimeoutError('Frozen CPU algebra budget exceeded.')
    try:
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout)
            signal.alarm(CPU_WALL_SECONDS)
        torch.set_num_threads(4)
        torch.set_default_device('cpu')
        original = json.loads(source.read_bytes())
        r['source_experiment_status'] = original.get('status')
        case = original.get('cases', {}).get(CASE)
        complete = case is not None and 'control_content_artifact' in case and all(
            step in case.get('scoring_points', {}) and 'input_receipt' in case['scoring_points'][step]
            for step in ('0', *STEPS))
        for key in original.get('cases', {}):
            if key != CASE:
                r['skipped_cases'][key] = 'Outside the frozen NI1 scope.'
        if not complete:
            r['skipped_cases'][CASE] = 'No completed capture receipt and four actual scoring inputs; no reconstruction or rerun.'
            r['status'] = 'skipped_no_completed_capture'
        else:
            receipt = case['control_content_artifact']
            path = (directory / receipt['file']).resolve()
            assert path.parent == directory and path.is_file()
            assert path.stat().st_size == receipt['bytes'] and digest(path) == receipt['sha256']
            captures = torch.load(path, map_location='cpu', weights_only=True)
            row = {'input_artifact': receipt, 'input_artifact_sha256_before': digest(path),
                   'input_sha256': case['input']['input_sha256'],
                   'scoring_input_receipts': {s: case['scoring_points'][s]['input_receipt'] for s in ('0', *STEPS)}}
            r['cases'][CASE] = row
            save()
            row.update(audit_case(case, captures))
            row['input_artifact_sha256_after'] = digest(path)
            assert row['input_artifact_sha256_before'] == row['input_artifact_sha256_after']
            del captures
            r['status'] = 'CPU_local_write_audit_complete'
        r['source_results_sha256_after'] = digest(source)
        assert r['source_results_sha256_after'] == r['source_results_sha256']
    except Exception:
        r['status'] = 'failed'
        r['error'] = traceback.format_exc()
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        r['CPU_job_seconds'] = time.perf_counter() - started
        r['readout_contract'] = 'CPU64 three-BMM r formula on actual native chunk h/k/g/v_new. No new state trajectory; no separately observed native r tensor.'
        r['projection_contract'] = 'Current/candidate use their own actual frozen L and runtime r0 on the same captured operands. Changed upstream projection does not establish a repaired FLA rule.'
        r['scope'] = 'Local beta*(v-r) mismatch only. Excludes inherited c propagation errors, complete input attribution, RISE/MAS and full recursive source sums. Do not add this local term again to a total already containing it.'
        r['group_contract'] = 'Groups label the update location, not the original input source of that update or a complete token attribution.'
        save()
        print(json.dumps({'status': r['status'], 'CPU_seconds': r['CPU_job_seconds'], 'cases': list(r['cases']),
                          'error': r.get('error')}), flush=True)
    return 1 if r['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
