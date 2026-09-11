"""Check pilot forward controls against previously completed paired GPU runs."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    pilot = HERE / 'recall_dev_v1'
    parent = HERE / 'focus_v1_completed_2'
    r = json.loads((pilot / 'results.json').read_bytes())
    assert r['status'] == 'complete' and r['stage'] == 'development'
    vectors = np.load(pilot / 'vectors.npz')
    controls = {}
    for task in ('vt_h4_c1', 'hotpotqa_long'):
        original = json.loads((parent / task / 'results.json').read_bytes())
        original_vectors = np.load(parent / task / 'vectors.npz')
        assert original['status'] == 'complete'
        rows = {x['index']: x for x in original['cases']}
        comparisons = []
        for row in r['cases']:
            if row['dataset'] != task:
                continue
            previous = rows[row['index']]
            assert row['input_sha256'] == previous['input_sha256']
            key = f"{task}_{row['index']}_"
            for method, old in [('DT_body_forward_signed_full', 'DT_signed_full'),
                                ('DT_full_forward_signed_full', 'DT_full_reference_signed_full'),
                                ('FT_K1_prompt', 'FT_K1_prompt'), ('FT_K3_prompt', 'FT_K3_prompt')]:
                a, b = vectors[key + method], original_vectors[key + old]
                comparisons.append({'index': row['index'], 'method': method,
                    'bitwise_equal': bool(np.array_equal(a, b)), 'max_absolute_difference': float(np.max(np.abs(a - b)))})
        controls[task] = {'parent_results_sha256': hashlib.sha256((parent / task / 'results.json').read_bytes()).hexdigest(),
            'comparisons': comparisons, 'all_bitwise_equal': all(x['bitwise_equal'] for x in comparisons)}
    deltas = []
    for row in r['cases']:
        for ref in ('body', 'full'):
            f, rev = (row[f'DT_{ref}_{direction}_details'] for direction in ('forward', 'reverse'))
            deltas.append({'dataset': row['dataset'], 'index': row['index'], 'reference': ref,
                'forward_plus_reverse_native_delta': f['target_delta_score32_sum64'] + rev['target_delta_score32_sum64']})
    out = {'status': 'checked', 'control_vectors': controls,
        'endpoint_exchange_native_delta_checks': deltas,
        'max_exchange_delta_mismatch': max(abs(x['forward_plus_reverse_native_delta']) for x in deltas)}
    path = HERE / 'recall_development/control_verification.json'
    path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'all_forward_controls_bitwise_equal': all(x['all_bitwise_equal'] for x in controls.values()),
        'max_exchange_delta_mismatch': out['max_exchange_delta_mismatch']}))


if __name__ == '__main__':
    main()
