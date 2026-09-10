"""Verify small GPU Recall experiments; freeze a choice on development only."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments/official'))
from evidence_protocol import source_span, select_source_tokens, recovery_curve, reference_token_ids
from retrieval_views import sentence_density_order
from analyze import paired_bootstrap, interval

TASKS = ['vt_h2_c3', 'vt_h4_c1', 'vt_h6_c1', 'vt_h10_c1', 'hotpotqa_long']
FRACTIONS = [.05, .1, .2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_tree(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for k, v in expected.items():
            same_tree(actual[k], v)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            same_tree(a, b)
    elif isinstance(expected, float):
        assert abs(actual - expected) < 1e-10, (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--publication', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--freeze-choice', action='store_true')
    a = p.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    split = json.loads((HERE / 'recall_split.json').read_bytes())
    r = json.loads((a.run / 'results.json').read_bytes())
    assert r['status'] == 'complete' and r['experiment'] == 'recall-pilot-v1'
    assert r['driver_sha256'] == digest(HERE / 'evaluate_recall.py')
    assert r['split_sha256'] == digest(HERE / 'recall_split.json')
    assert r['plan_sha256'] == split['plan_sha256'] == digest(HERE / 'RECALL_PILOT.md')
    assert r['vectors_sha256'] == digest(a.run / 'vectors.npz')
    assert r['generation_calls'] == 0 and r['ft_source'] == 'live'
    assert r['family'] == 'qwen3' and r['evaluation_protocol'] == 'source-v2'
    stage = r['stage']
    assert stage in ('development', 'validation')
    assert not a.freeze_choice or stage == 'development'
    tasks = ['vt_h4_c1', 'hotpotqa_long'] if stage == 'development' else TASKS
    assert set(r['selected_counts']) == set(tasks)
    assert r['selected_counts'] == {t: len(split['tasks'][t][stage]) for t in tasks}
    expected = {(t, i) for t in tasks for i in split['tasks'][t][stage]}
    assert {(row['dataset'], row['index']) for row in r['cases']} == expected
    assert len(r['cases']) == len(expected)
    vectors = np.load(a.run / 'vectors.npz')
    caches, published = {}, {}
    for t in tasks:
        path = a.data / (t + '.jsonl')
        assert digest(path) == split['tasks'][t]['cache_sha256']
        caches[t] = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        old = json.loads(gzip.decompress((a.publication / 'raw' / (t + '.results.json.gz')).read_bytes()))
        published[t] = {row['index']: row for row in old['cases']}
    case_rows, residuals = [], []
    for row in r['cases']:
        t, i = row['dataset'], row['index']
        old, cache = published[t][i], caches[t][i]
        assert row['status'] == 'complete'
        for field in ('input_ids', 'user_positions', 'gold', 'prompt_length', 'target_length'):
            assert row[field] == old[field], (t, i, field)
        assert row['author_keep'] == old['keep']
        text = ' ' + cache['prompt']
        enc = tokenizer.encode(text, add_special_tokens=False)
        ids, positions = np.asarray(row['input_ids'], dtype=np.int64), row['user_positions']
        assert hashlib.sha256(ids.tobytes()).hexdigest() == row['input_sha256']
        assert len(enc.ids) == len(positions)
        assert np.flatnonzero(np.asarray(enc.ids) != ids[positions]).tolist() == row['standalone_token_boundary_differences']
        span = source_span(t, cache['prompt'])
        assert span == row['source_span']
        keep = select_source_tokens(span, enc.offsets, row['author_keep'], row['gold'])
        assert keep == row['keep']
        assert not set(keep) & set(row['standalone_token_boundary_differences'])
        eos = int(ids[-1])
        for ref, selected in [('body', keep), ('full', row['author_keep'])]:
            baseline = np.asarray(reference_token_ids(ids, [positions[j] for j in selected], eos), dtype=np.int64)
            assert hashlib.sha256(baseline.tobytes()).hexdigest() == row['references'][ref]
        prefix = f'{t}_{i}_'
        for method, views in row['metrics'].items():
            if method.startswith('DT_'):
                signed = vectors[prefix + method + '_signed_full']
                assert signed.shape == ids.shape and np.isfinite(signed).all()
                scores = signed[positions].astype(np.float32)
                if method.endswith('_symmetric'):
                    ref = method.split('_')[1]
                    assert np.array_equal(signed, .5 * (vectors[prefix + 'DT_' + ref + '_forward_signed_full']
                                                       + vectors[prefix + 'DT_' + ref + '_reverse_signed_full']))
                else:
                    detail = row[method + '_details']
                    assert detail['pilot_reported_sign'] == (-1 if method.endswith('_reverse') else 1)
                    reported_sum = detail['pilot_reported_sign'] * detail['signed_sum']
                    assert np.isclose(signed.sum(), reported_sum, atol=1e-10)
                    residuals.append({'dataset': t, 'index': i, 'method': method,
                        'target_delta': detail['pilot_reported_sign'] * detail['target_delta_score32_sum64'],
                        'unassigned_residual': detail['pilot_reported_sign'] * detail['unassigned_total']})
            else:
                assert method in ('FT_K1', 'FT_K3')
                scores = vectors[prefix + method + '_prompt'].astype(np.float32)
            assert scores.shape == (len(positions),)
            positive = np.maximum(scores, 0)
            order = sentence_density_order(text, enc.offsets, positive, keep)
            assert len(order) == len(keep) and set(order) == set(keep)
            rank = np.zeros_like(positive)
            rank[order] = np.arange(len(keep), 0, -1)
            for view, values in [('raw', positive), ('density', rank)]:
                curve = recovery_curve(values, keep, row['gold'], FRACTIONS)
                same_tree(views[view], curve)
                for point in curve['points']:
                    case_rows.append({'dataset': t, 'index': i, 'method': method, 'view': view,
                        'fraction': point['fraction'], 'recall': point['recall'],
                        'budget': point['budget'], 'gold': point['gold'],
                        'ceiling': point['ceiling']})
    table = {(x['dataset'], x['index'], x['method'], x['view'], x['fraction']): x['recall'] for x in case_rows}
    candidates = sorted('/'.join((m.split('_')[1], m.split('_')[2], view))
        for m in r['cases'][0]['metrics'] if m.startswith('DT_') for view in ('raw', 'density'))
    means, contrasts = {}, {}
    for candidate in candidates:
        ref, direction, view = candidate.split('/')
        method = f'DT_{ref}_{direction}'
        means[candidate], contrasts[candidate] = {}, {}
        for t in tasks:
            if not all((t, i, method, view, .1) in table for i in split['tasks'][t][stage]):
                continue
            indices = split['tasks'][t][stage]
            dt = np.array([table[t, i, method, view, .1] for i in indices])
            ft = np.array([table[t, i, 'FT_K3', view, .1] for i in indices])
            means[candidate][t] = {'DT': float(dt.mean()), 'FT_K3': float(ft.mean()),
                'difference': float((dt - ft).mean())}
            diff = np.array([[table[t, i, method, view, f] - table[t, i, 'FT_K3', view, f]
                              for f in FRACTIONS] for i in indices])
            rng = np.random.default_rng(np.random.SeedSequence(73, spawn_key=(TASKS.index(t),)))
            boot = paired_bootstrap(diff, rng)
            contrasts[candidate][t] = {str(f): interval(diff[:, j], boot[:, j]) for j, f in enumerate(FRACTIONS)}
    result = {'status': 'verified_' + stage, 'stage': stage, 'case_count': len(r['cases']),
        'run_results_sha256': digest(a.run / 'results.json'), 'run_vectors_sha256': r['vectors_sha256'],
        'split_sha256': r['split_sha256'], 'driver_sha256': r['driver_sha256'],
        'analyzer_sha256': digest(Path(__file__)), 'tokenizer_sha256': digest(a.tokenizer),
        'scope_inputs_targets_vectors_and_recovery_verified': True, 'means_at10': means,
        'paired_contrasts': contrasts, 'residuals': residuals, 'choice': r['choice'],
        'completed_gpu_operation_seconds': sum(c['seconds'] for c in r['costs'] if c['status'] == 'returned')}
    if stage == 'validation':
        selected = r['choice']['candidate']
        assert selected in means and set(means[selected]) == set(TASKS)
        ref, direction, view = selected.split('/')
        method = f'DT_{ref}_{direction}'
        selected_boot, selected_diff = {}, {}
        for t in TASKS:
            indices = split['tasks'][t][stage]
            diff = np.array([[table[t, i, method, view, f] - table[t, i, 'FT_K3', view, f]
                              for f in FRACTIONS] for i in indices])
            rng = np.random.default_rng(np.random.SeedSequence(73, spawn_key=(TASKS.index(t),)))
            selected_boot[t] = paired_bootstrap(diff, rng)
            selected_diff[t] = diff
        groups = {}
        for name, members in [('VT', TASKS[:4]), ('HotpotQA', TASKS[4:]), ('all_five_tasks', TASKS)]:
            boot = np.mean([selected_boot[t] for t in members], axis=0)
            point = np.mean([selected_diff[t].mean(axis=0) for t in members], axis=0)
            groups[name] = {'tasks': members, 'equal_task_weight': True,
                'DT_recall_at10': float(np.mean([means[selected][t]['DT'] for t in members])),
                'FT_K3_recall_at10': float(np.mean([means[selected][t]['FT_K3'] for t in members])),
                'differences': {str(f): {'mean_difference': float(point[j]),
                    'ci95': np.quantile(boot[:, j], [.025, .975]).tolist()} for j, f in enumerate(FRACTIONS)}}
        result['selected_groups'] = groups
        result['predeclared_shared_advantage_criterion_met'] = (
            groups['VT']['differences']['0.1']['mean_difference'] > 0 and
            groups['HotpotQA']['differences']['0.1']['mean_difference'] > 0 and
            groups['all_five_tasks']['differences']['0.1']['ci95'][0] > 0)
    a.output.mkdir(parents=True, exist_ok=True)
    if stage == 'development':
        ranked = sorted(candidates, key=lambda c: (-min(v['difference'] for v in means[c].values()),
            -np.mean([v['difference'] for v in means[c].values()]), c))
        chosen = ranked[0]
        result['ranked_candidates'] = ranked
        result['best_candidate'] = chosen
        result['both_development_task_means_positive'] = all(v['difference'] > 0 for v in means[chosen].values())
        ref, direction, view = chosen.split('/')
        exact_ceilings = {}
        for t in tasks:
            relevant = [x for x in case_rows if x['dataset'] == t and x['fraction'] == .1
                and x['view'] == view and x['method'] in (f'DT_{ref}_{direction}', 'FT_K3')]
            exact_ceilings[t] = all(x['recall'] == x['ceiling'] for x in relevant)
        result['joint_exact_ceiling_by_task'] = exact_ceilings
        eligible = (any(v['difference'] > 0 for v in means[chosen].values()) and
            all(v['difference'] > 0 or (v['difference'] == 0 and exact_ceilings[t])
                for t, v in means[chosen].items()))
        if a.freeze_choice:
            assert eligible, 'No shared candidate improved both tasks or tied at an exact ceiling; preserve failure and investigate'
            choice = {'status': 'frozen_for_validation', 'candidate': chosen, 'split_sha256': r['split_sha256'],
                'development_results_sha256': digest(a.run / 'results.json'),
                'development_vectors_sha256': r['vectors_sha256'], 'selection_rule': 'max_min_then_macro_then_lexicographic',
                'development_means': means[chosen], 'joint_exact_ceiling_by_task': exact_ceilings,
                'ceiling_eligibility_addendum_sha256': digest(HERE / 'RECALL_CEILING_ADDENDUM.md')}
            path = a.output / 'choice.json'
            if path.exists():
                assert json.loads(path.read_bytes()) == choice
            else:
                path.write_text(json.dumps(choice, indent=2) + '\n', encoding='utf-8')
    (a.output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (a.output / 'cases.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    print(json.dumps({k: v for k, v in result.items() if k in ('status', 'case_count', 'means_at10',
        'best_candidate', 'both_development_task_means_positive')}, indent=2))


if __name__ == '__main__':
    main()
