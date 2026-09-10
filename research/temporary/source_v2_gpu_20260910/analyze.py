"""Independently verify vectors and compare paired references on one scoring scope."""
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
from evidence_protocol import (SUPPORTED_TASKS, recovery_curve, sentence_recovery_curve,
                               source_span, select_source_tokens)
from summarize import summarize

FRACTIONS = [.05, .1, .2, .3, .5]
METHODS = ['DT', 'DT_full_reference', 'FT_K1', 'FT_K3']
UNIT_METRICS = ('recall', 'precision', 'ceiling_adjusted_recall', 'selected_token_fraction')
FIELDS = ([f'{metric}@{int(f * 100):02d}' for f in FRACTIONS
           for metric in ('recall', 'precision', 'ceiling_adjusted_recall')]
          + [f'unit_{metric}@{int(f * 100):02d}' for f in FRACTIONS for metric in UNIT_METRICS]
          + ['rise', 'mas'])
CONTRASTS = {'reference_change': ('DT', 'DT_full_reference'),
             'new_DT_minus_live_FT': ('DT', 'FT'),
             'old_DT_minus_live_FT': ('DT_full_reference', 'FT')}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(matrix, rng, draws=10000):
    """Resample entire cases, preserving all metric correlations."""
    n, width = matrix.shape
    output = np.empty((draws, width), dtype=np.float64)
    for start in range(0, draws, 100):
        size = min(100, draws - start)
        indices = rng.integers(n, size=(size, n))
        output[start:start + size] = matrix[indices].mean(axis=1)
    return output


def interval(values, boot):
    return {'mean_difference': float(np.mean(values)),
            'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'positive_cases': int(np.sum(values > 0)), 'negative_cases': int(np.sum(values < 0)),
            'tied_cases': int(np.sum(values == 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--completed-tasks-only', action='store_true',
                        help='Interim report for complete task shards; never include a task prefix or label it the full suite.')
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    suite = json.loads((args.run / 'suite_status.json').read_bytes())
    assert suite['status'] == 'complete' or (args.completed_tasks_only and suite['status'] == 'running')
    smoke = suite['plan']['stage'] == 'smoke'
    assert not smoke or args.allow_smoke, 'Smoke results are execution checks, not quality conclusions'
    assert suite['plan']['protocol'] == digest(HERE / 'PROTOCOL.md')
    focus = suite['plan'].get('scope') == 'vt_hotpot'
    expected_tasks = ['niah_mq_q2', 'vt_h4_c1', 'hotpotqa_long'] if smoke else list(SUPPORTED_TASKS)
    if focus:
        assert not smoke and suite['plan']['focus_addendum_sha256'] == digest(HERE / 'FOCUS.md')
        expected_tasks = [t for t in SUPPORTED_TASKS if t.startswith('vt_') or t == 'hotpotqa_long']
    assert len(suite['plan']['tasks']) == len(expected_tasks) and set(suite['plan']['tasks']) == set(expected_tasks)
    receipts_by_task = {r['dataset']: r for r in suite['completed_tasks']}
    assert len(receipts_by_task) == len(suite['completed_tasks']) and set(receipts_by_task) <= set(expected_tasks)
    remaining = [t for t in expected_tasks if t not in receipts_by_task]
    interim = bool(remaining)
    assert not interim or (args.completed_tasks_only and not smoke)
    expected_tasks = [t for t in expected_tasks if t in receipts_by_task]
    assert expected_tasks, 'No complete task is available for an interim report'
    release = json.loads((ROOT / 'experiments/official/protocol.json').read_bytes())
    expected_cases = 3 if smoke else sum(release['tasks'][t]['count'] for t in (expected_tasks + remaining))
    fractions = json.loads((ROOT / 'experiments/official/source_protocol.json').read_bytes())['budget_fractions']
    assert fractions == FRACTIONS
    # A task keeps the same resamples in interim and final reports, regardless
    # of execution priority or which other tasks have already completed.
    task_rngs = {task: np.random.default_rng(np.random.SeedSequence(73, spawn_key=(index,)))
                 for index, task in enumerate(SUPPORTED_TASKS)}
    records, task_reports, bootstrap_by_task, means_by_task, receipts = [], {}, {}, {}, []
    costs, cache_receipts = [], []
    for task in expected_tasks:
        receipt = receipts_by_task[task]
        folder = args.run / task
        assert digest(folder / 'results.json') == receipt['results_sha256']
        report = json.loads((folder / 'results.json').read_bytes())
        assert report['paired_reference_audit'] and report['evaluation_protocol'] == 'source-v2'
        assert report['sentence_recovery_enabled']
        assert report['selection'] == ('smoke' if smoke else 'paper')
        assert report['selected_counts'] == {task: 1 if smoke else release['tasks'][task]['count']}
        assert len(report['cases']) == receipt['cases'] == report['selected_counts'][task]
        assert report['driver_sha256'] == suite['plan']['code']['evaluate.py']
        assert digest(folder / 'vectors.npz') == receipt['vectors_sha256'] == report['vectors_sha256']
        summarize(report, release)
        old = json.loads(gzip.decompress((args.publication / 'raw' / (task + '.results.json.gz')).read_bytes()))
        assert digest(args.publication / 'raw' / (task + '.vectors.npz')) == old['vectors_sha256']
        assert report['clean_sources_sha256'] == old['clean_sources_sha256']
        if not smoke:
            assert report['weight_identity'] == old['weight_identity']
        old_cases = {r['index']: r for r in old['cases']}
        cache_path = args.data / (task + '.jsonl')
        assert digest(cache_path) == release['tasks'][task]['cache_sha256']
        caches = [json.loads(line) for line in cache_path.read_text(encoding='utf-8').splitlines()]
        cache_receipts.append({'dataset': task, 'cache_sha256': digest(cache_path)})
        comparison_rows = []
        absolute_rows, tie_rows = [], []
        fresh_legacy_max_abs = []
        with np.load(folder / 'vectors.npz', allow_pickle=False) as vectors, np.load(
                args.publication / 'raw' / (task + '.vectors.npz'), allow_pickle=False) as old_vectors:
            for row in report['cases']:
                before = old_cases[row['index']]
                for field in ('input_ids', 'input_sha256', 'prompt_length', 'target_length', 'user_positions', 'gold'):
                    assert row[field] == before[field], (task, row['index'], field)
                assert row['author_keep'] == before['keep']
                prompt = caches[row['index']]['prompt']
                encoding = tokenizer.encode(' ' + prompt, add_special_tokens=False)
                actual_ids = np.asarray(row['input_ids'])[row['user_positions']]
                assert len(encoding.ids) == len(actual_ids)
                assert np.flatnonzero(np.asarray(encoding.ids) != actual_ids).tolist() == row['standalone_token_boundary_differences']
                span = source_span(task, prompt)
                assert span == row['source_span']
                assert select_source_tokens(span, encoding.offsets, row['author_keep'], row['gold']) == row['keep']
                assert hashlib.sha256(np.asarray(row['input_ids'], dtype=np.int64).tobytes()).hexdigest() == row['input_sha256']
                eos = row['input_ids'][-1]
                assert eos == tokenizer.token_to_id('<|im_end|>')
                for field, keep in [('reference_input_sha256', row['keep']),
                                    ('full_prompt_reference_input_sha256', row['author_keep'])]:
                    altered = np.asarray(row['input_ids'], dtype=np.int64).copy()
                    altered[[row['user_positions'][j] for j in keep]] = eos
                    assert hashlib.sha256(altered.tobytes()).hexdigest() == row[field]
                for method in ('DT', 'DT_full_reference', 'FT_K1'):
                    curve = row['metrics'][method]
                    previous = set()
                    for deleted, actual_hash in zip(curve['deleted_user_indices'], curve['actual_input_hashes'], strict=True):
                        assert previous <= set(deleted) <= set(row['keep'])
                        altered = np.asarray(row['input_ids'], dtype=np.int64).copy()
                        altered[[row['user_positions'][j] for j in deleted]] = eos
                        assert hashlib.sha256(altered.tobytes()).hexdigest() == actual_hash
                        previous = set(deleted)
                    assert previous == set(row['keep'])
                key = f'{task}_{row["index"]}'
                record = {'dataset': task, 'index': row['index'], 'eligible': len(row['keep']),
                          'gold': row['source_gold_count']}
                method_values = {}
                case_ties = {}
                for method in METHODS:
                    vector_name = key + ('_' + method + '_positive_prompt' if method.startswith('DT') else '_' + method + '_prompt')
                    score = vectors[vector_name]
                    assert np.isfinite(score).all()
                    if method != 'FT_K3':
                        positive = np.maximum(score, 0)
                        steps = min(20, len(row['keep']))
                        base, remainder = divmod(len(row['keep']), steps)
                        for step, deleted in enumerate(row['metrics'][method]['deleted_user_indices']):
                            assert len(deleted) == step * base + min(step, remainder)
                            remaining_tokens = sorted(set(row['keep']) - set(deleted))
                            if deleted and remaining_tokens:
                                assert positive[deleted].min() >= positive[remaining_tokens].max(), (key, method, 'deletion ranking')
                    calculated = recovery_curve(score, row['keep'], row['gold'], FRACTIONS)
                    saved = row['FT_K3_recovery'] if method == 'FT_K3' else row['metrics'][method]['recovery']
                    assert calculated == saved, (key, method, 'recovery recomputation')
                    primary = saved['points'][1]
                    case_ties[method] = {
                        'recall_low': primary['recall_tie_low'],
                        'recall_high': primary['recall_tie_high'],
                        'cutoff_zero': primary['cutoff'] == 0,
                        'tie_changes_recall': primary['recall_tie_low'] != primary['recall_tie_high'],
                    }
                    for name, value in case_ties[method].items():
                        record[method + '_tie10_' + name] = value
                    values = [point[metric] for point in saved['points']
                              for metric in ('recall', 'precision', 'ceiling_adjusted_recall')]
                    units = row['FT_K3_sentence_recovery'] if method == 'FT_K3' else row['metrics'][method]['sentence_recovery']
                    assert sentence_recovery_curve(prompt, span, encoding.offsets, score, row['keep'], row['gold'], FRACTIONS) == units
                    values += [point[metric] for point in units['points'] for metric in UNIT_METRICS]
                    values += [row['metrics'][method][f] for f in ('rise', 'mas')] if method != 'FT_K3' else [np.nan, np.nan]
                    method_values[method] = np.asarray(values, dtype=np.float64)
                    for field, value in zip(FIELDS, values):
                        record[method + '_' + field] = float(value) if np.isfinite(value) else None
                # FT comparison uses K3 recovery and K1 faithfulness, as frozen above.
                method_values['FT'] = method_values['FT_K3'].copy()
                method_values['FT'][-2:] = method_values['FT_K1'][-2:]
                absolute_rows.append(np.stack([method_values[m] for m in ('DT', 'DT_full_reference', 'FT')]))
                tie_rows.append(case_ties)
                comparison_rows.append(np.stack([method_values[a] - method_values[b] for a, b in CONTRASTS.values()]))
                full_ref = vectors[key + '_DT_full_reference_signed_full']
                old_full_ref = old_vectors[key + '_DT_signed_full']
                assert full_ref.shape == old_full_ref.shape
                discrepancy = float(np.max(np.abs(full_ref - old_full_ref)))
                record['rerun_full_reference_max_abs_vs_frozen'] = discrepancy
                fresh_legacy_max_abs.append(discrepancy)
                records.append(record)
        data = np.stack(comparison_rows)
        assert np.isfinite(data).all()
        flattened = data.reshape(len(data), -1)
        boot = paired_bootstrap(flattened, task_rngs[task]).reshape(10000, len(CONTRASTS), len(FIELDS))
        bootstrap_by_task[task] = boot
        means_by_task[task] = data.mean(axis=0)
        task_reports[task] = {'count': len(data), 'contrasts': {
            name: {field: interval(data[:, ci, fi], boot[:, ci, fi]) for fi, field in enumerate(FIELDS)}
            for ci, name in enumerate(CONTRASTS)},
            'method_means': {method: dict(zip(FIELDS, np.mean(absolute_rows, axis=0)[mi].tolist()))
                             for mi, method in enumerate(('DT', 'DT_full_reference', 'FT'))},
            'tie10_diagnostics': {method: {
                'cases_with_zero_cutoff': sum(r[method]['cutoff_zero'] for r in tie_rows),
                'cases_with_recall_sensitive_tie': sum(r[method]['tie_changes_recall'] for r in tie_rows),
                'mean_recall_low': float(np.mean([r[method]['recall_low'] for r in tie_rows])),
                'mean_recall_high': float(np.mean([r[method]['recall_high'] for r in tie_rows])),
            } for method in METHODS},
            'fresh_full_reference_vs_frozen_vector_max_abs': max(fresh_legacy_max_abs)}
        costs.extend(report['costs'])
        receipts.append(receipt)
    aggregate_name = 'completed_tasks_only' if interim else 'all_tasks'
    groups = {'NI': [t for t in expected_tasks if t.startswith('niah_')],
              'VT': [t for t in expected_tasks if t.startswith('vt_')],
              'HotpotQA': [t for t in expected_tasks if t == 'hotpotqa_long'], aggregate_name: expected_tasks}
    group_reports = {}
    for name, tasks in groups.items():
        if not tasks:
            continue
        mean = np.mean([means_by_task[t] for t in tasks], axis=0)
        boot = np.mean([bootstrap_by_task[t] for t in tasks], axis=0)
        group_reports[name] = {'task_count': len(tasks), 'aggregation': 'equal_task_weight', 'contrasts': {
            contrast: {field: {'mean_difference': float(mean[ci, fi]),
                'ci95': np.quantile(boot[:, ci, fi], [.025, .975]).tolist()}
                for fi, field in enumerate(FIELDS)} for ci, contrast in enumerate(CONTRASTS)}}
        group_reports[name]['method_means'] = {method: {field: float(np.mean([
            task_reports[t]['method_means'][method][field] for t in tasks])) for field in FIELDS}
            for method in ('DT', 'DT_full_reference', 'FT')}
    output = {'status': 'verified_complete_tasks_partial_suite' if interim else 'verified_complete',
        'scope': 'smoke_execution_only' if smoke else 'complete_task_interim' if interim else f'full_{expected_cases}_paired_evaluation',
        'task_scope': 'vt_hotpot' if focus else 'all', 'expected_case_count': expected_cases,
        'focus_addendum_sha256': digest(HERE / 'FOCUS.md') if focus else None,
        'quality_conclusion_allowed': not smoke, 'case_count': len(records), 'task_count': len(expected_tasks),
        'full_suite_complete': not smoke and not interim, 'remaining_tasks': remaining,
        'deletion_ranking_verified_from_saved_vectors': True,
        'protocol_sha256': digest(HERE / 'PROTOCOL.md'), 'analysis_sha256': digest(Path(__file__)),
        'tokenizer_sha256': digest(args.tokenizer), 'data_caches_verified': cache_receipts,
        'sentence_curves_verified_from_saved_vectors': True,
        'bootstrap': {'draws': 10000, 'seed': 73, 'unit': 'paired cases within each fixed task',
                      'streams': 'SeedSequence(73, spawn_key=(fixed SUPPORTED_TASKS index,))'},
        'difference_direction': 'new minus comparator; recovery higher is better; RISE/MAS lower is better',
        'tasks': task_reports, 'groups': group_reports, 'source_receipts': receipts,
        'measured_call_seconds': sum(c['seconds'] for c in costs),
        'peak_allocated_bytes': max(c['peak_allocated'] for c in costs)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'analysis.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (args.output / 'paired_cases.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({k: output[k] for k in ['status', 'scope', 'case_count', 'measured_call_seconds']}, indent=2))
    print(json.dumps(group_reports[aggregate_name]['contrasts']['reference_change'], indent=2))


if __name__ == '__main__':
    main()
