"""Rescore existing H6/H10 vectors at a fixed, retrospective budget grid."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'all_baselines_20260910'
sys.path.insert(0, str(PARENT))
from common import previous_cases, sha
from recovery_diagnostics import recovery_diagnostics


def write_json(path, value, compact=False):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                              indent=None if compact else 2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_check(scores, keep, gold, fraction, row, selected):
    """Use Python sorting and sets, independently of NumPy's scoring routine."""
    eligible = sorted(set(map(int, keep)))
    assert eligible and min(eligible) >= 0 and max(eligible) < len(scores)
    truth = set(map(int, gold)) & set(eligible)
    ranking = sorted(eligible, key=lambda i: (-max(0., float(scores[i])), i))
    count = math.ceil(len(eligible) * fraction)
    assert selected == ranking[:count]
    hit = len(set(selected) & truth)
    cutoff = max(0., float(scores[selected[-1]]))
    above = {i for i in eligible if max(0., float(scores[i])) > cutoff}
    equal = {i for i in eligible if max(0., float(scores[i])) == cutoff}
    free = count - len(above)
    expected = dict(
        eligible=len(eligible), gold=len(truth), budget=count,
        gold_density=len(truth) / len(eligible),
        ceiling=min(1., count / len(truth)),
        random_expected_recall=count / len(eligible),
        recall=hit / len(truth), precision=hit / count,
        ceiling_adjusted_recall=hit / min(count, len(truth)),
        cutoff=cutoff, cutoff_tie_size=len(equal),
        recall_tie_low=(len(above & truth) + max(0, free - len(equal - truth))) / len(truth),
        recall_tie_high=(len(above & truth) + min(free, len(equal & truth))) / len(truth),
    )
    for metric, value in expected.items():
        assert math.isclose(float(row[metric]), value, rel_tol=0., abs_tol=1e-12), (metric, row, value)
    assert 0 <= row['recall'] <= row['ceiling'] <= 1


def build_report(plan, summaries):
    names = {'DT': 'DeltaTrace', 'FT_K3': 'FlashTrace K3', 'FT_K1': 'FlashTrace K1',
             'Perturbation': 'Perturbation (20 segments)', 'REAGENT': 'REAGENT (20 segments)',
             'CLP': 'CLP (20 segments)', 'IFR': 'IFR', 'AttnLRP': 'AttnLRP †'}
    lookup = {(r['dataset'], r['method'], r['fraction']): r for r in summaries}
    lines = [
        '# VT H6/H10: increased retrieval budgets', '',
        'The user requested larger retrieval budgets after every DT, FT K3, IFR and AttnLRP',
        'case reached the Recall ceiling at 10% on these two tasks. The previously measured',
        '20% results distinguish H6, while H10 still shows ceiling saturation. This extension',
        'therefore reports a common 10%, 20%, 30%, 40% grid for all methods on both tasks.',
        'The 30%/40% grid was fixed before calculating those new results. This is a retrospective',
        'extension, not a new holdout or a newly preregistered primary comparison.', '',
        '**200 inputs; 7 main methods plus supplementary FT K1; 6,400 score rows; zero new',
        'attribution/model calls.** All rankings use the existing verified attribution vectors.',
        'Only the number of retrieved tokens increases. Target scope, positive-score view,',
        'eligible body-token mask and stable position tie-breaking retain the frozen VT policy.',
        'Gold labels are used only to measure recall and its theoretical ceiling.', '',
        'The original [seven-method Recall@10% report](../all_baselines_20260910/RESULTS.md)',
        'and its macro average remain the historical fixed-budget results. No macro average',
        'mixing different task budgets is substituted for that Recall@10% column.', '',
    ]
    for task in plan['tasks']:
        lines += [f'## {task}: body-token Recall', '',
                  '| Method | 10% budget | 20% budget | 30% budget | 40% budget |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for method in plan['main_methods']:
            lines.append('| ' + names[method] + ' | ' + ' | '.join(
                f"{lookup[task, method, f]['recall'] * 100:.2f}%" for f in plan['fractions']) + ' |')
        lines.append('| Theoretical ceiling (case mean) | ' + ' | '.join(
            f"{lookup[task, 'DT', f]['ceiling'] * 100:.2f}%" for f in plan['fractions']) + ' |')
        lines += ['', 'Number of cases at their individual budget-dependent Recall ceiling:', '',
                  '| Method | 10% | 20% | 30% | 40% |', '| --- | ---: | ---: | ---: | ---: |']
        for method in plan['main_methods']:
            lines.append('| ' + names[method] + ' | ' + ' | '.join(
                f"{lookup[task, method, f]['at_ceiling_cases']}/100" for f in plan['fractions']) + ' |')
        lines += ['', 'Precision (case mean), showing the cost of retrieving more tokens:', '',
                  '| Method | 10% | 20% | 30% | 40% |', '| --- | ---: | ---: | ---: | ---: |']
        for method in plan['main_methods']:
            lines.append('| ' + names[method] + ' | ' + ' | '.join(
                f"{lookup[task, method, f]['precision'] * 100:.2f}%" for f in plan['fractions']) + ' |')
        lines.append('')
    lines += ['## Supplementary FT K1', '', '| Task | 10% | 20% | 30% | 40% |',
              '| --- | ---: | ---: | ---: | ---: |']
    for task in plan['tasks']:
        lines.append('| ' + task + ' | ' + ' | '.join(
            f"{lookup[task, 'FT_K1', f]['recall'] * 100:.2f}%" for f in plan['fractions']) + ' |')
    lines += ['', '## Interpretation and verification', '',
              'Recall ceilings are computed per case as min(1, ceil(f × eligible tokens) / gold tokens),',
              'then averaged. A ceiling count means the selected prefix attains the maximum possible',
              'Recall at that budget; it does not mean Recall is necessarily 100%. At 40%, both tasks',
              'permit 100% Recall in every case, so remaining misses are not forced by token capacity.',
              'Larger budgets can increase Recall while decreasing Precision. These tables are',
              'descriptive; no new significance claim or best-budget selection is made.', '',
              '† AttnLRP retains the recorded numeric repair and lossless saved-tensor offload.',
              'Perturbation, REAGENT and CLP retain the native 20-source-segment approximation.', '',
              'The check reconstructs all selected prefixes using an independent Python sort,',
              'recomputes every metric and tie bound, checks nested selections and monotonic Recall,',
              'and exactly reproduces all 3,200 historical 10%/20% metric rows within 1e-12.',
              'Parent inputs and results are bound by hashes, including all 1,000 new-baseline',
              'raw vector files used here and the original DT/FT source archives.', '',
              '- [Fixed extension protocol](protocol.json)',
              '- [All per-case metrics](per_case.csv)',
              '- [Summary and tie-range diagnostics](summary.csv)',
              '- [Exact retrieved token lists](selections.json)',
              '- [Input/vector provenance](provenance.json)',
              '- [Verification receipt](verification.json)', '',
              'Reproduce from the repository root with NumPy available:', '', '```bash',
              'python research/temporary/vt_budget_extension_20260911/recompute.py', '```', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    plan = json.loads((HERE / 'protocol.json').read_bytes())
    for name, digest in plan['parent_sha256'].items():
        assert sha(PARENT / name) == digest, name
    parent = json.loads((PARENT / 'analysis.json').read_bytes())
    verified = json.loads((PARENT / 'verification.json').read_bytes())
    assert verified['status'] == 'passed'
    assert verified['output_sha256']['analysis.json'] == sha(PARENT / 'analysis.json')
    assert verified['output_sha256']['per_case.csv'] == sha(PARENT / 'per_case.csv')
    source = {name: {(r['dataset'], r['index']): r for r in json.loads(
        (PARENT / (name + '.json')).read_bytes())['cases']} for name in ('inputs', 'candidates', 'labels')}
    registered = {(r['dataset'], r['index'], r['method']): r for r in parent['new_receipts']}
    old, origins = previous_cases()
    assert origins == parent['origins']
    with (PARENT / 'per_case.csv').open(newline='', encoding='utf-8') as stream:
        historical = {(r['dataset'], int(r['index']), r['method'], float(r['fraction'])): r
                      for r in csv.DictReader(stream) if r['dataset'] in plan['tasks']
                      and r['aggregation'] == 'signed_sum' and r['view'] == 'raw'}
    rows, selections, receipts = [], [], []
    reused_checks = nested_checks = 0
    methods = plan['main_methods'] + plan['supplementary_methods']
    for task, count in plan['tasks'].items():
        for index in range(count):
            key = (task, index)
            item, candidate, label = [source[name][key] for name in ('inputs', 'candidates', 'labels')]
            assert item['target_mode'] == plan['target_mode']
            for method in methods:
                if method in ('DT', 'FT_K3', 'FT_K1'):
                    values = old[key][1][method]
                    if method == 'DT':
                        values = values[item['user_positions']]
                else:
                    folder = PARENT / 'raw' / method / task / f'{index:03d}'
                    record = registered[task, index, method]
                    assert sha(folder / 'results.json') == record['results_sha256']
                    assert sha(folder / 'vectors.npz') == record['vectors_sha256']
                    with np.load(folder / 'vectors.npz') as vectors:
                        values = vectors['prompt_signed'].copy()
                    receipts.append(record)
                scores = np.asarray(values, dtype=np.float32)
                assert scores.shape == (len(item['user_positions']),) and np.isfinite(scores).all()
                previous, previous_recall = None, -1.
                for fraction in plan['fractions']:
                    metrics = recovery_diagnostics(np.maximum(scores, 0), candidate['keep'], label['gold'], fraction)
                    selected = metrics.pop('selected')
                    independent_check(scores, candidate['keep'], label['gold'], fraction, metrics, selected)
                    if previous is not None:
                        assert selected[:len(previous)] == previous
                        assert metrics['recall'] >= previous_recall
                        nested_checks += 1
                    previous, previous_recall = selected, metrics['recall']
                    if fraction in (.1, .2):
                        reference = historical[task, index, method, fraction]
                        for metric, value in metrics.items():
                            assert math.isclose(float(reference[metric]), value, rel_tol=0., abs_tol=1e-12)
                        reused_checks += 1
                    base = dict(dataset=task, index=index, method=method, target_mode='answer_only',
                                aggregation='signed_sum', view='raw', budget_unit='eligible_body_tokens', fraction=fraction)
                    rows.append(dict(base, **metrics))
                    selections.append(dict(base, budget=metrics['budget'], selected_tokens=selected))
        print(json.dumps(dict(status='task_scored_and_checked', dataset=task, rows=len(rows))), flush=True)
    assert len(rows) == len(selections) == 6400 and reused_checks == 3200 and nested_checks == 4800
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['method'], row['fraction']].append(row)
    summaries = []
    for (task, method, fraction), cases in groups.items():
        assert len(cases) == 100
        summary = dict(dataset=task, method=method, fraction=fraction, cases=100)
        for metric in ('recall', 'precision', 'ceiling', 'ceiling_adjusted_recall', 'budget',
                       'gold_density', 'random_expected_recall', 'recall_tie_low', 'recall_tie_high'):
            summary[metric] = float(np.mean([r[metric] for r in cases]))
        summary['at_ceiling_cases'] = sum(abs(r['recall'] - r['ceiling']) < 1e-12 for r in cases)
        summary['full_recall_cases'] = sum(r['recall'] == 1 for r in cases)
        summaries.append(summary)
    assert len(summaries) == 64
    assert all(r['ceiling'] == 1 for r in rows if r['fraction'] == .4)
    write_csv(HERE / 'per_case.csv', rows)
    write_csv(HERE / 'summary.csv', summaries)
    write_json(HERE / 'selections.json', selections, compact=True)
    write_json(HERE / 'provenance.json', dict(parent_sha256=plan['parent_sha256'],
               old_vector_origins=origins, baseline_vector_receipts=receipts))
    build_report(plan, summaries)
    for name, digest in plan['parent_sha256'].items():
        assert sha(PARENT / name) == digest, name
    output_names = ['protocol.json', 'per_case.csv', 'summary.csv', 'selections.json', 'provenance.json', 'RESULTS.md']
    write_json(HERE / 'verification.json', dict(status='passed', inputs=200, main_methods=7,
               supplementary_methods=['FT_K1'], attribution_runs=0, score_rows=6400,
               independently_reconstructed_selections=6400, exact_historical_metric_rows=3200,
               nested_budget_transitions=4800, baseline_vector_hashes=1000,
               parent_files_unchanged=True, output_sha256={name: sha(HERE / name) for name in output_names},
               scorer_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status='verified_budget_extension', score_rows=6400, summaries=summaries), indent=2))


if __name__ == '__main__':
    main()
