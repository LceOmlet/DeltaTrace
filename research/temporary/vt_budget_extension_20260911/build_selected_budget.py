"""Publish the user's H6=20%, H10=30% reporting policy from verified scores."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent / 'selected_budget'
ROOT = HERE.parents[3]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    plan = json.loads((HERE / 'protocol.json').read_bytes())
    for name, digest in plan['source_sha256'].items():
        assert sha(HERE / name) == digest, name
    frozen = HERE.parent.parent / 'all_baselines_20260910'
    extension = HERE.parent
    for folder in (frozen, extension):
        receipt = json.loads((folder / 'verification.json').read_bytes())
        assert receipt['status'] == 'passed'
        assert receipt['output_sha256']['per_case.csv'] == sha(folder / 'per_case.csv')
    methods = plan['main_methods'] + plan['supplementary_methods']
    tasks = list(plan['budgets'])
    rows = []
    for folder in (frozen, extension):
        for row in read_csv(folder / 'per_case.csv'):
            task = row['dataset']
            if task not in plan['budgets'] or row['method'] not in methods:
                continue
            is_extension_task = task in ('vt_h6_c1', 'vt_h10_c1')
            if is_extension_task != (folder == extension):
                continue
            if row['aggregation'] != 'signed_sum' or float(row['fraction']) != plan['budgets'][task]:
                continue
            expected = ('native_sentence', 'all_body_tokens', 'full') if task == 'hotpotqa_long' else (
                'raw', 'eligible_body_tokens', 'answer_only')
            if (row['view'], row['budget_unit'], row['target_mode']) == expected:
                rows.append(row)
    rows.sort(key=lambda r: (tasks.index(r['dataset']), methods.index(r['method']), int(r['index'])))
    assert len(rows) == 3584
    keys = {(r['dataset'], r['method'], int(r['index'])) for r in rows}
    assert keys == {(t, m, i) for t in tasks for m in methods
                    for i in range(48 if t == 'hotpotqa_long' else 100)}
    # Preserve the common CSV schema, including the untouched HotpotQA diagnostics.
    fields = list(dict.fromkeys(k for row in rows for k in row))
    rows = [{k: row.get(k, '') for k in fields} for row in rows]
    write_csv(HERE / 'per_case.csv', rows)
    means = {(t, m): statistics.fmean(float(r['recall']) for r in rows
                                    if r['dataset'] == t and r['method'] == m)
             for t in tasks for m in methods}
    columns = {t: f'{t}_recall_at_{round(plan["budgets"][t] * 100)}pct' for t in tasks}
    primary = []
    for method in methods:
        item = dict(method=method, **{columns[t]: means[t, method] for t in tasks})
        item['vt_macro_task_specific_budgets'] = statistics.fmean(means[t, method] for t in tasks if t.startswith('vt_'))
        primary.append(item)
    write_csv(HERE / 'primary.csv', primary)
    # Check the selected means against each source's independently verified summary.
    for task in tasks:
        folder = extension if task in ('vt_h6_c1', 'vt_h10_c1') else frozen
        source_summary = read_csv(folder / 'summary.csv')
        for method in methods:
            candidates = [r for r in source_summary if r['dataset'] == task and r['method'] == method
                          and float(r['fraction']) == plan['budgets'][task]
                          and r.get('aggregation', 'signed_sum') == 'signed_sum'
                          and r.get('view', 'raw') == ('native_sentence' if task == 'hotpotqa_long' else 'raw')
                          and r.get('budget_unit', 'eligible_body_tokens') == (
                              'all_body_tokens' if task == 'hotpotqa_long' else 'eligible_body_tokens')]
            assert len(candidates) == 1, (task, method)
            assert math.isclose(means[task, method], float(candidates[0]['recall']), rel_tol=0, abs_tol=1e-12)
    names = {'DT': 'DeltaTrace', 'FT_K3': 'FlashTrace K3', 'FT_K1': 'FlashTrace K1',
             'Perturbation': 'Perturbation', 'REAGENT': 'REAGENT', 'CLP': 'CLP', 'IFR': 'IFR', 'AttnLRP': 'AttnLRP †'}
    table = ['| Method | H2-C3 @10% | H4-C1 @10% | H6-C1 @20% | H10-C1 @30% | VT macro (task-specific budgets) | HotpotQA @10% |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in primary:
        if row['method'] not in plan['main_methods']:
            continue
        values = [row[columns[t]] for t in tasks if t.startswith('vt_')]
        values += [row['vt_macro_task_specific_budgets'], row[columns['hotpotqa_long']]]
        table.append('| ' + names[row['method']] + ' | ' + ' | '.join(f'{x * 100:.2f}%' for x in values) + ' |')
    lines = ['# Current comparison with user-selected task budgets', '',
             '**VT budgets: H2-C3 10%, H4-C1 10%, H6-C1 20%, H10-C1 30%. HotpotQA remains 10%.**', '',
             'The user selected H6=20% and H10=30% after seeing the complete budget grid.',
             'This is a retrospective reporting choice. It is not an independent holdout',
             'or a budget choice fixed before observing results. All algorithms use the same',
             'budget within each task. The complete [10%–40% grid](../RESULTS.md) remains available.', '',
             'VT reports eligible body-token Recall using its saved answer-only input and positive',
             'token ranking. HotpotQA reports official supporting-fact Recall using full-response',
             'signed sentence sums and the native all-body-token prefix budget.', '', *table, '',
             'The VT macro is the equal mean of the four task recalls at their stated budgets.',
             'It is labeled **task-specific-budget macro**, not Recall@10%. It does not include',
             'HotpotQA, whose recall denominator is different. The old all-10% table and its',
             '[historical intervals](../../all_baselines_20260910/RESULTS.md) remain archived.',
             'These selected-budget tables are descriptive and make no new significance claim.', '',
             '## Provenance', '',
             'All 3,584 selected per-case rows (448 inputs × 8 methods, including supplementary',
             'FT K1) are copied from verified scores. No attribution or model computation is',
             'rerun. Each of the 40 task/method means is checked against its source summary.', '',
             '† AttnLRP retains its documented numeric repair and lossless saved-tensor offload.',
             'Perturbation, REAGENT and CLP retain the author’s 20-source-segment approximation.', '',
             '- [Selected policy](protocol.json)', '- [Primary CSV, including supplementary FT K1](primary.csv)',
             '- [Selected per-case metrics](per_case.csv)', '- [Verification](verification.json)', '',
             'Reproduce from the repository root:', '', '```bash',
             'python research/temporary/vt_budget_extension_20260911/build_selected_budget.py', '```', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    # Deserialization confirms that the published rows preserve their source values.
    assert read_csv(HERE / 'per_case.csv') == rows
    output_names = ['protocol.json', 'primary.csv', 'per_case.csv', 'RESULTS.md']
    write_json(HERE / 'verification.json', dict(status='passed', cases=448, main_methods=7,
               supplementary_methods=['FT_K1'], score_rows=3584, checked_source_means=40,
               new_attribution_runs=0, source_sha256=plan['source_sha256'],
               output_sha256={name: sha(HERE / name) for name in output_names}, builder_sha256=sha(Path(__file__))))
    readme = ROOT / 'README.md'
    original = readme.read_bytes()
    text = original.decode('utf-8')
    start, end = '<!-- selected-vt-budgets:start -->', '<!-- selected-vt-budgets:end -->'
    block = '\n'.join([start, '### Current comparison with task-specific VT budgets', '',
        'VT retrieval budgets are **H2-C3 10%, H4-C1 10%, H6-C1 20%, H10-C1 30%**;',
        'HotpotQA remains at 10%. These budgets were selected by the user after viewing',
        'the budget sweep. The table is a retrospective comparison using the same budget',
        'for every method within a task.', '', *table, '',
        'The VT macro uses the four stated budgets; it is not Recall@10%. All 3,584 selected',
        'per-case rows and 40 source means are verified, with no attribution reruns.',
        'See the [current report, policy and data](research/temporary/vt_budget_extension_20260911/selected_budget/RESULTS.md)',
        'and the [complete budget grid](research/temporary/vt_budget_extension_20260911/RESULTS.md).', end])
    if start in text:
        left = text.index(start)
        right = text.index(end, left) + len(end)
        updated = text[:left] + block + text[right:]
    else:
        anchor = '<!-- frozen-all-baselines:start -->'
        assert text.count(anchor) == 1
        updated = text.replace(anchor, block + '\n\n' + anchor, 1)
        assert updated.replace(block + '\n\n', '', 1) == text
    readme.write_bytes(updated.encode('utf-8'))
    write_json(HERE / 'index_receipt.json', dict(status='current_budget_report_indexed',
               previous_readme_sha256=hashlib.sha256(original).hexdigest(), readme_sha256=sha(readme),
               report_sha256=sha(HERE / 'RESULTS.md'), verification_sha256=sha(HERE / 'verification.json')))
    for name, digest in plan['source_sha256'].items():
        assert sha(HERE / name) == digest, name
    print(json.dumps(dict(status='current_task_budgets_published_and_verified', primary=primary), indent=2))


if __name__ == '__main__':
    main()
