"""Independently verify the manuscript CSV and all three quality tables."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
METHODS = ['Perturbation', 'REAGENT', 'CLP', 'IFR', 'AttnLRP', 'FT', 'DT']
NAMES = {'niah_mq_q2':'MQ-Q2', 'niah_mq_q4':'MQ-Q4', 'niah_mq_q8':'MQ-Q8',
         'niah_mv_v2':'MV-V2', 'niah_mv_v4':'MV-V4', 'niah_mv_v8':'MV-V8',
         'vt_h2_c3':'VT-H2-C3', 'vt_h4_c1':'VT-H4-C1', 'vt_h6_c1':'VT-H6-C1',
         'vt_h10_c1':'VT-H10-C1', 'hotpotqa_long':'HotpotQA', 'math':'MATH', 'morehopqa':'MoreHopQA'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    provenance = json.loads((HERE / 'sources.json').read_bytes())
    for source in provenance['sources'].values():
        assert sha(HERE / source['fixture']) == source['fixture_sha256']
    manifest = json.loads((DATA / 'current_recovery_sources.json').read_bytes())
    assert manifest['status'] == 'verified_current_recovery_import'
    for entry in manifest['files']:
        assert sha(HERE / entry['fixture']) == entry['sha256']
    assert sha(DATA / 'current_recovery.csv') == manifest['output_sha256']
    assert sha(HERE / 'import_current_recovery.py') == manifest['importer_sha256']
    snapshot = DATA / 'current_recovery'
    receipt = json.loads((snapshot / 'verification.json').read_bytes())
    assert receipt['status'] == 'passed'
    for name in ('protocol.json', 'primary.csv', 'per_case.csv'):
        assert sha(snapshot / name) == receipt['output_sha256'][name]
    selected = read_csv(snapshot / 'per_case.csv')
    policy = json.loads((snapshot / 'protocol.json').read_bytes())
    full = {r['dataset']: r for r in read_csv(DATA / 'qwen3_full.csv')}
    published = {(r['dataset'], r['method']): r for r in read_csv(DATA / 'published_baselines.csv')}
    current = {(r['dataset'], r['method']): r for r in read_csv(DATA / 'current_recovery.csv')}
    combined = read_csv(HERE / 'all_methods.csv')
    assert len(combined) == 91 and len(current) == 35
    assert {(r['dataset'], r['method']) for r in combined} == {(t, m) for t in full for m in METHODS}
    historical_cells = current_cells = total_cells = 0
    expected = {}
    for row in combined:
        task, method = row['dataset'], row['method']
        assert int(row['released_task_count']) == int(full[task]['count'])
        pair = method in ('DT', 'FT')
        source = full[task] if pair else published[task, method]
        metrics = {key: float(source[f'{method}_{key}' if pair else key]) for key in ('RISE', 'MAS')}
        for key, value in metrics.items():
            assert float(row[key]) == value
            historical_cells += 1
            total_cells += 1
        if task.startswith('niah_'):
            reference = float(source[f'{method}_Recall10' if pair else 'Recovery10'])
            assert row['RecoveryProtocol'] == 'table1-data-v1'
            assert row['RecoveryMetric'] == 'released_token_recovery'
            assert row['RecoveryBudgetUnit'] == 'released_eligible_tokens'
            assert float(row['RecoveryBudgetFraction']) == .1
            historical_cells += 1
        elif (task, method) in current:
            source = current[task, method]
            native_method = 'FT_K3' if method == 'FT' else method
            cases = [r for r in selected if r['dataset'] == task and r['method'] == native_method]
            assert len(cases) == (48 if task == 'hotpotqa_long' else 100)
            assert all(float(r['fraction']) == policy['budgets'][task] for r in cases)
            reference = statistics.fmean(float(r['recall']) for r in cases)
            assert float(source['Recovery']) == reference
            assert float(row['RecoveryBudgetFraction']) == policy['budgets'][task]
            for key in ('RecoveryMetric', 'RecoveryBudgetUnit', 'RecoveryTargetMode', 'RecoveryProtocol'):
                assert row[key] == source[key]
            assert row['RecoveryAttributionVariant'] == native_method
            current_cells += 1
        else:
            reference = None
            assert not row['Recovery'] and not row['RecoveryBudgetFraction']
        if reference is not None:
            assert float(row['Recovery']) == reference
            total_cells += 1
        metrics['Recovery'] = reference
        expected[task, method] = metrics
    assert (historical_cells, current_cells, total_cells) == (224, 35, 259)
    vt = [task for task in full if task.startswith('vt_')]
    macro = {m: statistics.fmean(expected[t, m]['Recovery'] for t in vt) for m in METHODS}
    for row in read_csv(HERE / 'recovery_summary.csv'):
        assert math.isclose(float(row['recall']), macro[row['method']], rel_tol=0, abs_tol=1e-12)
        assert row['task_budgets'] == 'H2=10%;H4=10%;H6=20%;H10=30%'
    tables = (HERE / 'full_table.tex').read_text(encoding='utf-8')
    inverse = {name: task for task, name in NAMES.items()}
    checked_table_rows = 0
    best_counts = {}
    for metric, label in [('RISE', 'tab:full'), ('MAS', 'tab:mas'), ('Recovery', 'tab:recovery')]:
        part = tables.split(r'\label{' + label + '}', 1)[1].split(r'\end{table}', 1)[0]
        best_counts[metric] = 0
        found = set()
        for line in part.splitlines():
            cells = [cell.strip() for cell in line.split(' & ')]
            if cells[0] not in inverse and cells[0] != 'VT macro':
                continue
            is_macro = cells[0] == 'VT macro'
            task = inverse.get(cells[0])
            found.add(cells[0])
            values = [macro[m] if is_macro else expected[task, m][metric] for m in METHODS]
            assert all(v is not None for v in values)
            assert cells[1] == ('400' if is_macro else full[task]['count'])
            if metric == 'Recovery':
                budget = 'mixed' if is_macro else f"{round(float(next(r['RecoveryBudgetFraction'] for r in combined if r['dataset']==task))*100)}" + r'\%'
                assert cells[2] == budget
            formatted = cells[3:] if metric == 'Recovery' else cells[2:]
            formatted[-1] = formatted[-1].removesuffix(r'\\').strip()
            assert len(formatted) == 7
            best = (max if metric == 'Recovery' else min)(values)
            if not is_macro:
                best_counts[metric] += abs(values[-1] - best) < 1e-12
            for cell, value in zip(formatted, values):
                wanted = f'{100 * value:.2f}' if metric == 'Recovery' else f'{value:.4f}'
                if abs(value - best) < 1e-12:
                    wanted = r'\textbf{' + wanted + '}'
                assert cell == wanted, (metric, task, cell, wanted)
            checked_table_rows += 1
        required = {NAMES[t] for t in full if metric != 'Recovery' or expected[t, 'DT']['Recovery'] is not None}
        if metric == 'Recovery':
            required.add('VT macro')
        assert found == required
    assert checked_table_rows == 38 and best_counts == dict(RISE=9, MAS=11, Recovery=10)
    summary = json.loads((HERE / 'summary.json').read_bytes())
    assert summary['best_among_all_methods'] == best_counts
    assert summary['current_recovery']['metric_cells'] == 259
    outputs = ['all_methods.csv', 'recovery_summary.csv', 'full_table.tex', 'summary.json']
    result = dict(status='passed', all_methods=METHODS, method_task_rows=91, reported_metric_cells=259,
                  historical_metric_cells_preserved=224, added_recovery_method_task_cells=35,
                  unavailable_recovery_method_task_cells=14, checked_table_rows=38,
                  quality_examples=1243, quality_tasks=13, current_recovery_examples=448,
                  best_task_counts=best_counts, vt_macro_task_specific_budgets=macro,
                  recovery_budgets=policy['budgets'], no_model_calls=True,
                  current_recovery_manifest_sha256=sha(DATA / 'current_recovery_sources.json'),
                  output_sha256={name: sha(HERE / name) for name in outputs},
                  builder_sha256=sha(HERE / 'build_results.py'), verifier_sha256=sha(Path(__file__)))
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('sha256')}))


if __name__ == '__main__':
    main()
