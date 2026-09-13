"""Evaluate the frozen candidate using the prespecified paired stratified bootstrap."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run
    status = json.loads((run / 'status.json').read_bytes())
    protocol = json.loads((run / 'protocol.json').read_bytes())
    assert status['status'] == 'complete'
    assert len(status['cases']) == protocol['expected_cases']
    assert hashlib.sha256((run / 'protocol.json').read_bytes()).hexdigest() == status['protocol_sha256']
    data = json.loads((run / 'input_cases.json').read_bytes())
    assert hashlib.sha256((run / 'input_cases.json').read_bytes()).hexdigest() == protocol['validation_cases_sha256']
    assert data['candidate_receipt_sha256'] == protocol['candidate_receipt_sha256']
    intended = {(task, index) for task, indices in protocol['selection'].items() for index in indices}
    assert {(c['dataset'], c['index']) for c in data['cases']} == intended
    rows = []
    runtime_checks = []
    seen = set()
    for meta in status['cases']:
        file = run / meta['path'] / 'results.json'
        assert hashlib.sha256(file.read_bytes()).hexdigest() == meta['results_sha256']
        c = json.loads(file.read_bytes())
        key = c['dataset'], c['index']
        assert key not in seen
        seen.add(key)
        assert c['status'] == 'complete' and c['baseline_repeat_relative_l2'] < 1e-4
        assert c['methods']['DT_original']['root_effect'] == c['methods']['DT_gdn_symmetric']['root_effect']
        runtime_checks.append(dict(dataset=c['dataset'], index=c['index'],
                                   baseline_repeat_relative_l2=c['baseline_repeat_relative_l2'],
                                   baseline_repeat_bitwise=c['baseline_repeat_bitwise'],
                                   **{method: c['methods'][method]['relative_residual'] for method in protocol['methods']}))
        for method in protocol['methods']:
            rows.append(dict(dataset=c['dataset'], index=c['index'], method=method,
                             **{k: c['methods'][method][k] for k in ('rise', 'mas', 'recall')}))
        rows.append(dict(dataset=c['dataset'], index=c['index'], method='FT_K1', **c['FT_K1']))
        rows.append(dict(dataset=c['dataset'], index=c['index'], method='FT_K3',
                         rise=None, mas=None, recall=c['FT_K3_recall']))
    assert seen == intended
    tasks = list(protocol['selection'])
    methods = protocol['methods'] + ['FT_K1', 'FT_K3']
    metrics = ('rise', 'mas', 'recall')
    lookup = {(r['dataset'], r['index'], r['method']): r for r in rows}
    means = []
    for task in tasks:
        for method in methods:
            rr = [r for r in rows if r['dataset'] == task and r['method'] == method]
            means.append(dict(dataset=task, method=method, n=len(rr), **{
                metric: float(np.mean([r[metric] for r in rr])) if rr[0][metric] is not None else None
                for metric in metrics}))
    for method in methods:
        rr = [r for r in means if r['method'] == method]
        means.append(dict(dataset='macro', method=method, n=len(seen), **{
            metric: float(np.mean([r[metric] for r in rr])) if rr[0][metric] is not None else None
            for metric in metrics}))
    rng = np.random.default_rng(protocol['bootstrap']['seed'])
    draws = {task: rng.integers(0, len(protocol['selection'][task]),
                              (protocol['bootstrap']['samples'], len(protocol['selection'][task]))) for task in tasks}
    paired = []
    for metric in metrics:
        task_bootstraps = []
        task_deltas = []
        for task in tasks:
            diff = np.array([lookup[task, i, 'DT_gdn_symmetric'][metric] - lookup[task, i, 'DT_original'][metric]
                             for i in protocol['selection'][task]])
            bootstrap = diff[draws[task]].mean(axis=1)
            task_bootstraps.append(bootstrap)
            task_deltas.append(float(diff.mean()))
            paired.append(dict(dataset=task, metric=metric, delta=float(diff.mean()),
                               ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                               better=int(sum(diff > 0 if metric == 'recall' else diff < 0)),
                               equal=int(sum(diff == 0)), n=len(diff)))
        paired.append(dict(dataset='macro', metric=metric, delta=float(np.mean(task_deltas)),
                           ci95=np.quantile(np.mean(task_bootstraps, axis=0), [.025, .975]).tolist(), n=len(seen)))
    macro = {r['metric']: r for r in paired if r['dataset'] == 'macro'}
    checks = dict(rise_ci_upper_below_zero=macro['rise']['ci95'][1] < 0,
                  mas_mean_not_worse=macro['mas']['delta'] <= 0,
                  recall_mean_not_worse=macro['recall']['delta'] >= 0)
    result = dict(status='complete', expected_cases=len(intended), actual_cases=len(seen),
                  frozen_success_criteria=checks, passed=all(checks.values()), means=means, paired=paired, rows=rows,
                  runtime_checks=runtime_checks,
                  max_baseline_repeat_relative_l2=max(r['baseline_repeat_relative_l2'] for r in runtime_checks),
                  all_baseline_repeats_bitwise=all(r['baseline_repeat_bitwise'] for r in runtime_checks),
                  max_absolute_relative_residual={method: max(abs(r[method]) for r in runtime_checks) for method in protocol['methods']},
                  bootstrap=protocol['bootstrap'],
                  inference='Frozen single candidate on newly assigned keys and values. Equal-task paired bootstrap; intervals quantify these template variants, not unseen template or natural-task generalization. Per-task and secondary-metric intervals are descriptive, without multiplicity correction.')
    (run / 'validation_analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'actual_cases', 'passed', 'frozen_success_criteria')}))
    print(json.dumps(dict(macro=[r for r in means if r['dataset'] == 'macro'], paired=list(macro.values())), indent=2))


if __name__ == '__main__':
    main()
