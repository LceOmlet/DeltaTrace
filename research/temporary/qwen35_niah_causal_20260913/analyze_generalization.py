"""Describe the frozen broader-task results, deployment identity and measured cost."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--archive-root', type=Path)
    a = p.parse_args()
    status = json.loads((a.run / 'status.json').read_bytes())
    protocol = json.loads((a.run / 'protocol.json').read_bytes())
    assert status['status'] == 'complete' and len(status['cases']) == protocol['expected_cases']
    assert hashlib.sha256((a.run / 'protocol.json').read_bytes()).hexdigest() == status['protocol_sha256']
    if 'input_cases_sha256' in protocol:
        assert hashlib.sha256((a.run / 'input_cases.json').read_bytes()).hexdigest() == protocol['input_cases_sha256'] == status['cases_sha256']
    expected = {(task, index) for task, indices in protocol['selection'].items() for index in indices}
    cases = []
    for item in status['cases']:
        file = a.run / item['path'] / 'results.json'
        assert hashlib.sha256(file.read_bytes()).hexdigest() == item['results_sha256']
        c = json.loads(file.read_bytes())
        assert c['status'] == 'complete'
        assert c['baseline_repeat_relative_l2'] < 1e-4
        if 'archive_dt_relative_l2' in c:
            assert c['archive_dt_relative_l2'] < 1e-4
        assert c['methods']['DT_original']['root_effect'] == c['methods']['DT_gdn_symmetric']['root_effect']
        cases.append(c)
    assert {(c['dataset'], c['index']) for c in cases} == expected
    history = []
    if a.archive_root:
        for c in cases:
            task, index = c['dataset'], c['index']
            source = json.loads((a.archive_root / task / 'results.json').read_bytes())
            old = next(row for row in source['cases'] if row['index'] == index)
            for field in ('input_ids', 'input_sha256', 'user_positions', 'keep', 'gold', 'prompt_length', 'target_length'):
                assert c[field] == old[field], (task, index, field)
            with np.load(a.archive_root / task / 'vectors.npz', allow_pickle=False) as archive, \
                    np.load(a.run / f'{task}_{index}' / 'vectors.npz', allow_pickle=False) as fresh:
                before = archive[f'{task}_{index}_DT_signed_full']
                now = fresh['DT_original_full_sequence']
                relative_l2 = float(np.linalg.norm(now - before) / max(np.linalg.norm(before), 1e-30))
            history.append(dict(dataset=task, index=index, input_identical=True,
                                root_effect_change=c['methods']['DT_original']['root_effect'] - old['DT_details']['root_effect'],
                                vector_relative_l2=relative_l2,
                                dt_rise_change=c['methods']['DT_original']['rise'] - old['metrics']['DT']['rise'],
                                ft_rise_change=c['FT_K1']['rise'] - old['metrics']['FT_K1']['rise']))
    methods = protocol['methods'] + ['FT_K1']
    means, paired = [], []
    rng = np.random.default_rng(20260913)
    bootstraps = {metric: [] for metric in ('rise', 'mas')}
    for task, indices in protocol['selection'].items():
        cc = sorted([c for c in cases if c['dataset'] == task], key=lambda c: c['index'])
        draws = rng.integers(0, len(cc), (10000, len(cc)))
        for method in methods:
            rr = [c['FT_K1'] if method == 'FT_K1' else c['methods'][method] for c in cc]
            means.append(dict(dataset=task, method=method, n=len(cc), **{
                metric: float(np.mean([r[metric] for r in rr])) for metric in ('rise', 'mas')}))
        for metric in ('rise', 'mas'):
            delta = np.array([c['methods']['DT_gdn_symmetric'][metric] - c['methods']['DT_original'][metric] for c in cc])
            boot = delta[draws].mean(1)
            bootstraps[metric].append(boot)
            paired.append(dict(dataset=task, metric=metric, n=len(cc), delta=float(delta.mean()),
                               ci95=np.quantile(boot, [.025, .975]).tolist(), better=int(sum(delta < 0)), equal=int(sum(delta == 0))))
    for method in methods:
        rr = [r for r in means if r['method'] == method]
        means.append(dict(dataset='macro', method=method, n=len(cases), **{
            metric: float(np.mean([r[metric] for r in rr])) for metric in ('rise', 'mas')}))
    for metric in ('rise', 'mas'):
        paired.append(dict(dataset='macro', metric=metric, n=len(cases),
                           delta=float(np.mean([r['delta'] for r in paired if r['metric'] == metric])),
                           ci95=np.quantile(np.mean(bootstraps[metric], axis=0), [.025, .975]).tolist()))
    first = cases[0]
    assert first['deployment_bitwise'] and first['candidate_repeat_bitwise']
    cost = []
    for method in protocol['methods']:
        rr = [r for r in first['profile_cost']['rows'] if r['method'] == method]
        assert len(rr) == 5
        times = [r['seconds'] for r in rr]
        cost.append(dict(method=method, repeats=5, mean_seconds=float(np.mean(times)), median_seconds=float(np.median(times)),
                         min_seconds=min(times), max_seconds=max(times), peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rr)))
    out = dict(status='complete', cases=len(cases), means=means, paired=paired, cost=cost,
               cost_case=dict(dataset=first['dataset'], index=first['index'], input_tokens=len(first['input_ids']), target_tokens=first['target_length']),
               cost_scope=first['profile_cost']['unit'],
               cost_time_ratio=cost[1]['median_seconds'] / cost[0]['median_seconds'],
               deployment_bitwise=first['deployment_bitwise'],
               max_archive_relative_l2=max((c['archive_dt_relative_l2'] for c in cases if 'archive_dt_relative_l2' in c), default=None),
               max_baseline_repeat_relative_l2=max(c['baseline_repeat_relative_l2'] for c in cases),
               max_archive_rise_difference=max((abs(c['archive_dt_rise_difference']) for c in cases if 'archive_dt_rise_difference' in c), default=None),
               max_archive_ft_rise_difference=max((abs(c['archive_ft_rise_difference']) for c in cases if 'archive_ft_rise_difference' in c), default=None),
               historical_diagnostics=history,
               baseline_policy=protocol.get('baseline_policy', 'Verified archival identities'),
               interpretation='Previously inspected benchmark, fixed indices; descriptive transfer check without candidate reselection. Cost applies only to the recorded single input and checkpoint/diagnostic configuration.')
    (a.run / 'generalization_analysis.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
