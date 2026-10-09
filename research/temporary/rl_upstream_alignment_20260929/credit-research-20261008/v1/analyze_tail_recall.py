"""Audit negative-tail recall and error composition on frozen saved observations.

The uniform source sample and the all-development predicted-tail census have
different frames. Only the uniform sample supplies a recall denominator here.
No new model/reference query, candidate credit, or production change is made.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import time

from summarize_author_collection import quantiles

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
THRESHOLD = -math.log(2)  # Existing PLAN diagnostic boundary, not a new gate.


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def identity(point):
    return point['traj_uid'], point['packed_slot']


def status(interval):
    lo, hi = interval
    return 'tail' if hi < THRESHOLD else 'outside' if lo >= THRESHOLD else 'unresolved'


def recall(rows):
    positive = [p for p in rows if status(p['native_d_interval']) == 'tail']
    groups = {}
    for p in positive:
        groups.setdefault(p['initial_state_sha256'], []).append(p)
    cells = {actual + ':' + predicted: sum(
        status(p['native_d_interval']) == actual and status(p['DT_d_interval']) == predicted
        for p in rows) for actual in ('tail', 'outside', 'unresolved')
        for predicted in ('tail', 'outside', 'unresolved')}
    return dict(sampled_positions=len(rows), initial_states=len({p['initial_state_sha256'] for p in rows}),
        certified_native_tail_positions=len(positive), positive_initial_states=len(groups),
        positive_trajectories=len({p['traj_uid'] for p in positive}), confusion_cells=cells,
        observed_recalled_positions=cells['tail:tail'], observed_missed_positions=cells['tail:outside'],
        predicted_status_unresolved_on_certified_native_tail=cells['tail:unresolved'],
        observed_conditional_recall_range=(
            [cells['tail:tail']/len(positive),
             (cells['tail:tail']+cells['tail:unresolved'])/len(positive)] if positive else None),
        population_recall_estimate=None, population_recall_interval=None,
        interval_reason='Few positive initial states; a degenerate zero-hit bootstrap is not evidence of zero uncertainty. '
            'This frozen development sample does not establish representativeness for task-wide recall.',
        by_positive_initial_state={s:dict(positions=len(ps),
            recalled=sum(status(p['DT_d_interval']) == 'tail' for p in ps),
            missed=sum(status(p['DT_d_interval']) == 'outside' for p in ps)) for s, ps in groups.items()},
        interpretation='Observed conditional fraction only. No tail-census observations are added to this denominator; '
            'native threshold-crossing observations remain unresolved, not negative labels.')


def residual_summary(rows):
    if not rows:
        return dict(points=0, missing='Empty, not zero')
    names = ('global_DT_minus_native_d', 'FA_core', 'FA_QK_softmax', 'FA_PV_background',
             'FA_value', 'FA_PV_source_key', 'head_final_norm', 'head_logsoftmax', 'last_block_MLP')
    return dict(points=len(rows), states=len({p['initial_state_sha256'] for p in rows}),
        trajectories=len({p['traj_uid'] for p in rows}),
        conditional_quantiles={n:dict(signed=quantiles([p['diagnostics'][n] for p in rows]),
            absolute=quantiles([abs(p['diagnostics'][n]) for p in rows])) for n in names},
        counts=dict(head_and_FA_core_have_opposite_sign=sum(
            p['diagnostics']['head_total']*p['diagnostics']['FA_core'] < 0 for p in rows),
            head_logsoftmax_and_FA_PV_have_opposite_sign=sum(
            p['diagnostics']['head_logsoftmax']*p['diagnostics']['FA_PV_background'] < 0 for p in rows)),
        scope='Recorded same-call residuals, not independent errors, causal shares, replacement advantages or a repair forecast.')


def main():
    begin = time.perf_counter()
    stats_path = REPO/'experiments/rl/results_grouped_negative_support_20261009.json'
    stats = json.loads(stats_path.read_bytes())
    inputs_path = HERE/'author-original-artifacts/collection-inputs.json'
    inputs = json.loads(inputs_path.read_bytes())
    manifest_path = HERE/'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    rank_paths = [HERE/f'attention-pv-textcraft-v2/rank{r}.json' for r in (0, 1)]
    ledgers = {identity(p):p for path in rank_paths
        for b in json.loads(path.read_bytes())['batches'] for p in b['points']}
    region_path = HERE/'FA-PV-key-regions-v1/support.json'
    region_data = json.loads(region_path.read_bytes())
    assert region_data['phase'] == 'complete'
    regions = {identity(p):p for p in region_data['points']}
    tasks = {}
    for task, data in stats['tasks'].items():
        uniform = data['cohorts']['uniform']['points_with_identity']
        by_trajectory = {p['traj_uid']:[] for p in uniform}
        for p in uniform:
            by_trajectory[p['traj_uid']].append(p)
        source_sampling = []
        for entry in inputs['tasks'][task]:
            queries = entry['uniform_queries']
            count = entry['source_tokens']
            seed = int(hashlib.sha256((inputs['seed']+'\0uniform-development-diagnostic\0'+task+
                '\0'+entry['traj_uid']).encode()).hexdigest(), 16)
            assert [p['source_index'] for p in queries] == random.Random(seed).sample(range(count), 4)
            assert all(p['inclusion_probability'] == 4/count for p in queries)
            saved = by_trajectory[entry['traj_uid']]
            assert {identity(p) for p in saved} == {identity(dict(p, traj_uid=entry['traj_uid'])) for p in queries}
            source_sampling.append(dict(traj_uid=entry['traj_uid'], source_tokens=count,
                sampled_sources=4, inclusion_probability=4/count))
        cohorts = data['cohorts']
        tasks[task] = dict(
            sampling_audit=dict(trajectories=len(source_sampling), positions=sum(p['sampled_sources'] for p in source_sampling),
                original_seed_and_inclusion_probabilities_reproduced=True, trajectories_with_source_counts=source_sampling,
                uniform_frame='Frozen first-stage development trajectories: two hash-selected trajectories per initial state, four uniform sources each.',
                census_frame='All saved development captures with original predicted c>2; broader trajectory frame than the uniform sample.',
                all_development_trajectories=manifest['tasks'][task]['counts']['development']['trajectories'],
                estimand='Equal initial state, equal sampled trajectory, equal eligible source within trajectory. '
                    'This is not token-weighted deployed training load.',
                weighting='The uniform sample has equal four-source samples per trajectory; raw observed confusion counts are descriptive. '
                    'Inverse-inclusion weighting would be required for a token-total estimand, and cannot justify mixing the two frames without accounting for both designs.'),
            uniform_recall=recall(uniform),
            by_exposure={str(e):recall([p for p in uniform if p['previously_examined'] == e]) for e in (False, True)},
            by_original_native_magnitude_bin={b:recall([p for p in uniform if p['FP32_head_ratio_bin'] == b])
                for b in sorted({p['FP32_head_ratio_bin'] for p in uniform})},
            predicted_tail_census=dict(points=cohorts['predicted_tail_census']['all']['points'],
                recall=None, reason='Selection uses the prediction and a different trajectory frame; this census supports precision/error analysis, not recall.'),
            points=uniform)
        if task != 'textcraft':
            tasks[task]['operator_diagnosis'] = dict(missing='This saved FA/PV operand collection is TextCraft only; no AppWorld operator conclusion.')
            continue
        rows = []
        for cohort, saved_cohort in cohorts.items():
            for p in saved_cohort['points_with_identity']:
                ledger = ledgers[identity(p)]
                assert ledger['token_id'] == p['token_id']
                fa = ledger['final_FA_PV_ledger']
                head = ledger['suboperations']['head']['native_terms']
                rows.append(dict(p, cohort=cohort, diagnostics=dict(
                    global_DT_minus_native_d=ledger['fresh_DT_d']-ledger['native_single_d'],
                    FA_core=fa['native_pair_core_residual'], FA_QK_softmax=fa['joint_QK_softmax_residual'],
                    FA_PV_background=fa['PV_background_residual'], FA_value=fa['factual_P_value_residual'],
                    FA_PV_source_key=regions[identity(p)]['PV_key_regions']['at'],
                    head_final_norm=head['final_norm'], head_logsoftmax=head['logsoftmax_background'],
                    head_total=sum(head.values()), last_block_MLP=ledger['suboperations']['31']['native_terms']['MLP'])))
        tasks[task]['operator_diagnosis'] = dict(
            by_cohort={c:dict(all=residual_summary([p for p in rows if p['cohort'] == c]),
                robust_missed_tail=residual_summary([p for p in rows if p['cohort'] == c and p['robust_missed_negative_tail']]),
                robust_spurious_tail=residual_summary([p for p in rows if p['cohort'] == c and p['robust_spurious_negative_tail']]))
                for c in cohorts},
            robust_missed_points=[p for p in rows if p['robust_missed_negative_tail']],
            interpretation='Opposite residual signs can cancel. A smaller isolated FA residual cannot by itself predict a more accurate composed token score. '
                'The two missed TextCraft points belong to one state; they diagnose that state, not all missed tails.')
    result = dict(scope=__doc__, observed_unix=time.time(),
        sources=[ref(p) for p in (stats_path, inputs_path, manifest_path, *rank_paths, region_path, Path(__file__))],
        tasks=tasks, operations=dict(model=0, DT=0, FA=0, GPU=0, optimizer=0),
        production_modified=False, elapsed_seconds=time.perf_counter()-begin,
        conclusions=[
            'Recall requires independently sampled native-positive positions. The prior predicted-tail census and uniform sample are not simply pooled.',
            'Observed robust misses are TextCraft 2 positions in 1 state and AppWorld 4 positions in 3 states. These counts establish misses, not task-wide recall estimates.',
            'AppWorld native large-tail status crosses the threshold in 10 uniform positions; they remain unresolved.',
            'FP16 overflow repair remains the accepted version. This statistical/operator readout changes no credit, tolerance, production source, or training parameter.'
        ])
    output = REPO/'experiments/rl/results_tail_recall_20261009.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=ref(output), elapsed_seconds=result['elapsed_seconds'],
        recall={t:v['uniform_recall'] for t,v in tasks.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
