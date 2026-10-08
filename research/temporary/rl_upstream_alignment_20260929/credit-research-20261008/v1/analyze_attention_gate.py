"""Describe a passive final-FA ledger on the frozen collection, without repair.

Joint-background contractions of single-deletion changes are not isolated
causal shares. Tail/body cells, cohorts, and exposure remain separate.
"""
import argparse
import json
from pathlib import Path

from analyze_layer_collection import ref
from summarize_author_collection import quantiles, stratum

HERE = Path(__file__).resolve().parent
TERMS = ('coefficient_recompute', 'product_background', 'sigmoid_rounded_secant',
         'sigmoid_background', 'native_sigmoid_rounding', 'native_product_rounding')
BRANCHES = ('core_and_input_residual', 'gate_residual', 'projection_residual')
INPUT_TERMS = ('input_projection_norm_RoPE_residual', 'finite_FA_core_residual')


def term_value(point, name):
    return point['final_FA_input_ledger'][name] if name in INPUT_TERMS else point['final_FA_gate_ledger'][name]


def state_frequency(points, predicate):
    states = {}
    for p in points:
        states.setdefault(p['initial_state_sha256'], {}).setdefault(p['traj_uid'], []).append(predicate(p))
    return sum(sum(sum(v)/len(v) for v in trajectories.values())/len(trajectories)
               for trajectories in states.values())/len(states) if states else None


def summarize(points):
    if not points:
        return dict(points=0, states=0, status='Empty cell; no imputation')
    names = TERMS + BRANCHES + (INPUT_TERMS if 'final_FA_input_ledger' in points[0] else ())
    result = dict(points=len(points), trajectories=len({p['traj_uid'] for p in points}),
                  states=len({p['initial_state_sha256'] for p in points}))
    result['conditional_finite_sample_values'] = {
        name: dict(signed=quantiles([term_value(p, name) for p in points]),
                   absolute=quantiles([abs(term_value(p, name)) for p in points]))
        for name in names}
    result['state_equal_largest_absolute_branch_frequency'] = {
        name: state_frequency(points, lambda p: name == max(BRANCHES,
            key=lambda n: abs(p['final_FA_gate_ledger'][n]))) for name in BRANCHES}
    result['state_equal_largest_absolute_gate_term_frequency'] = {
        name: state_frequency(points, lambda p: name == max(TERMS,
            key=lambda n: abs(p['final_FA_gate_ledger'][n]))) for name in TERMS}
    result['state_equal_negative_branch_frequency'] = {
        name: state_frequency(points, lambda p: p['final_FA_gate_ledger'][name] < 0) for name in BRANCHES}
    result['closures_and_recomputed_product'] = {
        name: quantiles([p['final_FA_gate_ledger'][name] for p in points]) for name in
        ('gate_decomposition_roundoff', 'whole_mixer_decomposition_roundoff', 'native_product_recompute_maxabs')}
    result['interpretation'] = 'Descriptive conditional distributions and bounded state-equal frequencies; no pooled tail mean, causal share, or repair acceptance.'
    if 'final_FA_input_ledger' in points[0]:
        result['state_equal_largest_absolute_core_term_frequency'] = {
            name: state_frequency(points, lambda p: name == max(INPUT_TERMS,
                key=lambda n: abs(term_value(p, n)))) for name in INPUT_TERMS}
        result['input_decomposition_roundoff'] = quantiles([
            p['final_FA_input_ledger']['decomposition_roundoff'] for p in points])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=('textcraft', 'appworld'))
    parser.add_argument('--attention-input', action='store_true')
    args = parser.parse_args()
    folder = HERE/(('attention-input-' if args.attention_input else 'attention-gate-')+args.task)
    paths = [folder/f'rank{r}.json' for r in (0, 1)]
    ranks = [json.loads(p.read_bytes()) for p in paths]
    assert all(r['phase'] == 'complete' for r in ranks)
    points = [p for r in ranks for b in r['batches'] for p in b['points']]
    baseline_paths = [HERE/f'layer-suboperations-{args.task}-v2-observations/rank{r}.json' for r in (0, 1)]
    baseline = {(p['traj_uid'], p['packed_slot']): p for f in baseline_paths
                for b in json.loads(f.read_bytes())['batches'] for p in b['points']}
    assert len(points) == len(baseline) == 165
    assert {(p['traj_uid'], p['packed_slot']) for p in points} == set(baseline)
    stable_path = HERE.parents[4]/'experiments/rl/results_stable_negative_credit_20261008.json'
    stable = json.loads(stable_path.read_bytes())['tasks'][args.task]['cohorts']
    flags = {(p['traj_uid'], p['packed_slot']): p for c in stable.values() for p in c['points_with_identity']}
    comparisons = []
    parity_groups = {'unchanged_DT_native_mixer': [], 'changed_DT_native_or_mixer': []}
    for p in points:
        old = baseline[p['traj_uid'], p['packed_slot']]
        comparison = dict(traj_uid=p['traj_uid'], packed_slot=p['packed_slot'],
            DT_difference=p['fresh_DT_d']-old['fresh_DT_d'],
            native_difference=p['native_single_d']-old['native_single_d'],
            original_mixer_difference=p['suboperations']['31']['matched_terms']['mixer']-
                old['suboperations']['31']['matched_terms']['mixer'],
            DT_factual_target_logp_difference=p['DT_factual_target_logp']-old['DT_factual_target_logp'],
            native_factual_target_logp_difference=p['factual_target_logp']-old['factual_target_logp'])
        comparisons.append(comparison)
        p['stable_flags'] = {name: flags[p['traj_uid'], p['packed_slot']][name] for name in
            ('spurious_DT_tail_all_references', 'missed_native_tail_all_references', 'probability_bound_violated_all_references')}
        name = ('unchanged_DT_native_mixer' if all(comparison[k] == 0 for k in
            ('DT_difference', 'native_difference', 'original_mixer_difference')) else 'changed_DT_native_or_mixer')
        parity_groups[name].append(p)
    cohorts = {}
    for cohort in ('uniform', 'predicted_tail_census'):
        rows = [p for p in points if any(c['cohort'] == cohort for c in p['previous_comparisons'])]
        cells = {}
        for p in rows:
            c = next(c for c in p['previous_comparisons'] if c['cohort'] == cohort)
            cell = stratum(c.get('saved_d', c.get('d')))+':'+stratum(c['native_single_d'])
            cells.setdefault(cell, []).append(p)
        cohorts[cohort] = dict(points=len(rows), crossed_ratio_cells={
            cell: dict(all=summarize(items), by_exposure={str(seen): summarize(
                [p for p in items if p['previously_examined'] == seen]) for seen in (False, True)})
            for cell, items in cells.items()}, robust_error_subsets={
            flag: dict(points=sum(p['stable_flags'][flag] for p in rows),
                states=len({p['initial_state_sha256'] for p in rows if p['stable_flags'][flag]}),
                crossed_ratio_cells={cell: summarize([p for p in items if p['stable_flags'][flag]])
                    for cell, items in cells.items()})
            for flag in next(iter(points))['stable_flags']})
    result = dict(scope=__doc__, task=args.task, complete=True, inputs=[ref(p) for p in paths+baseline_paths+[stable_path]],
        launch=ref(folder/'launch.json'), protocol=ref(HERE/('attention-input-protocol.json' if args.attention_input else 'attention-gate-protocol.json')),
        unique_points=len(points), cohorts=cohorts, unchanged_output_comparisons=comparisons,
        points=points, operations=[r['operations'] for r in ranks], elapsed_seconds=[r['elapsed_seconds'] for r in ranks],
        gate_readouts=[[b['attention_branch_readout'] for b in r['batches']] for r in ranks],
        maximum_observed_retained_host_bank_bytes=max(b['retained_bank_bytes']+b['retained_suboperation_bytes'] for r in ranks for b in r['batches']),
        original_DT_exact_equal_points=sum(c['DT_difference'] == 0 for c in comparisons),
        native_max_abs_difference=max(abs(c['native_difference']) for c in comparisons),
        original_mixer_max_abs_difference=max(abs(c['original_mixer_difference']) for c in comparisons),
        repeatability_strata={name: dict(points=len(items),
            robust_spurious_tail_from_original_run=summarize([p for p in items if p['stable_flags']['spurious_DT_tail_all_references']]))
            for name, items in parity_groups.items()},
        original_run_flags_revalidated_for_this_run=False,
        official_tolerance_test=False, candidate=False, production_modified=False, credit_repaired=False,
        interpretation='Core/input contains finite FA routing, Q/K normalization, RoPE and input projections; it is not a claim that routing alone causes the error. .25 is a smooth sigmoid bound, not a BF16 tolerance. Joint gate element counts repeat across queries and must not be summed as independent occurrences. Original robust flags identify frozen original-run subsets, not newly repeated confidence bounds. Changed runs remain explicitly separate in repeatability_strata; do not claim full unchanged DT parity or a new official whole-model threshold. No ledger term is subtracted from credit.')
    if args.attention_input:
        prior_paths = [HERE/('attention-gate-'+args.task)/f'rank{r}.json' for r in (0, 1)]
        prior = {(p['traj_uid'], p['packed_slot']): p for f in prior_paths
                 for b in json.loads(f.read_bytes())['batches'] for p in b['points']}
        result['inputs'] += [ref(p) for p in prior_paths]
        result['comparison_with_previous_gate_readout'] = [dict(traj_uid=p['traj_uid'], packed_slot=p['packed_slot'],
            DT_difference=p['fresh_DT_d']-prior[p['traj_uid'], p['packed_slot']]['fresh_DT_d'],
            native_difference=p['native_single_d']-prior[p['traj_uid'], p['packed_slot']]['native_single_d']) for p in points]
        result['comparison_scope'] = 'Both fixed prior references are reported; no per-token selection of whichever baseline is closest.'
    output = folder/'analysis.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=ref(output), points=len(points), identical_DT=result['original_DT_exact_equal_points'],
        native_max_abs_difference=result['native_max_abs_difference'], mixer_max_abs_difference=result['original_mixer_max_abs_difference'],
        elapsed_seconds=result['elapsed_seconds'], robust_error_subsets={k:{f:dict(points=v['points'], states=v['states'])
            for f,v in c['robust_error_subsets'].items()} for k,c in cohorts.items()})))


if __name__ == '__main__':
    main()
