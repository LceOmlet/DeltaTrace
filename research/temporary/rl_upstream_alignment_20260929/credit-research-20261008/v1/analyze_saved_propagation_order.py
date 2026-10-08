"""Locate crossings in saved passive contractions, without new model queries.

This is a diagnostic of the existing joint-background finite rule applied to
actual single-deletion differences. Crossing a threshold is not a causal
ablation and does not establish which replacement operator would fix it.
"""
import collections
import json
import math
from pathlib import Path

from analyze_layer_collection import ref
from summarize_author_collection import stratum

HERE = Path(__file__).resolve().parent
FLAGS = HERE.parents[4] / 'experiments/rl/results_stable_negative_credit_20261008.json'
REVERSE_OPERATIONS = ('residual_add2', 'MLP', 'norm2', 'residual_add1', 'mixer', 'norm1')


def points(paths):
    return {(p['traj_uid'], p['packed_slot']): p for path in paths
            for b in json.loads(path.read_bytes())['batches'] for p in b['points']}


def trace(p, sub, mode):
    boundary_key = 'value' if mode == 'native' else 'DT_factual_minus_native_deleted_effect'
    score = p['native_single_d'] + (p['DT_minus_native_factual_score'] if mode == 'matched' else 0)
    head = sub['suboperations']['head']
    rows = [dict(location='native_target_score_difference', value=score, resolution='score'),
            dict(location='head/logsoftmax_background', value=head['logits'][mode], resolution='operation'),
            dict(location='head/linear_projection', value=head['recomputed_norm'][mode], resolution='operation'),
            dict(location='head/seed_projection_recompute', value=head['actual_norm'][mode], resolution='operation'),
            dict(location='head/final_norm', value=p['boundaries']['32'][boundary_key], resolution='operation')]
    closure = []
    for layer in range(31, -1, -1):
        location = str(layer)
        if location in sub['suboperations']:
            value = rows[-1]['value']
            for name in REVERSE_OPERATIONS:
                value += sub['suboperations'][location][mode + '_terms'][name]
                rows.append(dict(location=location + '/' + name, value=value, resolution='operation'))
            closure.append(dict(layer=layer, roundoff=value - p['boundaries'][location][boundary_key]))
        else:
            rows.append(dict(location=location + '/whole_decoder',
                             value=p['boundaries'][location][boundary_key], resolution='whole_decoder'))
    # Retain the difference between the saved embedding contraction and the
    # reported source attribution; never silently identify them.
    rows.append(dict(location='embedding_to_reported_source', value=p['fresh_DT_d'], resolution='source_readout'))
    return rows, closure


def crossings(rows, threshold):
    entries = [dict(index=i, location=rows[i]['location'], resolution=rows[i]['resolution'],
                    before=rows[i-1]['value'], after=rows[i]['value'])
               for i in range(1, len(rows))
               if rows[i-1]['value'] >= threshold and rows[i]['value'] < threshold]
    exits = [dict(index=i, location=rows[i]['location'], resolution=rows[i]['resolution'],
                  before=rows[i-1]['value'], after=rows[i]['value'])
             for i in range(1, len(rows))
             if rows[i-1]['value'] < threshold and rows[i]['value'] >= threshold]
    return dict(threshold=threshold, start_below=rows[0]['value'] < threshold,
                end_below=rows[-1]['value'] < threshold, entries=entries, exits=exits,
                first_entry=entries[0] if entries else None,
                last_entry=entries[-1] if entries else None)


def describe(items, mode):
    result = dict(points=len(items), states=len({p['initial_state_sha256'] for p in items}),
                  trajectories=len({p['traj_uid'] for p in items}))
    for measure in ('negative_sign', 'ratio_above_2', 'implied_probability_above_1'):
        measures = [p['modes'][mode]['crossings'][measure] for p in items]
        result[measure] = dict(start_below=sum(m['start_below'] for m in measures),
                              end_below=sum(m['end_below'] for m in measures),
                              multiple_entries=sum(len(m['entries']) > 1 for m in measures))
        for order in ('first_entry', 'last_entry'):
            labels = [m[order]['location'] if m[order] else 'no_entry' for m in measures]
            # Source -> trajectory -> initial state: do not let a long tail
            # trajectory dominate a bounded frequency.
            grouped = {}
            for p, label in zip(items, labels):
                grouped.setdefault(p['initial_state_sha256'], {}).setdefault(p['traj_uid'], []).append(label)
            result[measure][order] = dict(counts=dict(collections.Counter(labels)),
                state_equal_frequency={label: math.fsum(
                    math.fsum(sum(v == label for v in values)/len(values) for values in trajectories.values())/len(trajectories)
                    for trajectories in grouped.values())/len(grouped)
                    for label in sorted(set(labels))} if grouped else {})
    return result


def main():
    stable = json.loads(FLAGS.read_bytes())['tasks']
    cfg_path = HERE / 'gdn-owner-readonly.json'
    layer_types = json.loads(cfg_path.read_bytes())['model_config']['text_config']['layer_types']
    result = dict(scope=__doc__, inputs=[ref(FLAGS), ref(cfg_path), ref(Path(__file__))],
                  operations=dict(model_forward=0, DT=0, optimizer=0, rollout=0),
                  production_modified=False, official_tolerance_test=False, tasks={})
    for task in ('textcraft', 'appworld'):
        baseline_paths = [HERE/f'layer-factual-controls-{task}-observations/rank{r}.json' for r in (0, 1)]
        sub_paths = [HERE/f'layer-suboperations-{task}-v2-observations/rank{r}.json' for r in (0, 1)]
        baseline, subs = points(baseline_paths), points(sub_paths)
        assert set(baseline) == set(subs) and len(baseline) == 165
        flags = {(p['traj_uid'], p['packed_slot']): p for c in stable[task]['cohorts'].values()
                 for p in c['points_with_identity']}
        rows = []
        for key, p in baseline.items():
            sub = subs[key]
            assert p['fresh_DT_d'] == sub['fresh_DT_d']
            assert all(p['boundaries'][k] == v for k, v in sub['boundaries'].items())
            row = {k: p[k] for k in ('traj_uid', 'packed_slot', 'token_id', 'initial_state_sha256', 'previously_examined')}
            row.update(native_single_d=p['native_single_d'], DT_d=p['fresh_DT_d'],
                       robust_flags={k: flags[key][k] for k in ('spurious_DT_tail_all_references',
                           'missed_native_tail_all_references', 'probability_bound_violated_all_references')},
                       cohorts=[c['cohort'] for c in p['previous_comparisons']], modes={})
            row['ratio_cell'] = stratum(flags[key]['DT_d']) + ':' + stratum(flags[key]['FP32_head_d'])
            for mode in ('native', 'matched'):
                path, closure = trace(p, sub, mode)
                factual_score = p['factual_target_logp'] + (p['DT_minus_native_factual_score'] if mode == 'matched' else 0)
                row['modes'][mode] = dict(path=path, closure_roundoff=closure,
                    crossings={name: crossings(path, threshold) for name, threshold in (
                        ('negative_sign', 0.), ('ratio_above_2', -math.log(2.)),
                        ('implied_probability_above_1', factual_score))})
            rows.append(row)
        cohorts = {}
        for cohort in ('uniform', 'predicted_tail_census'):
            selected = [p for p in rows if cohort in p['cohorts']]
            cells = {}
            for cell in sorted({p['ratio_cell'] for p in selected}):
                items = [p for p in selected if p['ratio_cell'] == cell]
                cells[cell] = dict(all={mode: describe(items, mode) for mode in ('native', 'matched')},
                    by_exposure={str(exposure): {mode: describe([p for p in items if p['previously_examined'] == exposure], mode)
                        for mode in ('native', 'matched')} for exposure in (False, True)},
                    robust_error_subsets={flag: {mode: describe([p for p in items if p['robust_flags'][flag]], mode)
                        for mode in ('native', 'matched')} for flag in rows[0]['robust_flags']})
            cohorts[cohort] = dict(points=len(selected), ratio_cells=cells)
        result['tasks'][task] = dict(inputs=[ref(p) for p in baseline_paths + sub_paths], points=rows, cohorts=cohorts,
            robust_spurious_tail_summary={mode: describe([p for p in rows if p['robust_flags']['spurious_DT_tail_all_references']], mode)
                                         for mode in ('native', 'matched')},
            measured_decoder_suboperations=[18, 30, 31], layer_types=layer_types,
            maximum_suboperation_closure_roundoff=max(abs(c['roundoff']) for p in rows
                for mode in p['modes'].values() for c in mode['closure_roundoff']))
        if task == 'textcraft':
            variants = {}
            for label, name, field in (
                ('FA', 'conditional-attention-owner-v1/textcraft-v3-analysis.json', 'fresh_conditional_d'),
                ('GDN', 'conditional-gdn-collection-v1/textcraft-analysis.json', 'fresh_conditional_gdn_d'),
                ('FA_GDN', 'conditional-mixers-collection-v1/textcraft-analysis.json', 'fresh_conditional_mixers_d')):
                source_path = HERE/name
                data = json.loads(source_path.read_bytes())
                variants[label] = {(p['traj_uid'], p['packed_slot']): p[field]
                                   for c in data['cohorts'].values() for p in c['points_with_identity']}
                result['tasks'][task]['inputs'].append(ref(source_path))
                assert set(variants[label]) == set(baseline)
            for row in rows:
                row['existing_candidate_d'] = {name: values[row['traj_uid'], row['packed_slot']]
                                               for name, values in variants.items()}
            false_tail = [p for p in rows if p['robust_flags']['spurious_DT_tail_all_references']]
            comparisons = {}
            for mode in ('native', 'matched'):
                groups = {'all_robust_false_tail': false_tail}
                for p in false_tail:
                    entry = p['modes'][mode]['crossings']['negative_sign']['first_entry']
                    groups.setdefault(entry['location'] if entry else 'no_entry', []).append(p)
                comparisons[mode] = {name: dict(points=len(items),
                    states=len({p['initial_state_sha256'] for p in items}),
                    no_longer_tail={label: sum(p['existing_candidate_d'][label] >= -math.log(2.) for p in items)
                                    for label in variants}) for name, items in groups.items()}
            lost_repairs = {}
            for label in ('FA', 'GDN'):
                items = [p for p in false_tail if p['existing_candidate_d'][label] >= -math.log(2.)
                         and p['existing_candidate_d']['FA_GDN'] < -math.log(2.)]
                lost_repairs[label] = dict(points=len(items), states=len({p['initial_state_sha256'] for p in items}),
                    identities=[dict(traj_uid=p['traj_uid'], packed_slot=p['packed_slot'],
                                     candidate_d=p['existing_candidate_d']) for p in items])
            result['tasks'][task]['existing_candidate_comparison'] = dict(by_first_sign_entry=comparisons,
                standalone_tail_repairs_lost_in_composition=lost_repairs,
                interpretation='Observed existing outcomes, not independent causal contributions. No output selected per token; no new candidate evaluated.')
    result['interpretation'] = ('These are saved joint-background coefficients applied to actual native single-deletion '
        'changes. Paths are telescoping diagnostic contractions, not new policy values or corrected credit. '
        'Intermediate contractions are not log-probability ratios and are not required to satisfy probability bounds. '
        'First entry can be reversed later; last entry and all recoveries are retained. An unmeasured decoder '
        'is explicitly unresolved. Operation crossings and state-equal frequencies locate a discrepancy but '
        'do not prove an isolated kernel bug or a sufficient fix. Existing ratio bins are diagnostics, not tolerances.')
    output = HERE/'saved-propagation-order.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), identity=ref(output), tasks={t: dict(
        points=len(v['points']), closure_roundoff=v['maximum_suboperation_closure_roundoff'],
        robust_false_tail_points=v['robust_spurious_tail_summary']['native']['points'],
        first_sign_crossings=v['robust_spurious_tail_summary']['native']['negative_sign']['first_entry']['counts'])
        for t, v in result['tasks'].items()})))


if __name__ == '__main__':
    main()
