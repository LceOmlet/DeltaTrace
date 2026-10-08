"""Original-loss gradients of measured coefficient differences, by frozen cohort."""
import json
import math
from pathlib import Path

from analyze_collection_gradients import receipt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
LABELS = ('full_pg', 'predicted_tail_error', 'uniform_bounded_error',
          'uniform_missed_native_tail_error')


def family(point):
    return ('predicted_tail_error' if point['cohort'] == 'predicted_tail_census'
            else 'uniform_bounded_error' if point['native_band'] in ('ratio_le_1','ratio_1_to_2')
            else 'uniform_missed_native_tail_error')


def geometry(stats, label):
    full, delta = stats['norms']['full_pg'], stats['norms'][label]
    dot = stats['inner_products']['full_pg:' + label]
    combined = math.sqrt(max(0., full * full + delta * delta + 2 * dot))
    cosine = (full * full + dot) / (full * combined) if full and combined else None
    return dict(delta_norm=delta, norm_ratio=delta / full if full else None,
        delta_projection_on_full=dot / (full * full) if full else None,
        cosine_delta_with_full=dot / (full * delta) if full and delta else None,
        full_plus_delta_norm=combined,
        full_plus_delta_angle_degrees=math.degrees(math.acos(max(-1.,min(1.,cosine))))
            if cosine is not None else None,
        interpretation='Geometric addition of a fixed-original-scale coefficient difference. Not a corrected/rewhitened update, parameter update, full-bulk estimate or evidence of a GDN cause.')


def measured_sum_geometry(stats, labels):
    """Sum disjoint selected-position loss vectors, not statistical stratum means."""
    def dot(a, b):
        return stats['inner_products'].get(a + ':' + b,
                                          stats['inner_products'].get(b + ':' + a))

    full = stats['norms']['full_pg']
    delta_squared = sum(dot(a, b) for a in labels for b in labels)
    delta = math.sqrt(max(0., delta_squared))
    full_dot_delta = sum(dot('full_pg', label) for label in labels)
    combined = math.sqrt(max(0., full * full + delta_squared + 2 * full_dot_delta))
    cosine = ((full * full + full_dot_delta) / (full * combined)
              if full and combined else None)
    return dict(labels=labels, delta_norm=delta, norm_ratio=delta / full if full else None,
        delta_projection_on_full=full_dot_delta / (full * full) if full else None,
        full_plus_selected_deltas_norm=combined,
        full_plus_selected_deltas_angle_degrees=math.degrees(math.acos(max(-1., min(1., cosine))))
            if cosine is not None else None,
        interpretation='Exact vector geometry from the measured native pairwise inner products and original loss denominators. Strata remain separate above. This sums only selected-position errors; it is neither a pooled raw mean nor an estimate of all unmeasured bulk errors or a corrected/rewhitened training update.')


def main():
    paths = list(HERE.glob('collection-error-gradient-observation-*.json'))
    assert paths
    latest = max(paths, key=lambda p:json.loads(p.read_bytes())['unix'])
    observed = json.loads(latest.read_bytes())
    coefficients = json.loads((HERE / 'collection-coefficients.json').read_bytes())
    files = observed['files']
    inspection = files['input-inspection.json']
    assert receipt(HERE / 'collection-coefficients.json')['sha256'] == inspection['coefficient_errors']['receipt']['sha256']
    minibatches = []
    for index in range(4):
        names = [f'minibatch{index}-rank{rank}.json' for rank in (0,1)]
        if not all(name in files for name in names):
            continue
        ranks = [files[name] for name in names]
        points = [p for p in coefficients['records'] if any(
            m['optimizer_minibatch'] == index for m in p['actor_matches'])]
        counts = {label:sum(family(p) == label for p in points) for label in LABELS[1:]}
        labels = [label for label in LABELS if label == 'full_pg' or counts[label]]
        assert ranks[0]['gradient_statistics'] == ranks[1]['gradient_statistics']
        for rank in ranks:
            assert rank['labels'] == labels and all(rank['parameters_exact_unchanged'].values())
            assert not rank['optimizer_step_executed'] and not rank['scheduler_step_executed']
            assert rank['effective_config']['ppo_micro_batch_size_per_gpu'] == 4
            for label in labels:
                passed = rank['passes'][label]
                assert len(passed['microbatch_losses']) == 8
                assert passed['optimizer_boundary']['optimizer_step_no_op_calls'] == 1
                assert math.isfinite(passed['optimizer_boundary']['native_clip_return'])
                assert all(m['pg_clipfrac'] == m['ppo_kl'] == m['pg_clipfrac_lower'] == 0
                           for m in passed['microbatch_losses'])
        stats = ranks[0]['gradient_statistics']
        assert all(math.isfinite(v) for v in stats['inner_products'].values())
        clip_returns = {label:[rank['passes'][label]['optimizer_boundary']['native_clip_return']
                              for rank in ranks] for label in labels}
        minibatches.append(dict(index=index, full_pg_norm=stats['norms']['full_pg'], counts=counts,
            components={label:dict(points=counts[label], states=len({p['initial_state_sha256']
                for p in points if family(p)==label}),
                original_actor_slots=sum(sum(m['optimizer_minibatch'] == index
                    for m in p['actor_matches']) for p in points if family(p)==label),
                state_counts={state:sum(p['initial_state_sha256'] == state and family(p)==label
                    for p in points) for state in sorted({p['initial_state_sha256']
                    for p in points if family(p)==label})}, geometry=geometry(stats,label))
                if label in labels else dict(points=0,status='Structurally absent; no native numerical pass claimed')
                for label in LABELS[1:]}, statistics=stats,
            selected_error_vector_sum=measured_sum_geometry(stats, labels[1:]),
            native_clip_returns=dict(by_rank=clip_returns,
                observed_euclidean_sum_of_rank_returns={label:math.hypot(*values)
                    for label,values in clip_returns.items()},
                scope='Retain the actual original returns. They differ by rank in this runtime; do not present one rank return as the mesh-wide FP64 pre-clip norm. The Euclidean sums above are descriptive comparisons, without a new numerical pass threshold or alteration of native clipping.'),
            elapsed_seconds_by_rank=[{k:v['elapsed_seconds'] for k,v in rank['passes'].items()} for rank in ranks]))
    phase = files.get('phase.json', {})
    all_returns = len(minibatches) == 4 and phase.get('phase') == 'original_minibatch_gradients_complete' and phase.get('minibatch') == 3
    budget = observed['launch']['wall_budget_seconds']
    budget_exit = all_returns and f'exceeded {budget} seconds' in observed['log_tail']
    complete = all_returns and ('completed.json' in files or (budget_exit and not observed['driver_same_birth']))
    result = dict(scope=__doc__, status='Complete original-owner measurement' if complete else 'Partial measurement',
        observed_unix=observed['unix'], launch=observed['launch'], observation=receipt(latest),
        inspection=inspection, minibatches=minibatches, final_phase=phase,
        completed=files.get('completed.json'), post_measurement_budget_exit=budget_exit,
        driver_same_birth=observed['driver_same_birth'], formal_releases=observed['text_releases'],
        coefficient_receipt=receipt(HERE / 'collection-coefficients.json'),
        scope_limits=['Predicted-tail census and uniform bounded/missed-tail samples remain separate.',
            'Uniform sample errors have no inverse-inclusion weights: these are selected-position contributions, not a whole-bulk estimate.',
            'All four minibatches use unchanged base LoRA, not sequential updates or the historical degradation window.',
            'Fixed original whitening scale and untouched original PPO. No native credit replaces any training value.',
            'No new GDN or kernel candidate; this measurement cannot identify the error-producing operator by itself.'])
    (HERE / 'collection-error-gradient-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    if complete:
        (REPO / 'experiments/rl/results_credit_collection_error_gradients_20261008.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], minibatches=[{k:v for k,v in m.items()
        if k not in ('statistics','elapsed_seconds_by_rank')} for m in minibatches])))


if __name__ == '__main__':
    main()
