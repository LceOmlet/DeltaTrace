"""Read-only decomposition of already measured DT deletion-credit errors.

This is diagnostic arithmetic, not a new credit rule or method specification.
Keep the frozen sampling frame, crossed magnitude cells, state groups, and all
recorded references. No model, backward, optimizer, or remote call is made.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from summarize_tail_probability_sample import weighted_quantiles

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SOURCE = REPO / 'experiments/rl/results_probability_sample_credit_error_20261009.json'
GRADIENT = REPO / 'experiments/rl/results_credit_collection_error_gradients_20261008.json'
OUTPUT = REPO / 'experiments/rl/results_credit_bias_components_20261010.json'


def ref(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def components(d_hat, d_reference):
    ratio = math.exp(-d_reference)
    epsilon = d_hat - d_reference
    # The algebraic decomposition is e = c*epsilon + c*(1-exp(-epsilon)-epsilon).
    # Subtract the already finite ratios; avoid an unnecessary large intermediate
    # exp(-epsilon) when reference and prediction differ greatly.
    error = ratio - math.exp(-d_hat)
    linear = ratio * epsilon
    curvature = error - linear
    return dict(d_reference=d_reference, d_error=epsilon,
                A_over_r_error=error, first_order=linear, curvature=curvature,
                curvature_share_of_negative_error=(
                    curvature / error if epsilon < 0 and error < 0 else None))


def summarize(points):
    value = dict(sampled_positions=len(points),
                 sampled_states=len({p['initial_state_sha256'] for p in points}),
                 HT_source_total=math.fsum(p['weight'] for p in points))
    value['references'] = {}
    for name in points[0]['references']:
        values = [p['references'][name] for p in points]
        weights = [p['weight'] for p in points]
        negative = [(v, w) for v, w in zip(values, weights)
                    if v['curvature_share_of_negative_error'] is not None]
        value['references'][name] = dict(
            d_underestimated=sum(v['d_error'] < 0 for v in values),
            d_overestimated=sum(v['d_error'] > 0 for v in values),
            positive_curvature_from_scalar_rounding=sum(v['curvature'] > 0 for v in values),
            weighted_quantiles={key: weighted_quantiles([v[key] for v in values], weights)
                for key in ('d_error', 'A_over_r_error', 'first_order', 'curvature')},
            negative_error_curvature_share_weighted_quantiles=weighted_quantiles(
                [v['curvature_share_of_negative_error'] for v, _ in negative],
                [w for _, w in negative]),
            negative_error_positions=len(negative))
    return value


def main():
    started = time.perf_counter()
    cpu_started = time.process_time()
    source_ref, gradient_ref = ref(SOURCE), ref(GRADIENT)
    data = json.loads(SOURCE.read_bytes())
    gradient = json.loads(GRADIENT.read_bytes())
    result = dict(
        scope=__doc__, observed_unix=time.time(), method_owner='experiments/rl/PLAN.md',
        sources=[source_ref, gradient_ref, ref(Path(__file__)),
                 ref(HERE/'summarize_tail_probability_sample.py')],
        decomposition=dict(
            epsilon='d_DT-d_reference', c_reference='exp(-d_reference)',
            error='C_DT/r-C_reference/r=c_reference-exp(-d_DT)',
            first_order='c_reference*epsilon',
            curvature='c_reference*(1-exp(-epsilon)-epsilon)',
            sign='Curvature is nonpositive analytically; numerical positive values are retained and counted, not corrected.',
            meaning='Exact algebraic decomposition of the coefficient error, not an attribution of causal operator shares.',
            negative_error_share='Only epsilon<0 and error<0: curvature/error. No ratio across cancelling positive/negative terms.',
            reward_scope='Normalized credit error C/r for positive-reward captures. No task reward is recomputed.'),
        gradient_bias=dict(
            ideal_baseline='A common action-independent deletion-reference return B(h) cancels in the sampling-policy score expectation; it need not equal V(h).',
            actual_error='Delta_g=E[sum_i grad_log_pi_i*(C_DT_i-C_exact_i)] under that ideal-baseline condition.',
            entire_piecewise_rule='For the actual self-target/source rule define b_hat(h,a)=E[R*exp(-d)*1_nonself | h,a]. The raw gradient difference from score*R is -E[score*b_hat]. A source-only proof must not silently prove the whole piecewise rule.',
            necessary_softmax_condition='At one prefix with independent supported action logits, bias_a=-pi_a*(b_hat_a-E_pi[b_hat]). This is an analytic criterion, not a new B module or a measured estimate.',
            conservation='Zero mean coefficient error or a conserved credit sum does not imply the score-weighted error is zero.',
            implementation_gap='The production runner attributes a simultaneous all-source EOS contrast; its conserved components are not generally factual-background single-deletion log ratios when sources interact.',
            whitening='Subtracting a shared batch mean does not generally remove action-correlated error. Random same-batch mean/scale and PPO clipping have their own nonlinear effects.',
            no_claim='This study does not estimate the full model-gradient bias or establish training degradation.'),
        aggregation='Prediction/reference crossed magnitude cells and initial states remain separate. No pooled raw credit moment, cross-task mean, new confidence interval, or official numerical tolerance.',
        reference_scope='All three current saved native/dtype views plus both endpoints of the recorded reference interval. Endpoints are sensitivity records, not confidence limits or exact world-oracle bounds.',
        sampling='Exactly the existing frozen 1061 positions per task, unchanged inclusion probabilities and no appended historical queries. Conditional weighted quantiles are descriptive estimates; no sampling uncertainty is invented.',
        tasks={}, existing_owner_gradient_measurement=[],
        operations=dict(model=0, DT=0, backward=0, optimizer=0, GPU=0, rollout=0,
                        checkpoint_restore=0, remote=0),
        production_modified=False, credit_clipping_or_correction=False,
        official_FA_FLA_tolerance_changed=False)
    for task, native in data['tasks'].items():
        points = []
        for index, point in enumerate(native['observations']):
            references = {name: value['d'] for name, value in point['scores'].items()}
            references.update(recorded_interval_low=point['native_d_interval'][0],
                              recorded_interval_high=point['native_d_interval'][1])
            label = point['prediction_stratum'] + ':reference_' + '..'.join(point['reference_bin_interval'])
            points.append(dict(source_observation_index=index, traj_uid=point['traj_uid'],
                packed_slot=point['packed_slot'], token_id=point['token_id'],
                initial_state_sha256=point['initial_state_sha256'], cell=label,
                d_DT=point['saved_d'], inclusion_probability=point['inclusion_probability'],
                weight=point['token_total_weight'],
                references={name: components(point['saved_d'], value)
                            for name, value in references.items()}))
        assert len(points) == len(native['observations']) == 1061
        cells = sorted({p['cell'] for p in points})
        result['tasks'][task] = dict(frame=native['frame'], observations=points,
            crossed_cells={cell: summarize([p for p in points if p['cell'] == cell]) for cell in cells},
            by_initial_state={state: {cell: summarize([p for p in points
                if p['initial_state_sha256'] == state and p['cell'] == cell])
                for cell in cells if any(p['initial_state_sha256'] == state and p['cell'] == cell for p in points)}
                for state in sorted({p['initial_state_sha256'] for p in points})})
    for batch in gradient['minibatches']:
        result['existing_owner_gradient_measurement'].append(dict(
            minibatch=batch['index'], full_pg_norm=batch['full_pg_norm'],
            components={name: dict(points=value['points'], states=value.get('states'),
                geometry=value.get('geometry'), status=value.get('status'))
                for name, value in batch['components'].items()}))
    result['existing_gradient_scope_limits'] = gradient['scope_limits']
    result['existing_gradient_scope_note'] = (
        'Older 37-tail/128-uniform frame, fixed original whitening scale, unchanged base LoRA. '
        'Not the 1061 probability frame, not sequential optimizer updates, and not a current-formal-model gradient. '
        'It is reused without rerunning any backward or extrapolating unweighted sampled errors.')
    assert ref(SOURCE) == source_ref and ref(GRADIENT) == gradient_ref
    result['elapsed_seconds'] = time.perf_counter() - started
    result['CPU_process_seconds'] = time.process_time() - cpu_started
    assert not OUTPUT.exists(), 'Preserve an existing completed diagnostic; do not overwrite it'
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=ref(OUTPUT), elapsed_seconds=result['elapsed_seconds'],
                         CPU_process_seconds=result['CPU_process_seconds'],
                         tasks={k: dict(positions=len(v['observations']), cells=len(v['crossed_cells']))
                                for k,v in result['tasks'].items()})))


if __name__ == '__main__':
    main()
