"""Describe original native-reader probes; no training signal or tolerance gate."""
import hashlib
import json
import math
from pathlib import Path
import statistics


LOCAL = Path(__file__).parent / 'textcraft-degradation-20261005/readout-quality-20261006/v2'


def source(path):
    return dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def describe(values):
    values = [float(x) for x in values]
    assert values and all(math.isfinite(x) for x in values)
    ordered = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values), abs_mean=statistics.mean(abs(x) for x in values),
                rms=math.sqrt(statistics.mean(x * x for x in values)),
                positive=sum(x > 0 for x in values), negative=sum(x < 0 for x in values),
                zero=sum(x == 0 for x in values),
                quantiles={str(q): ordered[round(q * (len(values) - 1))]
                           for q in (0, .1, .25, .5, .75, .9, 1)})


def pearson(xs, ys):
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    syy = sum((y - ybar) ** 2 for y in ys)
    return (sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)
            if sxx > 0 and syy > 0 else None)


def main():
    paths = [LOCAL / f'rank{rank}-readout.json' for rank in range(2)]
    completed = json.loads((LOCAL / 'completed.json').read_bytes())
    runs = [json.loads(path.read_bytes()) for path in paths]
    assert all(row['phase'] == 'complete_native_readout' for row in runs)
    assert [len(row['cases']) for row in runs] == [32, 32]
    assert all(row['optimizer_steps'] == row['scheduler_steps'] == row['backward_calls'] ==
               row['finite_trace_calls'] == 0 for row in runs)
    assert [row['native_forward_calls'] for row in runs] == [33, 33]
    cases = [dict(row, rank=run['rank']) for run in runs for row in run['cases']]
    assert len({row['traj_uid'] for row in cases}) == 64
    assert all(row['source_step'] == 0 for row in cases)
    successes = [row for row in cases if row['observed_return'] == 1]
    failures = [row for row in cases if row['observed_return'] == 0]
    assert len(successes) == 21 and len(failures) == 43
    probability = lambda row, variant=0: math.exp(row['native_outcome_log_probs'][variant][1])
    outcomes = [row['observed_return'] for row in cases]
    ps = [probability(row) for row in cases]
    factual_brier = statistics.mean((p - g) ** 2 for p, g in zip(ps, outcomes))
    batch_fraction = statistics.mean(outcomes)
    pools = {}
    for name, rows in [('all', cases), ('observed_success', successes), ('observed_failure', failures)]:
        pools[name] = dict(n=len(rows),
            factual_p_success=describe([probability(row) for row in rows]),
            full_response_eos_p_success=describe([probability(row, 1) for row in rows]),
            p_success_factual_minus_full_eos=describe([probability(row) - probability(row, 1) for row in rows]),
            full_response_target_log_ratio=describe([row['native_full_span_log_ratio'] for row in rows]),
            target_log_ratio_scope='Original observed class: success class on G1, failure class on G0; the all pool mixes target classes.')
    pairs = []
    for row in successes:
        for index, saved in enumerate(row['saved_probe_credit']):
            assert saved['d'] is not None
            dt = float(saved['d'])
            native = float(row['native_single_log_ratios'][index])
            pairs.append(dict(traj_uid=row['traj_uid'], rank=row['rank'],
                              source_step=0, source_position=row['probe_source_positions'][index],
                              token_id=row['probe_token_ids'][index], dt_d=dt,
                              native_single_delete_d=native, difference=dt-native,
                              saved_A=saved['A'], saved_Q=saved['Q'], saved_V=saved['V']))
    assert len(pairs) == 42
    xs, ys = [row['dt_d'] for row in pairs], [row['native_single_delete_d'] for row in pairs]
    opposite = [row for row in pairs if row['dt_d'] * row['native_single_delete_d'] < 0]
    repeats = [row['same_input_repeat'] for row in cases if row['same_input_repeat'] is not None]
    assert len(repeats) == 2
    prior_path = LOCAL.parent.parent / 'native-minibatch-v4/native64-first-response-existing-endpoints.json'
    prior = json.loads(prior_path.read_bytes())
    previous_by_uid = {row['traj_uid']: row for row in prior['endpoints']}
    endpoint_pairs = []
    for row in successes:
        previous = previous_by_uid[row['traj_uid']]
        assert previous['native_first_case_literal_ID_match'] and previous['source_step'] == 0
        scores = previous['pooled_duplicate_scores']
        endpoint_pairs.append(dict(traj_uid=row['traj_uid'],
            native_factual_lp=row['native_target_log_probs'][0],
            previous_factual_lp=scores['factual_target_logp']['mean'],
            native_full_eos_lp=row['native_target_log_probs'][1],
            previous_full_eos_lp=scores['reference_target_logp']['mean'],
            factual_difference=row['native_target_log_probs'][0]-scores['factual_target_logp']['mean'],
            full_eos_difference=row['native_target_log_probs'][1]-scores['reference_target_logp']['mean'],
            root_effect_difference=row['native_full_span_log_ratio']-scores['root_effect']['mean'],
            previous_transport_slots=previous['transport_slots']))
    result = dict(scope=__doc__, status='native_readout_descriptions_not_quality_repair',
        checkpoint=runs[0]['checkpoint'], selection=runs[0]['selection'],
        sources=[source(path) for path in paths] + [source(LOCAL / 'completed.json'), source(Path(__file__)), source(prior_path)],
        deployed_sources=[run['sources'] for run in runs],
        calls=dict(native_forwards=66, finite_attributions=0, backward=0, optimizer=0, scheduler=0),
        physical_devices=[4, 5], cases=64, prompt_groups=8,
        dtype=[dict(model=run['model_dtype'], native_fla_fp16=run['native_fla_fp16'],
                    categorical_log_prob_dtypes=sorted({row['native_logits_dtype'] for row in run['cases']}),
                    attention=run['event_attention_backend']) for run in runs],
        observed_success_fraction=batch_fraction,
        factual_forecast_brier=factual_brier,
        batch_fraction_constant_brier=statistics.mean((batch_fraction-g)**2 for g in outcomes),
        constant_brier_scope='Descriptive fitted-to-this-batch constant, not an out-of-sample predictive model.',
        factual_forecast_observed_target_nll=statistics.mean(-row['native_target_log_probs'][0] for row in cases),
        observed_outcome_vs_factual_probability_pearson=pearson(ps, outcomes),
        pools=pools,
        direct_vs_saved_success_token_probe=dict(
            selection='Two fixed-seed uniformly sampled distinct positions per actual first response; 21 successful trajectories, 42 dependent token samples.',
            dt_d=describe(xs), native_single_delete_d=describe(ys),
            paired_error=describe([row['difference'] for row in pairs]),
            pearson=pearson(xs, ys),
            paired_native_over_dt_absmean=statistics.mean(abs(y) for y in ys)/statistics.mean(abs(x) for x in xs),
            opposite_sign_count=len(opposite), denominator=42,
            opposite_sign_native_absolute_mass=sum(abs(row['native_single_delete_d']) for row in opposite),
            total_native_absolute_mass=sum(abs(y) for y in ys), pairs=pairs),
        same_native_input_repetition=repeats,
        previous_attribute_endpoint_comparison=dict(n=21,
            scope='Exact original selected IDs; native-reader B4 variant layout versus previous paired attribute batches with original dense EOS suffix. Descriptive, not a new tolerance gate.',
            factual_lp_difference=describe([row['factual_difference'] for row in endpoint_pairs]),
            full_eos_lp_difference=describe([row['full_eos_difference'] for row in endpoint_pairs]),
            root_effect_difference=describe([row['root_effect_difference'] for row in endpoint_pairs]),
            pairs=endpoint_pairs),
        case_seconds=describe([row['seconds'] for row in cases]),
        case_timing_scope='64 case timers include66 forwards: first case on each rank includes its same-input repeat; not64 single-forward times.',
        completed_unix=completed['completed_unix'],
        limitations=[
            'One checkpoint and original eight task groups; not 64 independent tasks or a historical bad-update replay.',
            'Native single deletion is the same model counterfactual, not an environment/world oracle.',
            'Forecast probabilities use original categorical readout; practical calibration is checked separately from the accepted exact-token-value premise.',
            'No failing-trajectory DT d was measured; G0 fixed-formula A=0 is not a missing action mask.',
            'Saved d is diagnostically inverted from the original FP32 A; it is not a new training credit.',
            'Descriptive differences/repetition are not new numerical tolerance assertions or official FA/FLA tests.',
            'All intervention rows retain the exact original prefix length and target predictor; no cross-prefix EOS padding.',
            'These observations alone do not establish Adam update direction or a repair of historical training.'])
    (LOCAL / 'native-readout-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['cases','observed_success_fraction','factual_forecast_brier',
        'observed_outcome_vs_factual_probability_pearson','pools']}, ensure_ascii=False, indent=2))
    print(json.dumps({key:result['direct_vs_saved_success_token_probe'][key] for key in
                     ['dt_d','native_single_delete_d','paired_error','pearson','paired_native_over_dt_absmean',
                      'opposite_sign_count','denominator']}, indent=2))


if __name__ == '__main__':
    main()
