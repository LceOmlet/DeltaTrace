"""Describe original paired-layout scores; no model call or tolerance gate.

The native observer keeps seven original B4 request groups per rank.  Its
scores stop before finite propagation.  This analysis retains transport
replicas, and labels the separate reduction to unique response/probe identities.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from analyze_textcraft_native_readout import describe, pearson, source


AUDIT = Path(__file__).resolve().parent
LOCAL = AUDIT / 'textcraft-degradation-20261005/native-layout-20261006/v1'
PACK = LOCAL.parent / 'native64-first-response-matched-layout-pack.json'
PRIOR = AUDIT / 'textcraft-degradation-20261005/readout-quality-20261006/v2'
OLD_ENDPOINTS = AUDIT / 'textcraft-degradation-20261005/native-minibatch-v4/native64-first-response-existing-endpoints.json'
PACK_SHA = '2e8390747c02801bdb763e4eac76112c0ac4caf3f513893f8218e7124e560f27'
VARIANTS = ('full_response_eos', 'single_eos_0', 'single_eos_1')


def paired_statistics(rows, left, right):
    """Pure descriptive statistics; orientation is left minus right."""
    xs, ys = [float(row[left]) for row in rows], [float(row[right]) for row in rows]
    errors = [x - y for x, y in zip(xs, ys)]
    opposite = [row for row in rows if row[left] * row[right] < 0]
    xmass, ymass = sum(abs(x) for x in xs), sum(abs(y) for y in ys)
    sign = lambda value: 'positive' if value > 0 else 'negative' if value < 0 else 'zero'
    return dict(n=len(rows), left=left, right=right,
        difference_orientation=f'{left} minus {right}',
        left_values=describe(xs), right_values=describe(ys),
        difference=describe(errors), pearson=pearson(xs, ys),
        mae=statistics.mean(abs(x) for x in errors),
        rmse=math.sqrt(statistics.mean(x * x for x in errors)),
        sign_counts=dict(Counter(f'{sign(x)}|{sign(y)}' for x, y in zip(xs, ys))),
        opposite_sign_count=len(opposite),
        opposite_sign_left_absolute_mass=sum(abs(row[left]) for row in opposite),
        opposite_sign_right_absolute_mass=sum(abs(row[right]) for row in opposite),
        total_left_absolute_mass=xmass, total_right_absolute_mass=ymass,
        opposite_sign_left_mass_fraction=(sum(abs(row[left]) for row in opposite) / xmass if xmass else None),
        opposite_sign_right_mass_fraction=(sum(abs(row[right]) for row in opposite) / ymass if ymass else None),
        right_over_left_abs_mean=(ymass / xmass if xmass else None))


def reduce_replicas(rows, identity_fields, value_fields):
    """Keep every original row, explicitly compute mean/min/max over replicas."""
    pools = defaultdict(list)
    for row in rows:
        pools[tuple(row[key] for key in identity_fields)].append(row)
    unique = []
    for identity, members in sorted(pools.items()):
        values = {key: dict(mean=statistics.mean(row[key] for row in members),
                           min=min(row[key] for row in members),
                           max=max(row[key] for row in members),
                           range=max(row[key] for row in members)-min(row[key] for row in members))
                  for key in value_fields}
        unique.append(dict(zip(identity_fields, identity), transport_slots=len(members),
            transport_locations=[dict(rank=row['rank'], original_owner_batch_index=row['original_owner_batch_index'],
                                      slot=row['slot']) for row in members],
            replica_values=values,
            **{key: values[key]['mean'] for key in value_fields}))
    return dict(scope='Each unique identity receives the arithmetic mean of its retained transport replicas; all replica values and ranges are reported separately.',
        identity_fields=list(identity_fields), transport_slots=len(rows), unique_identities=len(unique),
        duplicated_identities=sum(row['transport_slots'] > 1 for row in unique),
        max_replica_ranges={key: max(row['replica_values'][key]['range'] for row in unique) for key in value_fields},
        rows=unique)


def scores_for_slot(case, slot):
    values = case['target_log_probs']
    assert len(values) == 8
    return float(values[2 * slot + 1]), float(values[2 * slot])


def build_analysis(input_dir, pack_path, prior_dir, old_endpoint_path):
    paths = [input_dir / f'rank{rank}-readout.json' for rank in range(2)]
    runs = [json.loads(path.read_bytes()) for path in paths]
    pack = json.loads(pack_path.read_bytes())
    assert source(pack_path)['sha256'] == PACK_SHA
    assert all(run['input_sha256'] == PACK_SHA and run['phase'] == 'complete_native_readout' for run in runs)
    assert [run['rank'] for run in runs] == [0, 1]
    assert [len(run['cases']) for run in runs] == [21, 21]
    assert all(run['optimizer_steps'] == run['scheduler_steps'] == run['backward_calls'] ==
               run['finite_trace_calls'] == run['finite_seed_calls'] == 0 for run in runs)
    assert runs[0]['checkpoint'] == runs[1]['checkpoint']
    previous = json.loads(old_endpoint_path.read_bytes())
    previous_by_uid = {row['traj_uid']: row for row in previous['endpoints']}
    prior_paths = [prior_dir / f'rank{rank}-readout.json' for rank in range(2)]
    prior_runs = [json.loads(path.read_bytes()) for path in prior_paths]
    assert all(run['phase'] == 'complete_native_readout' and run['checkpoint'] == runs[0]['checkpoint']
               for run in prior_runs)
    prior_cases = [dict(case, rank=run['rank']) for run in prior_runs for case in run['cases']]
    prior_by_uid = {case['traj_uid']: case for case in prior_cases}
    assert len(prior_by_uid) == len(prior_cases) == 64

    baseline_rows, probe_rows, groups, unchanged_rows = [], [], [], []
    for rank, run in enumerate(runs):
        case_by_group_variant = {(case['original_owner_batch_index'], case['variant']): case
                                 for case in run['cases']}
        assert len(case_by_group_variant) == 21
        for group in pack['rank_groups'][rank]:
            index = group['original_owner_batch_index']
            cases = {variant: case_by_group_variant[index, variant] for variant in VARIANTS}
            baseline = cases['full_response_eos']
            assert all(case['samples'] == group['samples'] for case in cases.values())
            assert all(case['stopped_before_finite_seed'] and not case['finite_seed_called']
                       for case in cases.values())
            assert all(case['original_sequence_length'] == len(group['selected_input_ids'][0]) for case in cases.values())
            groups.append(dict(rank=rank, original_owner_batch_index=index,
                original_sequence_length=baseline['original_sequence_length'],
                source_derived_original_prefix=group['prefix_start_original_cross_rank_MIN_derived'],
                actual_variants=[dict(variant=variant, actual_prefix=case['observed_prefix_length'],
                    fixed_cut=case['fixed_cut'], selection=case['selection'],
                    changed_first_response_slots=case['changed_first_response_slots'],
                    native_forward_calls=case['native_forward_calls'], sync_calls=case['sync_calls'])
                    for variant, case in cases.items()]))
            for slot, sample in enumerate(group['samples']):
                factual, full_eos = scores_for_slot(baseline, slot)
                old = sample['original_trace']
                root = factual - full_eos
                common = dict(rank=rank, original_owner_batch_index=index, slot=slot,
                    traj_uid=sample['traj_uid'], source_step=sample['source_step'],
                    original_request_index=sample['original_request_index'],
                    source_start=sample['source_start'], source_end=sample['source_end'],
                    context_tokens=sample['context_tokens'], compute_tokens=sample['compute_tokens'])
                baseline_rows.append(dict(common, matched_factual_lp=factual, matched_full_eos_lp=full_eos,
                    matched_joint_root=root, old_factual_lp=old['factual_target_logp'],
                    old_full_eos_lp=old['reference_target_logp'], old_joint_root=old['root_effect'],
                    old_signed_sum=old['signed_sum']))
                for probe_index, variant in enumerate(VARIANTS[1:]):
                    single_factual, single_reference = scores_for_slot(cases[variant], slot)
                    if sample['source_step'] != 0:
                        unchanged_rows.append(dict(common, variant=variant,
                            baseline_factual_lp=factual, unchanged_variant_factual_lp=single_factual,
                            baseline_full_eos_lp=full_eos, unchanged_variant_full_eos_lp=single_reference,
                            baseline_joint_root=root, unchanged_variant_joint_root=single_factual-single_reference))
                        continue
                    saved = sample['saved_probe_credit'][probe_index]
                    source_position = sample['probe_source_positions'][probe_index]
                    assert saved['source_position'] == source_position and saved['d'] is not None
                    old_reader = prior_by_uid[sample['traj_uid']]
                    assert old_reader['source_step'] == 0 and old_reader['observed_return'] == 1
                    assert old_reader['probe_source_positions'] == sample['probe_source_positions']
                    assert old_reader['probe_token_ids'][probe_index] == saved['token_id']
                    original = previous_by_uid[sample['traj_uid']]
                    assert original['native_first_case_literal_ID_match']
                    probe_rows.append(dict(common, probe_index=probe_index,
                        source_position=source_position, input_position=saved['input_position'], token_id=saved['token_id'],
                        dt_d=float(saved['d']), saved_A=saved['A'], saved_Q=saved['Q'], saved_V=saved['V'],
                        saved_d_scope=saved['d_scope'],
                        matched_single_delete_d=single_factual-single_reference,
                        matched_single_factual_lp=single_factual, matched_single_reference_lp=single_reference,
                        matched_baseline_factual_lp=factual,
                        factual_lp_change_single_minus_full_eos_layout=single_factual-factual,
                        prior_reader_single_delete_d=old_reader['native_single_log_ratios'][probe_index],
                        prior_reader_single_factual_lp=old_reader['native_target_log_probs'][0],
                        prior_reader_single_reference_lp=old_reader['native_target_log_probs'][2+probe_index]))
    first_rows = [row for row in baseline_rows if row['source_step'] == 0]
    assert len(baseline_rows) == 56 and len(first_rows) == 26 and len(probe_rows) == 52
    probe_pool = reduce_replicas(probe_rows, ('traj_uid', 'source_step', 'source_position'),
        ('dt_d', 'matched_single_delete_d', 'matched_single_factual_lp', 'matched_single_reference_lp',
         'prior_reader_single_delete_d', 'prior_reader_single_factual_lp', 'prior_reader_single_reference_lp',
         'factual_lp_change_single_minus_full_eos_layout'))
    assert probe_pool['unique_identities'] == 42
    endpoint_pool = reduce_replicas(first_rows, ('traj_uid', 'source_step'),
        ('matched_factual_lp', 'matched_full_eos_lp', 'matched_joint_root',
         'old_factual_lp', 'old_full_eos_lp', 'old_joint_root', 'old_signed_sum'))
    assert endpoint_pool['unique_identities'] == 21
    for row in endpoint_pool['rows']:
        prior_case = prior_by_uid[row['traj_uid']]
        assert prior_case['source_step'] == 0 and prior_case['observed_return'] == 1
        row.update(prior_reader_factual_lp=prior_case['native_target_log_probs'][0],
                   prior_reader_full_eos_lp=prior_case['native_target_log_probs'][1],
                   prior_reader_joint_root=prior_case['native_full_span_log_ratio'],
                   previous_record_pooled_scores=previous_by_uid[row['traj_uid']]['pooled_duplicate_scores'])

    def endpoint_summary(rows):
        return {label: paired_statistics(rows, left, right) for label, left, right in
            [('factual_lp', 'matched_factual_lp', 'old_factual_lp'),
             ('full_eos_lp', 'matched_full_eos_lp', 'old_full_eos_lp'),
             ('joint_root', 'matched_joint_root', 'old_joint_root')]}

    def probe_summary(rows):
        return dict(saved_dt_vs_matched_single=paired_statistics(rows, 'dt_d', 'matched_single_delete_d'),
            saved_dt_vs_prior_cross_layout_single=paired_statistics(rows, 'dt_d', 'prior_reader_single_delete_d'),
            matched_vs_prior_cross_layout_single=paired_statistics(rows, 'matched_single_delete_d', 'prior_reader_single_delete_d'),
            single_factual_lp_matched_vs_prior=paired_statistics(rows, 'matched_single_factual_lp', 'prior_reader_single_factual_lp'),
            single_reference_lp_matched_vs_prior=paired_statistics(rows, 'matched_single_reference_lp', 'prior_reader_single_reference_lp'),
            absolute_dt_error_prior_minus_matched=describe([
                abs(row['dt_d']-row['prior_reader_single_delete_d'])-
                abs(row['dt_d']-row['matched_single_delete_d']) for row in rows]))

    used_paths = paths + [pack_path, old_endpoint_path] + prior_paths + [Path(__file__), AUDIT / 'analyze_textcraft_native_readout.py']
    completed_path = input_dir / 'completed.json'
    completed = json.loads(completed_path.read_bytes()) if completed_path.is_file() else None
    if completed is not None:
        used_paths.append(completed_path)
    return dict(scope=__doc__, status='descriptive_original_paired_layout_analysis_not_quality_repair',
        checkpoint=runs[0]['checkpoint'], sources=[source(path) for path in used_paths],
        actual_deployed_sources=[run['sources'] for run in runs],
        actual_original_owner_options=[run['numerical_runner_options'] for run in runs],
        dtype=[dict(rank=run['rank'], model=run['model_dtype'], native_fla_fp16=run['native_fla_fp16'],
                    attention=run['event_attention_backend'],
                    categorical_log_probability_dtypes=sorted({case['categorical_log_probability_dtype'] for case in run['cases']})) for run in runs],
        calls=[{key: run[key] for key in ('rank', 'paired_root_entries', 'native_forward_calls', 'native_root_calls',
                'native_prefix_calls', 'finite_seed_calls', 'finite_trace_calls', 'backward_calls', 'optimizer_steps', 'scheduler_steps')}
               for run in runs],
        coverage=dict(original_B4_groups_per_rank=[7, 7], original_layout_slots=56,
                      first_response_transport_slots=26, first_response_unique_UIDs=21,
                      probe_transport_slots=52, unique_probe_identities=42,
                      selection=pack['coverage']),
        original_layout_and_actual_native_boundaries=groups,
        baseline_full_eos_vs_original_endpoints=dict(
            scope='Same original transport slot and native target; full-response EOS root is not a single-token deletion effect.',
            all_original_slots=endpoint_summary(baseline_rows),
            first_response_transport_slots=endpoint_summary(first_rows),
            first_response_UID_balanced=endpoint_summary(endpoint_pool['rows']),
            original_slot_records=baseline_rows, first_response_replica_pool=endpoint_pool),
        same_layout_single_delete_vs_saved_DT=dict(
            scope='DT d is inferred from saved FP32 A. Direct d is the original native categorical score difference for one token changed to EOS, at the fixed full-EOS cut and original padded paired layout.',
            transport_weighted=probe_summary(probe_rows),
            unique_probe_balanced=probe_summary(probe_pool['rows']),
            original_probe_slot_records=probe_rows, unique_probe_replica_pool=probe_pool,
            factual_lp_change_within_matched_layout=describe([row['factual_lp_change_single_minus_full_eos_layout'] for row in probe_rows])),
        cross_layout_endpoint_comparison=dict(
            scope='Matched paired B8 with original EOS right extension/prefix versus previous native-reader B4 consisting of four variants of one prefix. Same literal case, different batch/cache layout.',
            unique_UIDs=21,
            factual_lp=paired_statistics(endpoint_pool['rows'], 'matched_factual_lp', 'prior_reader_factual_lp'),
            full_eos_lp=paired_statistics(endpoint_pool['rows'], 'matched_full_eos_lp', 'prior_reader_full_eos_lp'),
            joint_root=paired_statistics(endpoint_pool['rows'], 'matched_joint_root', 'prior_reader_joint_root')),
        unchanged_partner_observations=dict(
            scope='Non-first-response partner slots retain their original full-EOS reference in both single-probe calls; descriptive native repetition under the same collective schedule.',
            transport_variant_slots=len(unchanged_rows),
            factual_lp=paired_statistics(unchanged_rows, 'unchanged_variant_factual_lp', 'baseline_factual_lp'),
            full_eos_lp=paired_statistics(unchanged_rows, 'unchanged_variant_full_eos_lp', 'baseline_full_eos_lp'),
            joint_root=paired_statistics(unchanged_rows, 'unchanged_variant_joint_root', 'baseline_joint_root'),
            rows=unchanged_rows),
        completed_metadata=completed,
        limitations=[
            'Only the first seven original groups per rank and 21 successful first responses are selected; this is not the full training distribution or the 43 failing first responses.',
            'Two fixed-seed uniformly sampled positions per response, not positions selected by sign or magnitude; token probes from one response are dependent.',
            '26 transport slots and 52 probe slots contain replicas; the separate 21-UID/42-probe summaries use explicit replica means with every range retained.',
            'Native same-model deletion is not an environment/world oracle, calibration result, or evidence of the sole cause of historical training degradation.',
            'Stored DT d is diagnostically inferred from FP32 A and is not the original FP64 finite-entry value.',
            'Full-response root and individual deletion effects are distinct objects; neither sum conservation nor a root sign is a quality acceptance criterion.',
            'Descriptive error, signs, and mass have no numerical tolerance gate and are not official FA/FLA validation.',
            'The observer stops before finite seed/propagation; no new finite attribution, gradient, optimizer or scheduler update was performed.',
            'Absolute token-credit mass and opposite signs do not establish parameter-gradient cancellation or Adam update direction.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=LOCAL)
    parser.add_argument('--pack', type=Path, default=PACK)
    parser.add_argument('--prior-reader-dir', type=Path, default=PRIOR)
    parser.add_argument('--previous-endpoints', type=Path, default=OLD_ENDPOINTS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build_analysis(args.input_dir, args.pack, args.prior_reader_dir, args.previous_endpoints)
    path = args.output or args.input_dir / 'native-matched-layout-analysis.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=source(path), coverage=result['coverage'],
        baseline_first_UID=root_metrics(result),
        probe_transport=result['same_layout_single_delete_vs_saved_DT']['transport_weighted']['saved_dt_vs_matched_single'],
        probe_unique=result['same_layout_single_delete_vs_saved_DT']['unique_probe_balanced']['saved_dt_vs_matched_single']),
        ensure_ascii=False, indent=2))


def root_metrics(result):
    return {key: value['difference'] for key, value in
            result['baseline_full_eos_vs_original_endpoints']['first_response_UID_balanced'].items()}


if __name__ == '__main__':
    main()
