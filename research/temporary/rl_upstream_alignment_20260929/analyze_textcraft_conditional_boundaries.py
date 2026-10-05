"""Describe actual full-EOS coefficients on matched single-EOS native roots.

Only saved scalar owner observations are used.  Boundary C_b is the original
full-suffix contraction at decoder input b (0..31) or final-norm input32.
Differences, signs and layer increments are descriptive diagnostics, not a
tolerance gate, a causal percentage or a training-method change.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_textcraft_matched_layout import PACK, PACK_SHA, paired_statistics, reduce_replicas
from analyze_textcraft_native_readout import describe, source


AUDIT = Path(__file__).resolve().parent
LOCAL = AUDIT / 'textcraft-degradation-20261005/conditional-boundaries-20261006/v2'
MATCHED_MAP = AUDIT / 'textcraft-degradation-20261005/native-layout-20261006/v1/native-matched-layout-logits-map.json'


def _pair(case, slot):
    values = case['target_log_probs']
    assert len(values) == 8 and case['selection']['labels'][slot] == 16
    assert case['selection']['outcome_token_ids'] == [15, 16]
    reference, factual = values[2*slot], values[2*slot+1]
    return factual, reference, factual-reference


def _coordinate(row):
    return row['rank'], row['original_owner_batch_index'], row['slot'], row['probe_index']


def _boundary_analysis(rows):
    boundaries, decoders = [], []
    largest_residual_counts, largest_increment_counts = Counter(), Counter()
    for row in rows:
        native = row['native_d']
        residual = [row[f'C{boundary}']-native for boundary in range(33)]
        increments = [row[f'C{layer}']-row[f'C{layer+1}'] for layer in range(32)]
        largest_residual_counts[max(range(33), key=lambda boundary: abs(residual[boundary]))] += 1
        largest_increment_counts[max(range(32), key=lambda layer: abs(increments[layer]))] += 1
    for boundary in range(33):
        key = f'C{boundary}'
        stats = paired_statistics(rows, key, 'native_d')
        boundaries.append(dict(boundary_index=boundary,
            boundary_kind='final_norm_input' if boundary == 32 else 'decoder_input',
            contraction=stats['left_values'], residual_C_minus_native=stats['difference'],
            opposite_sign_count=stats['opposite_sign_count'], sign_counts=stats['sign_counts'],
            opposite_sign_native_absolute_mass=stats['opposite_sign_right_absolute_mass'],
            opposite_sign_contraction_absolute_mass=stats['opposite_sign_left_absolute_mass'],
            largest_absolute_residual_probe_count=largest_residual_counts[boundary]))
    for layer in range(32):
        increments = [row[f'C{layer}']-row[f'C{layer+1}'] for row in rows]
        changes = [abs(row[f'C{layer}']-row['native_d'])-abs(row[f'C{layer+1}']-row['native_d']) for row in rows]
        entered = sum(row[f'C{layer}']*row['native_d'] < 0 and
                      row[f'C{layer+1}']*row['native_d'] >= 0 for row in rows)
        exited = sum(row[f'C{layer}']*row['native_d'] >= 0 and
                     row[f'C{layer+1}']*row['native_d'] < 0 for row in rows)
        decoders.append(dict(decoder_index=layer,
            input_minus_output_increment=describe(increments),
            absolute_residual_change_input_minus_output=describe(changes),
            absolute_residual_increase_count=sum(value > 0 for value in changes),
            absolute_residual_decrease_count=sum(value < 0 for value in changes),
            entered_opposite_sign_count=entered, exited_opposite_sign_count=exited,
            largest_absolute_increment_probe_count=largest_increment_counts[layer]))
    return dict(n=len(rows), boundary_indices=boundaries, decoder_increments=decoders,
        ranking_scope='Ranks use the recorded values only. Max-absolute ties use the lowest index. Counts are dependent probes, not frequencies of training defects.',
        largest_mean_absolute_residual_boundaries=sorted(
            boundaries, key=lambda row: row['residual_C_minus_native']['abs_mean'], reverse=True)[:5],
        largest_mean_absolute_decoder_increments=sorted(
            decoders, key=lambda row: row['input_minus_output_increment']['abs_mean'], reverse=True)[:5],
        most_frequently_largest_absolute_decoder_increment=sorted(
            decoders, key=lambda row: row['largest_absolute_increment_probe_count'], reverse=True)[:5],
        C0_vs_actual_full_signed=paired_statistics(rows, 'C0', 'actual_signed_d'),
        actual_full_signed_vs_native=paired_statistics(rows, 'actual_signed_d', 'native_d'),
        actual_full_signed_vs_saved_DT=paired_statistics(rows, 'actual_signed_d', 'saved_dt_d'),
        native_vs_prior_matched_root=paired_statistics(rows, 'native_d', 'prior_matched_native_d'),
        telescoping_raw_difference=describe([
            (row['C0']-row['native_d'])-((row['C32']-row['native_d'])+
                sum(row[f'C{layer}']-row[f'C{layer+1}'] for layer in range(32))) for row in rows]))


def build_analysis(input_dir, pack_path, matched_map_path):
    paths = [input_dir / f'rank{rank}-readout.json' for rank in range(2)]
    runs = [json.loads(path.read_bytes()) for path in paths]
    pack = json.loads(pack_path.read_bytes())
    prior = json.loads(matched_map_path.read_bytes())
    assert source(pack_path)['sha256'] == PACK_SHA
    assert [run['rank'] for run in runs] == [0, 1]
    assert all(run['input_sha256'] == PACK_SHA and run['phase'] == 'complete_native_readout' for run in runs)
    assert [len(run['cases']) for run in runs] == [21, 21]
    assert all(run['optimizer_steps'] == run['scheduler_steps'] == run['backward_calls'] == 0 for run in runs)
    assert all(run['finite_trace_calls'] == run['finite_seed_calls'] == 7 for run in runs)
    assert runs[0]['checkpoint'] == runs[1]['checkpoint']
    prior_by_coordinate = {_coordinate(row): row for row in prior['observations']}
    assert len(prior_by_coordinate) == 52
    observations, flat_rows, group_metadata = [], [], []
    for rank, run in enumerate(runs):
        cases = {(case['original_owner_batch_index'], case['variant']): case for case in run['cases']}
        assert len(cases) == 21
        for group in pack['rank_groups'][rank]:
            index = group['original_owner_batch_index']
            full = cases[index, 'full_response_eos']
            assert full['samples'] == group['samples']
            assert full['finite_trace_calls'] == full['finite_seed_calls'] == 1
            actual_probes = {(row['original_slot'], probe['source_position']): probe
                for row in full['actual_signed_probe_values'] for probe in row['probes']}
            metadata = full['actual_coefficient_metadata']
            assert metadata['all_33_actual_boundaries_present'] and not metadata['diagnostics']
            group_metadata.append(dict(rank=rank, original_owner_batch_index=index,
                original_sequence_length=full['original_sequence_length'], observed_prefix_length=full['observed_prefix_length'],
                selection=full['selection'], actual_coefficient_metadata=metadata,
                full_finite_root_effect=full['original_finite_detail'].get('root_effect'),
                full_finite_signed_sum=full['original_finite_detail'].get('signed_sum')))
            for probe_index in range(2):
                variant = f'single_eos_{probe_index}'
                single = cases[index, variant]
                assert single['samples'] == group['samples'] and single['finite_trace_calls'] == single['finite_seed_calls'] == 0
                contractions = single['conditional_boundary_contractions']
                assert not contractions['diagnostics'] and contractions['score_calls'] == 1
                by_slot_boundary = {(row['original_slot'], row['boundary_index']): row
                    for row in contractions['boundaries']}
                assert len(by_slot_boundary) == len(contractions['boundaries'])
                for slot, sample in enumerate(group['samples']):
                    if sample['source_step'] != 0:
                        continue
                    assert sample['observed_return'] == 1
                    source_position = sample['probe_source_positions'][probe_index]
                    saved = sample['saved_probe_credit'][probe_index]
                    actual = actual_probes[slot, source_position]
                    assert actual['input_position'] == sample['probe_input_positions'][probe_index]
                    assert actual['token_id'] == saved['token_id'] and actual['saved_original_d'] == saved['d']
                    factual_lp, reference_lp, native = _pair(single, slot)
                    full_factual_lp, full_reference_lp, full_root = _pair(full, slot)
                    common = dict(rank=rank, original_owner_batch_index=index, slot=slot, probe_index=probe_index,
                        traj_uid=sample['traj_uid'], source_step=0, source_position=source_position,
                        input_position=actual['input_position'], token_id=actual['token_id'])
                    previous = prior_by_coordinate[_coordinate(common)]
                    assert all(common[field] == previous[field] for field in
                               ('traj_uid', 'source_step', 'source_position', 'input_position', 'token_id'))
                    values, boundary_records = {}, []
                    for boundary in range(33):
                        record = by_slot_boundary[slot, boundary]
                        assert record['prefix_cut'] == full['observed_prefix_length'] == single['observed_prefix_length']
                        assert record['suffix_length'] == full['selection']['length'] == single['selection']['length']
                        values[f'C{boundary}'] = record['full_suffix_contraction']
                        boundary_records.append(record)
                    flat = dict(common, **values, native_d=native, actual_signed_d=actual['actual_joint_d'],
                        saved_dt_d=saved['d'], prior_matched_native_d=previous['single_token_eos']['native_target_log_ratio'],
                        full_factual_lp=full_factual_lp, full_reference_lp=full_reference_lp,
                        single_factual_lp=factual_lp, single_reference_lp=reference_lp)
                    flat_rows.append(flat)
                    observations.append(dict(common,
                        observed_return=1, target_class_index=1, target_token_id=16,
                        original_request_index=sample['original_request_index'],
                        source_start=sample['source_start'], source_end=sample['source_end'],
                        context_tokens=sample['context_tokens'], compute_tokens=sample['compute_tokens'],
                        native_d=native, actual_full_signed_d=actual['actual_joint_d'], saved_DT_d=saved['d'],
                        saved_DT_d_scope=saved['d_scope'],
                        original_score_pairs=dict(single_factual_lp=factual_lp, single_reference_lp=reference_lp,
                            full_factual_lp=full_factual_lp, full_reference_lp=full_reference_lp,
                            full_span_native_d=full_root),
                        boundary_C=[values[f'C{boundary}'] for boundary in range(33)],
                        residual_C_minus_native=[values[f'C{boundary}']-native for boundary in range(33)],
                        per_decoder_input_minus_output=[values[f'C{layer}']-values[f'C{layer+1}'] for layer in range(32)],
                        C0_minus_actual_full_signed=values['C0']-actual['actual_joint_d'],
                        actual_boundary_metadata=boundary_records,
                        source_map=dict(rank_file=paths[rank].as_posix(), original_group=index, original_slot=slot,
                            full_variant='full_response_eos', single_variant=variant,
                            pack_sample_path=f'rank_groups[{rank}][{index}].samples[{slot}]')))
    assert len(flat_rows) == len(observations) == 52
    value_fields = tuple(f'C{boundary}' for boundary in range(33))+(
        'native_d', 'actual_signed_d', 'saved_dt_d', 'prior_matched_native_d',
        'full_factual_lp', 'full_reference_lp', 'single_factual_lp', 'single_reference_lp')
    pool = reduce_replicas(flat_rows, ('traj_uid', 'source_step', 'source_position'), value_fields)
    assert pool['unique_identities'] == 42
    old_means = {(row['traj_uid'], row['source_step'], row['source_position']): row for row in prior['unique_probe_means']}
    assert len(old_means) == 42
    for row in pool['rows']:
        old = old_means[row['traj_uid'], row['source_step'], row['source_position']]
        assert row['transport_slots'] == old['transport_slots']
        assert row['saved_dt_d'] == old['arithmetic_replica_means']['dt_d']
        assert row['prior_matched_native_d'] == old['arithmetic_replica_means']['matched_single_delete_d']
    # Keep every scalar transport record, and preserve dispersion for actual
    # replicas rather than silently substituting one transport row.
    unique_rows = []
    for row in pool['rows']:
        entry = {field: row[field] for field in ('traj_uid', 'source_step', 'source_position',
                                                 'transport_slots', 'transport_locations')}
        entry['explicit_arithmetic_replica_means'] = {field: row[field] for field in value_fields}
        if row['transport_slots'] > 1:
            entry['replica_min_max_range'] = row['replica_values']
        unique_rows.append(entry)
    source_paths = paths+[pack_path, matched_map_path, Path(__file__), AUDIT/'analyze_textcraft_matched_layout.py',
        AUDIT/'analyze_textcraft_native_readout.py']
    completed_path = input_dir/'completed.json'
    completed = json.loads(completed_path.read_bytes()) if completed_path.is_file() else None
    if completed is not None:
        source_paths.append(completed_path)
    return dict(scope=__doc__, status='descriptive_actual_conditional_boundary_observation_not_method_change',
        checkpoint=runs[0]['checkpoint'], sources=[source(path) for path in source_paths],
        actual_imported_sources=[run['sources'] for run in runs],
        actual_recipe_source=[run.get('reused_original_recipe') for run in runs],
        actual_collector_source=[run.get('passive_collector') for run in runs],
        actual_owner_options=[run['numerical_runner_options'] for run in runs],
        dtype=[dict(model=run['model_dtype'], native_fla_fp16=run['native_fla_fp16'],
                    event_attention=run['event_attention_backend']) for run in runs],
        calls=[{key:run[key] for key in ('rank', 'finite_trace_calls', 'finite_seed_calls',
            'actual_conditional_boundary_single_roots', 'backward_calls', 'optimizer_steps', 'scheduler_steps')} for run in runs],
        coverage=dict(original_groups_per_rank=[7,7], transport_probes=52, unique_probe_identities=42,
                      unique_UIDs=len({row['traj_uid'] for row in flat_rows}),
                      duplicated_probe_identities=pool['duplicated_identities'],
                      same_original_mapping_and_replica_counts_as_prior_logits_map=True),
        coefficient_groups=group_metadata,
        transport_weighted_boundary_analysis=_boundary_analysis(flat_rows),
        unique_probe_balanced_boundary_analysis=_boundary_analysis(pool['rows']),
        transport_observations=observations,
        explicit_unique_probe_means_scope=pool['scope'], explicit_unique_probe_means=unique_rows,
        maximum_replica_ranges=pool['max_replica_ranges'], completed_metadata=completed,
        limitations=[
            'C_b uses actual full-response-EOS finite coefficients on an actual single-EOS native activation difference. It is not a new per-token finite trace or world oracle.',
            'C_b contraction runs on CPU using the original owner function and native saved BF16 inputs; this is not official FA/FLA GPU tolerance acceptance.',
            'Layer increments telescope between the recorded boundaries, but are not causal percentages and do not separate independent causes.',
            'Seven original groups per rank, 21 successful first-response UID and two fixed random positions each; probes and replicas are dependent.',
            'Per-probe actual full signed values come from this full trace. Saved DT d comes from earlier FP32 A inversion and is reported separately.',
            'Cross-run native score differences from prior matched-layout v1 are descriptive; the current native score is the reference for current residuals.',
            'Counts of opposite signs or largest residuals do not establish task-gradient harm, a bug or the cause of historical degradation.',
            'No threshold, correction, normalization, new reference, sampling, optimization or training change is introduced.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=LOCAL)
    parser.add_argument('--pack',type=Path,default=PACK)
    parser.add_argument('--matched-map',type=Path,default=MATCHED_MAP)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=build_analysis(args.input_dir,args.pack,args.matched_map)
    path=args.output or args.input_dir/'conditional-boundary-analysis.json'
    path.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    unique=result['unique_probe_balanced_boundary_analysis']
    print(json.dumps(dict(output=source(path),coverage=result['coverage'],
        C0_vs_actual_full_signed=unique['C0_vs_actual_full_signed'],
        largest_mean_absolute_residual_boundaries=unique['largest_mean_absolute_residual_boundaries'],
        largest_mean_absolute_decoder_increments=unique['largest_mean_absolute_decoder_increments']),indent=2))


if __name__=='__main__':
    main()
