"""Map saved original paired-root logits to literal source/probe identities.

Standard-library data preparation only.  The original native categorical
logits and target log probabilities are read verbatim.  No head calculation,
model call, new credit calculation or training change is made.
"""
import argparse
import json
from pathlib import Path

from analyze_textcraft_matched_layout import (LOCAL, PACK, PACK_SHA,
    paired_statistics, reduce_replicas)
from analyze_textcraft_native_readout import source


NUMERIC_FIELDS = (
    'full_eos_reference_z0', 'full_eos_reference_z1',
    'full_eos_factual_z0', 'full_eos_factual_z1',
    'single_eos_reference_z0', 'single_eos_reference_z1',
    'single_eos_factual_z0', 'single_eos_factual_z1',
    'full_eos_reference_lp', 'full_eos_factual_lp',
    'single_eos_reference_lp', 'single_eos_factual_lp',
    'full_eos_native_d', 'matched_single_delete_d', 'dt_d')


def endpoint_pair(case, slot):
    """Select the saved ref/fact pair; do not recompute categorical scores."""
    reference_row, factual_row = 2 * slot, 2 * slot + 1
    assert len(case['categorical_logits_values']) == len(case['target_log_probs']) == 8
    assert case['selection']['outcome_token_ids'] == [15, 16]
    assert case['selection']['labels'][slot] == 16
    reference_z = case['categorical_logits_values'][reference_row]
    factual_z = case['categorical_logits_values'][factual_row]
    assert len(reference_z) == len(factual_z) == 2
    reference_lp = case['target_log_probs'][reference_row]
    factual_lp = case['target_log_probs'][factual_row]
    return dict(variant=case['variant'],
        original_paired_row_indices=dict(reference=reference_row, factual=factual_row),
        reference=dict(z0=reference_z[0], z1=reference_z[1], target_logp=reference_lp),
        factual=dict(z0=factual_z[0], z1=factual_z[1], target_logp=factual_lp),
        native_target_log_ratio=factual_lp-reference_lp,
        original_sequence_length=case['original_sequence_length'],
        observed_prefix_length=case['observed_prefix_length'], fixed_cut=case['fixed_cut'],
        native_target_predictor_position_in_suffix=case['selection']['positions'][slot],
        categorical_logits_dtype=case['categorical_logits']['dtype'],
        categorical_log_probability_dtype=case['categorical_log_probability_dtype'])


def flatten_values(full, single, saved_d):
    return dict(
        full_eos_reference_z0=full['reference']['z0'], full_eos_reference_z1=full['reference']['z1'],
        full_eos_factual_z0=full['factual']['z0'], full_eos_factual_z1=full['factual']['z1'],
        single_eos_reference_z0=single['reference']['z0'], single_eos_reference_z1=single['reference']['z1'],
        single_eos_factual_z0=single['factual']['z0'], single_eos_factual_z1=single['factual']['z1'],
        full_eos_reference_lp=full['reference']['target_logp'], full_eos_factual_lp=full['factual']['target_logp'],
        single_eos_reference_lp=single['reference']['target_logp'], single_eos_factual_lp=single['factual']['target_logp'],
        full_eos_native_d=full['native_target_log_ratio'],
        matched_single_delete_d=single['native_target_log_ratio'], dt_d=saved_d)


def prepare(input_dir, pack_path, analysis_path):
    rank_paths = [input_dir / f'rank{rank}-readout.json' for rank in range(2)]
    runs = [json.loads(path.read_bytes()) for path in rank_paths]
    pack = json.loads(pack_path.read_bytes())
    existing = json.loads(analysis_path.read_bytes())
    assert source(pack_path)['sha256'] == PACK_SHA
    assert all(run['input_sha256'] == PACK_SHA and run['phase'] == 'complete_native_readout' for run in runs)
    assert [run['rank'] for run in runs] == [0, 1]
    assert [len(run['cases']) for run in runs] == [21, 21]
    assert all(run['finite_seed_calls'] == run['finite_trace_calls'] == run['backward_calls'] ==
               run['optimizer_steps'] == run['scheduler_steps'] == 0 for run in runs)
    existing_rows = existing['same_layout_single_delete_vs_saved_DT']['original_probe_slot_records']
    key = lambda row: (row['rank'], row['original_owner_batch_index'], row['slot'], row['probe_index'])
    existing_by_slot = {key(row): row for row in existing_rows}
    assert len(existing_by_slot) == len(existing_rows) == 52
    observations, flat_rows = [], []
    for rank, run in enumerate(runs):
        raw_cases = {(case['original_owner_batch_index'], case['variant']): (case_index, case)
                     for case_index, case in enumerate(run['cases'])}
        assert len(raw_cases) == 21
        for group in pack['rank_groups'][rank]:
            group_index = group['original_owner_batch_index']
            full_case_index, full_case = raw_cases[group_index, 'full_response_eos']
            assert full_case['samples'] == group['samples']
            for slot, sample in enumerate(group['samples']):
                if sample['source_step'] != 0:
                    continue
                assert sample['observed_return'] == 1 and sample['native_target_case']['target_ids'] == [16]
                full = endpoint_pair(full_case, slot)
                for probe_index in range(2):
                    variant = f'single_eos_{probe_index}'
                    single_case_index, single_case = raw_cases[group_index, variant]
                    assert single_case['samples'] == group['samples']
                    single = endpoint_pair(single_case, slot)
                    saved = sample['saved_probe_credit'][probe_index]
                    source_position = sample['probe_source_positions'][probe_index]
                    input_position = sample['probe_input_positions'][probe_index]
                    assert saved['source_position'] == source_position
                    assert saved['input_position'] == input_position == sample['source_start']+source_position
                    assert group['selected_input_ids'][slot][input_position] == saved['token_id']
                    common = dict(rank=rank, original_owner_batch_index=group_index, slot=slot,
                        probe_index=probe_index, traj_uid=sample['traj_uid'], source_step=0,
                        source_position=source_position, input_position=input_position, token_id=saved['token_id'])
                    values = flatten_values(full, single, float(saved['d']))
                    original = existing_by_slot[key(common)]
                    assert original['traj_uid'] == sample['traj_uid']
                    assert original['source_position'] == source_position and original['token_id'] == saved['token_id']
                    assert values['matched_single_delete_d'] == original['matched_single_delete_d']
                    assert values['dt_d'] == original['dt_d']
                    flat_rows.append(dict(common, **values))
                    observations.append(dict(common,
                        original_request_index=sample['original_request_index'],
                        source_start=sample['source_start'], source_end=sample['source_end'],
                        context_tokens=sample['context_tokens'], compute_tokens=sample['compute_tokens'],
                        observed_return=sample['observed_return'], outcome_token_ids=[15, 16],
                        target_class_index=1, target_token_id=16,
                        saved_DT_d=values['dt_d'], saved_DT_d_scope=saved['d_scope'],
                        full_response_eos=full, single_token_eos=single,
                        raw_source_map=dict(rank_file=rank_paths[rank].as_posix(),
                            full_eos_case_index=full_case_index, single_eos_case_index=single_case_index,
                            categorical_logits_path='cases[case_index].categorical_logits_values[paired_row_index][class_index]',
                            target_log_probability_path='cases[case_index].target_log_probs[paired_row_index]',
                            pack_sample_path=f'rank_groups[{rank}][{group_index}].samples[{slot}]',
                            pack_probe_path=f'saved_probe_credit[{probe_index}]',
                            selected_input_ids_sha256=group['selected_input_ids_sha256'],
                            full_eos_reference_input_ids_sha256=group['reference_input_ids_sha256'])))
    assert len(observations) == len(flat_rows) == 52
    pool = reduce_replicas(flat_rows, ('traj_uid', 'source_step', 'source_position'), NUMERIC_FIELDS)
    assert pool['unique_identities'] == 42
    old_pool = existing['same_layout_single_delete_vs_saved_DT']['unique_probe_replica_pool']['rows']
    unique_key = lambda row: (row['traj_uid'], row['source_step'], row['source_position'])
    old_unique = {unique_key(row): row for row in old_pool}
    assert len(old_unique) == 42
    for row in pool['rows']:
        original = old_unique[unique_key(row)]
        assert row['transport_slots'] == original['transport_slots']
        assert row['matched_single_delete_d'] == original['matched_single_delete_d']
        assert row['dt_d'] == original['dt_d']
    transport_summary = paired_statistics(flat_rows, 'dt_d', 'matched_single_delete_d')
    unique_summary = paired_statistics(pool['rows'], 'dt_d', 'matched_single_delete_d')
    assert transport_summary == existing['same_layout_single_delete_vs_saved_DT']['transport_weighted']['saved_dt_vs_matched_single']
    assert unique_summary == existing['same_layout_single_delete_vs_saved_DT']['unique_probe_balanced']['saved_dt_vs_matched_single']
    # Keep all replicas above.  Emit concise explicit means, with min/max/range
    # only on the ten probe identities that have multiple transport slots.
    means = []
    for row in pool['rows']:
        item = {field: row[field] for field in ('traj_uid', 'source_step', 'source_position',
                'transport_slots', 'transport_locations')}
        item['arithmetic_replica_means'] = {field: row[field] for field in NUMERIC_FIELDS}
        if row['transport_slots'] > 1:
            item['replica_min_max_range'] = row['replica_values']
        means.append(item)
    return dict(scope=__doc__, sources=[source(path) for path in
        rank_paths+[pack_path, analysis_path, Path(__file__), Path(__file__).with_name('analyze_textcraft_matched_layout.py')]],
        checkpoint=runs[0]['checkpoint'], actual_owner_sources=[run['sources'] for run in runs],
        mapping_scope=dict(endpoint_order='reference,factual interleaved per original slot',
            z0='Saved native FP32 categorical logit for outcome token15/class0',
            z1='Saved native FP32 categorical logit for outcome token16/class1',
            target_class_index=1, target_token_id=16,
            full_eos='Entire original current response is EOS in the original reference row.',
            single_eos='Only the fixed random source position is EOS; original paired layout and full-EOS observed prefix cut are retained.'),
        coverage=dict(transport_observations=52, unique_probe_identities=42,
                      unique_UIDs=len({row['traj_uid'] for row in observations}),
                      duplicated_probe_identities=pool['duplicated_identities']),
        consistency_with_existing_native_d=dict(
            scope='Exact equality of the same saved scores/mapping/statistics; not a numerical quality or official tolerance test.',
            all_52_transport_native_d_and_DT_d_exact=True,
            all_42_explicit_replica_native_d_and_DT_d_means_exact=True,
            original_transport_paired_statistics_exact=True,
            original_unique_probe_paired_statistics_exact=True,
            transport_native_d=transport_summary['right_values'],
            unique_probe_native_d=unique_summary['right_values']),
        observations=observations,
        unique_probe_means_scope=pool['scope'], unique_probe_means=means,
        maximum_replica_ranges=pool['max_replica_ranges'],
        limitations=[
            'z0/z1 denote the two categorical classes, not the reference/factual endpoint identities; those endpoints are named explicitly.',
            'Means of logits and means of log probabilities are separately retained saved-data summaries; no score is recomputed from the averaged logits.',
            'Saved DT d comes from the prior FP32 A inversion, not a new DT call or the original FP64 finite entry.',
            'Joint full-response deletion and single-token deletion remain different objects.',
            'All results are descriptive same-model data mappings; no model/world accuracy or training-degradation cause is asserted.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=LOCAL)
    parser.add_argument('--pack', type=Path, default=PACK)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    analysis_path = args.analysis or args.input_dir / 'native-matched-layout-analysis.json'
    result = prepare(args.input_dir, args.pack, analysis_path)
    output = args.output or args.input_dir / 'native-matched-layout-logits-map.json'
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=source(output), coverage=result['coverage'],
                         consistency=result['consistency_with_existing_native_d']), indent=2))


if __name__ == '__main__':
    main()
