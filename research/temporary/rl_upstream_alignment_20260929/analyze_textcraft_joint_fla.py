"""Describe saved same-full-pair FLA projections without a numerical gate.

Reuse the original FLA parser and source map. Sum the actual head groups for
each original full-response EOS/factual sample, then explicitly reduce its
transport replicas. The population is 26 full samples / 21 successful UID,
not the 52 single-delete transports / 42 probe identities in the old parser.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from analyze_textcraft_conditional_fla import (
    build_fla_analysis, LAYERS, PACK, MATCHED_MAP, INPUTS, reduce_replicas, describe, source,
)
from analyze_textcraft_matched_layout import paired_statistics


LOCAL = Path(__file__).parent / 'textcraft-degradation-20261005/joint-fla-20261006/v1'
VALUES = ('F_joint', 'Y_public', 'Y_norm', 'F_minus_Y_public', 'public_minus_norm',
    'F_minus_Y_norm', 'F_minus_input_field_sum',
    'recorded_F_minus_Y_public_minus_recomputed',
    'recorded_public_minus_norm_minus_recomputed',
    'recorded_F_minus_Y_norm_minus_recomputed', 'residual_bridge_closure') + tuple(
    'input.'+name for name in INPUTS)
IDENTITY = ('traj_uid', 'source_step')
LAYOUT = ('rank', 'original_owner_batch_index', 'slot', 'prefix_cut',
    'native_suffix_length', 'capture_start', 'coefficient_suffix_length')


def _statistics(rows):
    return {name: describe([row[name] for row in rows]) for name in VALUES}


def _direction(rows):
    return {name: paired_statistics(rows, left, right) for name, left, right in (
        ('F_joint_vs_Y_public', 'F_joint', 'Y_public'),
        ('Y_public_vs_Y_norm', 'Y_public', 'Y_norm'),
        ('F_joint_vs_Y_norm', 'F_joint', 'Y_norm'))}


def build_joint_analysis(input_dir, pack_path=PACK, matched_map_path=MATCHED_MAP):
    input_dir = Path(input_dir)
    # This existing parser owns the original raw/source alignment. Its single
    # probe population is preserved only as provenance, not the new denominator.
    base = build_fla_analysis(input_dir, pack_path, matched_map_path)
    pack = json.loads(Path(pack_path).read_bytes())
    runs = [json.loads((input_dir/f'rank{rank}-readout.json').read_bytes()) for rank in range(2)]
    groups = {(rank, group['original_owner_batch_index']): group
        for rank, rank_groups in enumerate(pack['rank_groups']) for group in rank_groups}
    full_cases = {(rank, case['original_owner_batch_index']): case
        for rank, run in enumerate(runs) for case in run['cases'] if case['variant'] == 'full_response_eos'}
    head_rows, rows, full_metadata, resources = [], [], [], []
    dtype_sets = defaultdict(set)
    counter_totals = Counter()

    def dtype(name, metadata):
        if metadata is not None:
            dtype_sets[name].add(metadata['dtype'])

    for entry in base['full_finite_FLA_metadata']:
        rank, batch = entry['rank'], entry['original_owner_batch_index']
        original = entry['metadata']
        joint = original['joint_fla']
        group, full = groups[rank, batch], full_cases[rank, batch]
        assert full['samples'] == group['samples']
        expected_slots = [slot for slot, sample in enumerate(group['samples']) if sample['source_step'] == 0]
        assert joint['original_slots'] == expected_slots == original['original_slots']
        assert not joint['diagnostics']
        assert joint['recorded_layer_slot_scalars'] == joint['expected_layer_slot_scalars']
        assert joint['actual_public_output_returns'] == joint['expected_public_output_returns']
        assert joint['actual_norm_input_captures'] == joint['expected_norm_input_captures']
        assert not joint['unconsumed_output_layer_indices']
        assert joint['retained_cpu_output_bytes_after_cleanup'] == 0
        assert joint['added_native_model_calls'] == joint['added_native_FLA_calls'] == 0
        full_metadata.append(dict(rank=rank, original_owner_batch_index=batch, metadata=joint))
        resources.append(dict(rank=rank, original_owner_batch_index=batch,
            peak_retained_cpu_output_bytes=joint['peak_retained_cpu_output_bytes'],
            retained_cpu_output_bytes_before_cleanup=joint['retained_cpu_output_bytes_before_cleanup'],
            retained_cpu_output_bytes_after_cleanup=joint['retained_cpu_output_bytes_after_cleanup'],
            unconsumed_output_layer_indices=joint['unconsumed_output_layer_indices']))
        for name in ('actual_public_output_returns', 'actual_norm_input_captures',
                     'actual_original_finite_head_calls', 'recorded_layer_slot_scalars',
                     'expected_layer_slot_scalars', 'added_native_model_calls', 'added_native_FLA_calls'):
            counter_totals[name] += joint[name]
        for capture in joint['capture_metadata'].values():
            dtype('actual_public_initial_state', capture['actual_public_initial_state'])
            for name, tensor in capture['fields'].items():
                dtype('actual_'+name, tensor)

        by_layer_slot = defaultdict(list)
        for item in joint['layers']:
            layer, slot = item['decoder_index'], item['original_slot']
            assert layer in LAYERS and slot in expected_slots
            sample = group['samples'][slot]
            assert sample['observed_return'] == 1
            old_groups = original['layers'][str(layer)]['groups']
            matching = [old for old in old_groups if old['layout'] == item['layout']]
            assert len(matching) == 1
            old = matching[0]
            assert item['coordinates'] == old['coordinates']
            assert item['native_time_start'] == old['native_time_start']
            assert item['head_count'] == old['head_count'] and item['head_width'] == old['head_width']
            assert item['actual_native_do'] == old['incoming_do']
            assert item['actual_FLA_incoming_h'] == old['incoming_state']
            assert item['incoming_first_chunk_h_pair_absmax'] == old['incoming_state_pair_absmax'][str(slot)]
            fields = item['original_joint_input_fields']
            assert fields == old['actual_joint_input_projections'][str(slot)]
            assert set(fields) == set(INPUTS)
            F, public, norm = (float(item[name]) for name in (
                'F_joint', 'Y_joint_public_output', 'Y_joint_norm_boundary'))
            values = dict(F_joint=F, Y_public=public, Y_norm=norm,
                F_minus_Y_public=F-public, public_minus_norm=public-norm, F_minus_Y_norm=F-norm,
                F_minus_input_field_sum=F-math.fsum(fields.values()),
                recorded_F_minus_Y_public_minus_recomputed=item['F_joint_minus_Y_public']-(F-public),
                recorded_public_minus_norm_minus_recomputed=item['public_minus_norm_boundary']-(public-norm),
                recorded_F_minus_Y_norm_minus_recomputed=item['F_joint_minus_Y_norm_boundary']-(F-norm),
                residual_bridge_closure=(F-public)+(public-norm)-(F-norm))
            values.update({'input.'+name: float(fields[name]) for name in INPUTS})
            row = dict(rank=rank, original_owner_batch_index=batch, slot=slot, decoder_index=layer,
                traj_uid=sample['traj_uid'], source_step=sample['source_step'],
                prefix_cut=item['coordinates']['prefix_cut'], native_suffix_length=item['coordinates']['suffix_length'],
                capture_start=item['layout']['capture_start'],
                coefficient_suffix_length=item['coordinates']['suffix_length']-item['layout']['capture_start'],
                **values, original_head_record=item)
            head_rows.append(row)
            by_layer_slot[layer, slot].append(row)
            dtype('native_do', item['actual_native_do'])
            dtype('CPU_do', item['CPU_do'])
            dtype('actual_stage_incoming_h', item['actual_FLA_incoming_h'])

        for layer in LAYERS:
            expected_groups = original['layers'][str(layer)]['groups']
            for slot in expected_slots:
                heads = sorted(by_layer_slot[layer, slot], key=lambda row:row['original_head_record']['layout']['head_start'])
                assert len(heads) == len(expected_groups)
                assert [row['original_head_record']['layout'] for row in heads] == [old['layout'] for old in expected_groups]
                values = {name: math.fsum(row[name] for row in heads) for name in VALUES}
                head_difference_sums = {name:values[name] for name in (
                    'F_minus_Y_public','public_minus_norm','F_minus_Y_norm','residual_bridge_closure')}
                # Sum original head scalars without fitting/correcting them.
                # Preserve both head-level and post-sum arithmetic differences.
                values['F_minus_Y_public'] = values['F_joint']-values['Y_public']
                values['public_minus_norm'] = values['Y_public']-values['Y_norm']
                values['F_minus_Y_norm'] = values['F_joint']-values['Y_norm']
                values['residual_bridge_closure'] = values['F_minus_Y_public']+values['public_minus_norm']-values['F_minus_Y_norm']
                values['F_minus_input_field_sum'] = values['F_joint']-math.fsum(values['input.'+name] for name in INPUTS)
                sample = group['samples'][slot]
                row = {name: heads[0][name] for name in LAYOUT+IDENTITY+('decoder_index',)}
                row.update(values)
                row['actual_head_group_count'] = len(heads)
                row['sum_of_original_head_differences'] = head_difference_sums
                row['original_head_records'] = [head['original_head_record'] for head in heads]
                row['source_sample'] = {name:sample[name] for name in (
                    'original_request_index', 'source_start', 'source_end', 'context_tokens',
                    'compute_tokens', 'query_tokens', 'observed_return')}
                row['source_map'] = dict(pack_sample_path=f'rank_groups[{rank}][{batch}].samples[{slot}]',
                    original_variant='full_response_eos',
                    selected_input_ids_sha256=group['selected_input_ids_sha256'],
                    reference_input_ids_sha256=group['reference_input_ids_sha256'],
                    actual_prefix_cut=full['observed_prefix_length'], selection=full['selection'])
                row['post_sum_residual_bridge_closure'] = (values['F_joint']-values['Y_public'])+(values['Y_public']-values['Y_norm'])-(values['F_joint']-values['Y_norm'])
                rows.append(row)

    assert len(full_metadata) == 14 and len(rows) == 26*len(LAYERS)
    summaries, unique, cuts = [], [], []
    for layer in LAYERS:
        selected = [row for row in rows if row['decoder_index'] == layer]
        pool = reduce_replicas(selected, IDENTITY, VALUES)
        assert pool['transport_slots'] == 26 and pool['unique_identities'] == 21
        for row in pool['rows']:
            members = [member for member in selected if all(member[name] == row[name] for name in IDENTITY)]
            row['actual_replica_layouts'] = [{name:member[name] for name in LAYOUT} |
                dict(actual_head_group_count=member['actual_head_group_count'], source_sample=member['source_sample'],
                     original_head_records=member['original_head_records']) for member in members]
        unique.append(dict(decoder_index=layer, **pool))
        summaries.append(dict(decoder_index=layer, transport_n=len(selected), unique_n=pool['unique_identities'],
            transport_statistics=_statistics(selected), unique_UID_statistics=_statistics(pool['rows']),
            transport_direction=_direction(selected), unique_UID_direction=_direction(pool['rows']),
            maximum_replica_ranges=pool['max_replica_ranges'],
            head_counts=dict(Counter(str(row['actual_head_group_count']) for row in selected))))
        for start in sorted({row['capture_start'] for row in selected}):
            part = [row for row in selected if row['capture_start'] == start]
            cuts.append(dict(decoder_index=layer, capture_start=start, transport_n=len(part),
                original_group_coordinates=sorted({(row['rank'],row['original_owner_batch_index']) for row in part}),
                transport_statistics=_statistics(part)))

    return dict(scope=__doc__, status='descriptive_same_full_pair_not_official_tolerance',
        sources=base['sources']+[source(Path(__file__))]+[source(Path(__file__).with_name(name)) for name in (
            'observe_textcraft_joint_fla.py', 'verify_textcraft_joint_fla.py', 'stage_textcraft_joint_fla.py')],
        checkpoint=base['checkpoint'], actual_imported_sources=base['actual_imported_sources'],
        actual_owner_options=base['actual_owner_options'], original_call_counters=base['calls'],
        model_runtime_dtypes=base['dtype'],
        actual_native_call_counters=[{name:run[name] for name in (
            'rank','native_forward_calls','native_root_calls','native_prefix_calls','paired_root_entries')} for run in runs],
        coverage=dict(original_groups_per_rank=[7,7], full_transport_samples_per_layer=26,
            unique_full_response_UIDs_per_layer=21, layer_indices=list(LAYERS),
            full_sample_identity_fields=list(IDENTITY), single_probe_population_not_used_as_denominator=True),
        reused_original_FLA_parser_coverage=base['coverage'],
        observed_original_FLA_tensor_dtypes=base['observed_tensor_dtypes'],
        observed_same_pair_tensor_dtypes={name:sorted(values) for name,values in dtype_sets.items()},
        same_pair_head_observation_counts=dict(counter_totals) | dict(
            scalar_head_rows=len(head_rows), aggregated_layer_slot_rows=len(rows),
            completeness_scope='Counts and exact source/layout joins only; closure values have no new numerical acceptance threshold.'),
        same_pair_head_closure_statistics={name:describe([row[name] for row in head_rows]) for name in (
            'F_minus_input_field_sum', 'recorded_F_minus_Y_public_minus_recomputed',
            'recorded_public_minus_norm_minus_recomputed', 'recorded_F_minus_Y_norm_minus_recomputed', 'residual_bridge_closure')},
        post_head_sum_closure=describe([row['post_sum_residual_bridge_closure'] for row in rows]),
        layer_summaries=summaries, capture_start_groups=cuts,
        full_joint_metadata=full_metadata, full_original_FLA_metadata=base['full_finite_FLA_metadata'],
        CPU_output_capture_resources=resources,
        CPU_peak_output_payload_statistics=describe([row['peak_retained_cpu_output_bytes'] for row in resources]),
        original_average_coefficient_payloads=base['CPU_coefficient_bank_payloads'],
        original_average_coefficient_payload_statistics=base['CPU_coefficient_bank_payload_statistics'],
        head_transport_rows=head_rows, full_sample_transport_rows=rows, unique_full_UID_replica_means=unique,
        limitations=[
            'F_joint and Y_public use the same full-response EOS/factual pair; unlike the prior single-delete projection, no endpoint pair is changed between these two quantities.',
            'The native public initial_state metadata is distinct from stage h metadata; neither is reconstructed or replaced.',
            'Public-to-norm difference holds the actual native do fixed. It does not include the earlier FP32 mo to BF16 to FP16 seed conversion.',
            'Summation uses math.fsum of saved Python float head scalars. Raw head rows and rounding-sized closure differences remain visible; no value is corrected.',
            'UID statistics use arithmetic replica means with all original ranges/layouts retained. The 21 UID are successful first responses, not the whole training population or independent head groups.',
            'Same-pair closure is descriptive; this report introduces no official tolerance, gate, formula change, normalization, correction or kernel-error claim.',
            'CPU output payload is retained observation storage, not process RSS or peak GPU memory. Zero added native calls does not mean zero copy/contraction overhead.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=LOCAL)
    parser.add_argument('--pack', type=Path, default=PACK)
    parser.add_argument('--matched-map', type=Path, default=MATCHED_MAP)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build_joint_analysis(args.input_dir, args.pack, args.matched_map)
    output = args.output or args.input_dir/'joint-fla-analysis.json'
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',',':'), allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=source(output), coverage=result['coverage'],
        layers=[dict(decoder_index=layer['decoder_index'], transport_n=layer['transport_n'], unique_n=layer['unique_n'],
            differences={name:{key:layer['unique_UID_statistics'][name][key] for key in ('abs_mean','rms','positive','negative','zero')}
                for name in ('F_minus_Y_public','public_minus_norm','F_minus_Y_norm')},
            opposite_signs={name:value['opposite_sign_count'] for name,value in layer['unique_UID_direction'].items()},
            maximum_replica_ranges=layer['maximum_replica_ranges']) for layer in result['layer_summaries']],
        counters=result['same_pair_head_observation_counts'],
        actual_dtypes=result['observed_same_pair_tensor_dtypes'],
        payload=result['CPU_peak_output_payload_statistics']), ensure_ascii=False))


if __name__ == '__main__':
    main()
