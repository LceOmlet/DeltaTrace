"""Describe original FLA conditional projections without a numerical gate.

Reuse the existing GDN/boundary parser, transport identities, descriptive
statistics and explicit replica reduction. The three recorded differences
split the same-run remaining_input_FLA projection; closure is arithmetic,
not an official kernel acceptance test or a repaired learning signal.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_textcraft_conditional_gdn import (
    build_gdn_analysis, LAYERS, PACK, MATCHED_MAP, reduce_replicas, describe, source,
)


LOCAL = Path(__file__).parent / 'textcraft-degradation-20261005/conditional-fla-20261006/v1'
TERMS = ('input_chain', 'FLA_projection', 'native_cast_bridge')
INPUTS = ('q', 'k', 'v', 'beta', 'g')
FIELDS = TERMS + ('F', 'Y', 'I', 'Z', 'O', 'remaining_input_FLA', 'three_term_sum',
    'sum_minus_remaining', 'stored_closure', 'recorded_sum_minus_recomputed',
    'recorded_remaining_minus_original_GDN', 'F_minus_input_field_sum',
    'Y_minus_output_field', 'recorded_FLA_difference_minus_recomputed', 'native_d') + (
    tuple('input_projection.'+name for name in INPUTS)) + (
    tuple('term_identity.'+name for name in TERMS))


def _statistics(rows):
    # Keep the original describe implementation; no new metric or tolerance.
    return {name: describe([row[name] for row in rows]) for name in FIELDS}


def _row_key(row):
    return tuple(row[name] for name in (
        'rank', 'original_owner_batch_index', 'slot', 'probe_index', 'decoder_index'))


def build_fla_analysis(input_dir, pack_path=PACK, matched_map_path=MATCHED_MAP):
    input_dir = Path(input_dir)
    gdn = build_gdn_analysis(input_dir, pack_path, matched_map_path)
    originals = {_row_key(row): row for row in gdn['transport_rows']}
    paths = [input_dir/f'rank{rank}-readout.json' for rank in range(2)]
    runs = [json.loads(path.read_bytes()) for path in paths]
    full_metadata, reports, rows = [], [], []
    dtype_sets = dict(coefficients={}, incoming_do={}, joint_state={}, native_fields={}, native_initial_state={})

    def add_dtype(category, name, metadata):
        if metadata is not None:
            dtype_sets[category].setdefault(name, set()).add(metadata['dtype'])

    for rank, run in enumerate(runs):
        for case in run['cases']:
            if case['variant'] != 'full_response_eos':
                continue
            metadata = case['actual_coefficient_metadata']['conditional_fla']
            assert not metadata['diagnostics']
            assert sorted(int(index) for index in metadata['layers']) == list(LAYERS)
            for layer in metadata['layers'].values():
                assert layer['groups']
                for group in layer['groups']:
                    add_dtype('incoming_do', 'actual_seed', group['incoming_do'])
                    add_dtype('joint_state', 'h', group['incoming_state'])
                    for name, value in group['fields'].items():
                        add_dtype('coefficients', name, value['coefficient'])
            full_metadata.append(dict(rank=rank,
                original_owner_batch_index=case['original_owner_batch_index'], metadata=metadata))

    for original_report in gdn['actual_single_GDN_reports']:
        rank = original_report['rank']
        index = original_report['original_owner_batch_index']
        probe = original_report['probe_index']
        report = original_report['report']['conditional_fla']
        assert not report['diagnostics'] and report['score_calls'] == 1
        assert report['paired_root_capture_entries'] == 1
        assert report['recorded_layer_slot_scalars'] == report['expected_layer_slot_scalars']
        reports.append(dict(rank=rank, original_owner_batch_index=index, probe_index=probe, report=report))
        for metadata in report['capture_metadata'].values():
            for name, value in metadata['fields'].items():
                add_dtype('native_fields', name, value)
            add_dtype('native_initial_state', 'h0', metadata['actual_public_initial_state'])
        for fla in report['layers']:
            slot, layer = fla['original_slot'], fla['decoder_index']
            original = originals[rank, index, slot, probe, layer]
            effects = fla['original_GDN_effects']
            terms = fla['remaining_three_terms']
            assert set(terms) == set(TERMS)
            assert fla['coordinates']['prefix_cut'] == original['prefix_cut']
            assert fla['coordinates']['suffix_length'] == original['native_suffix_length']
            F = fla['F_input_projection']
            Y = fla['Y_public_output_projection']
            recomputed = dict(input_chain=effects['mixer_input']-effects['z_input']-F,
                FLA_projection=F-Y, native_cast_bridge=Y-effects['o_input'])
            net = sum(terms.values())
            values = {name: float(terms[name]) for name in TERMS}
            values.update({'input_projection.'+name: float(fla['fields'][name]) for name in INPUTS})
            values.update({'term_identity.'+name: terms[name]-recomputed[name] for name in TERMS})
            values.update(F=F, Y=Y, I=effects['mixer_input'], Z=effects['z_input'], O=effects['o_input'],
                remaining_input_FLA=original['remaining_input_FLA'], three_term_sum=net,
                sum_minus_remaining=net-original['remaining_input_FLA'],
                stored_closure=fla['remaining_closure'],
                recorded_sum_minus_recomputed=fla['remaining_three_term_sum']-net,
                recorded_remaining_minus_original_GDN=fla['original_GDN_remaining']-original['remaining_input_FLA'],
                F_minus_input_field_sum=F-sum(fla['fields'][name] for name in INPUTS),
                Y_minus_output_field=Y-fla['fields']['o'],
                recorded_FLA_difference_minus_recomputed=fla['conditional_FLA_difference']-(F-Y),
                native_d=original['native_d'])
            row = {name: original[name] for name in (
                'rank', 'original_owner_batch_index', 'slot', 'probe_index', 'decoder_index',
                'traj_uid', 'source_step', 'source_position', 'input_position', 'token_id',
                'prefix_cut', 'native_suffix_length', 'capture_start', 'coefficient_suffix_length')}
            row.update(values)
            row['actual_head_group_layouts'] = fla['group_layouts']
            row['actual_head_group_projections'] = fla['head_groups']
            rows.append(row)

    assert len(full_metadata) == 14 and len(rows) == 52*len(LAYERS)
    summaries, unique, cut_groups = [], [], []
    layout_fields = ('rank', 'original_owner_batch_index', 'slot', 'probe_index', 'prefix_cut',
                     'native_suffix_length', 'capture_start', 'coefficient_suffix_length')
    for layer in LAYERS:
        selected = [row for row in rows if row['decoder_index'] == layer]
        pool = reduce_replicas(selected, ('traj_uid', 'source_step', 'source_position'), FIELDS)
        assert pool['transport_slots'] == 52 and pool['unique_identities'] == 42
        for row in pool['rows']:
            members = [member for member in selected if all(member[key] == row[key]
                for key in ('traj_uid', 'source_step', 'source_position'))]
            row['actual_replica_layouts'] = [{name: member[name] for name in layout_fields} |
                dict(actual_head_group_layouts=member['actual_head_group_layouts']) for member in members]
        unique.append(dict(decoder_index=layer, **pool))
        summaries.append(dict(decoder_index=layer, transport_n=len(selected), unique_n=pool['unique_identities'],
            transport_statistics=_statistics(selected), unique_probe_statistics=_statistics(pool['rows']),
            maximum_replica_ranges=pool['max_replica_ranges']))
        for capture_start in sorted({row['capture_start'] for row in selected}):
            group = [row for row in selected if row['capture_start'] == capture_start]
            cut_groups.append(dict(decoder_index=layer, capture_start=capture_start,
                transport_n=len(group), unique_probe_identities=len({
                    (row['traj_uid'], row['source_step'], row['source_position']) for row in group}),
                original_group_coordinates=sorted({(row['rank'], row['original_owner_batch_index']) for row in group}),
                native_suffix_lengths=sorted({row['native_suffix_length'] for row in group}),
                actual_head_group_layouts=[dict(rank=row['rank'], original_owner_batch_index=row['original_owner_batch_index'],
                    slot=row['slot'], probe_index=row['probe_index'], groups=row['actual_head_group_layouts']) for row in group],
                transport_statistics=_statistics(group)))

    payloads = [dict(rank=entry['rank'], original_owner_batch_index=entry['original_owner_batch_index'],
        retained_cpu_tensor_bytes=entry['metadata']['retained_cpu_tensor_bytes'],
        scope=entry['metadata']['lifetime']) for entry in full_metadata]
    dependencies = ('analyze_textcraft_conditional_gdn.py', 'analyze_textcraft_conditional_primitives.py',
        'analyze_textcraft_conditional_boundaries.py', 'analyze_textcraft_matched_layout.py',
        'analyze_textcraft_native_readout.py', 'observe_textcraft_conditional_fla.py',
        'verify_textcraft_conditional_fla.py')
    return dict(scope=__doc__, status='descriptive_actual_FLA_projection_not_production_repair',
        sources=gdn['sources']+[source(Path(__file__))]+[
            source(Path(__file__).with_name(name)) for name in dependencies],
        checkpoint=gdn['checkpoint'], actual_imported_sources=gdn['actual_imported_sources'],
        actual_owner_options=gdn['actual_owner_options'], dtype=gdn['dtype'], calls=gdn['calls'],
        observed_tensor_dtypes={category: {name: sorted(values) for name, values in fields.items()}
            for category, fields in dtype_sets.items()},
        coverage=gdn['coverage'], layer_summaries=summaries, capture_start_groups=cut_groups,
        full_finite_FLA_metadata=full_metadata, actual_single_FLA_reports=reports,
        CPU_coefficient_bank_payloads=payloads,
        CPU_coefficient_bank_payload_statistics=describe([item['retained_cpu_tensor_bytes'] for item in payloads]),
        full_group_head_counts=dict(Counter(str(group['head_count']) for entry in full_metadata
            for layer in entry['metadata']['layers'].values() for group in layer['groups'])),
        transport_rows=rows, unique_probe_replica_means=unique,
        limitations=[
            'The three differences telescope the same-run GDN remaining projection; closure is not a credit-accuracy or official FLA tolerance test.',
            'F uses the actual final symmetric full-EOS q/k/v/beta/raw-g coefficients; Y uses the actual incoming do and original public FLA output. Neither is a newly fitted single-delete coefficient.',
            'Stage q/k are already L2-normalized; raw g and the FP16 public output come from the public RETURN, not the stage cumulative-g/output boundary.',
            'Actual compact-time and head8 partitions remain in raw metadata. Selected step0 transport cuts are distinct from all full-trace group cuts.',
            'Native cast bridge describes actual dtype-boundary projection differences; input-chain and FLA-projection terms still contain conditional approximation and native storage effects.',
            'Unique summaries explicitly average signed transport replicas and retain each range and original layout; the 42 probes are two dependent probes from 21 successful first-response UID.',
            'Recorded bank payload bytes are retained observation tensors, not process RSS or peak allocation; transient native operands are not a persistent extra bank.',
            'No numerical threshold, tolerance change, rescaling, correction, model call, reference change or training update.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=LOCAL)
    parser.add_argument('--pack', type=Path, default=PACK)
    parser.add_argument('--matched-map', type=Path, default=MATCHED_MAP)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build_fla_analysis(args.input_dir, args.pack, args.matched_map)
    output = args.output or args.input_dir/'conditional-fla-analysis.json'
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=source(output), coverage=result['coverage'],
        layers=[dict(decoder_index=layer['decoder_index'],
            terms={name: {key: layer['unique_probe_statistics'][name][key]
                for key in ('abs_mean', 'rms', 'positive', 'negative', 'zero')} for name in TERMS},
            closure=layer['unique_probe_statistics']['sum_minus_remaining'],
            maximum_replica_ranges=layer['maximum_replica_ranges']) for layer in result['layer_summaries']],
        actual_dtypes=result['observed_tensor_dtypes'], payload=result['CPU_coefficient_bank_payload_statistics']),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
