"""Describe actual GDN projections without changing credit or tolerances.

Reuse the primitive analysis's original boundary parser and replica reduction.
Keep all 52 transport probes, then separately average replicas of each of the
42 UID/source-position identities, for layers6/8. Three-term cancellation is
arithmetic cancellation of recorded local projections, not a causal share.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_textcraft_conditional_primitives import (
    build_analysis, PACK, MATCHED_MAP, _coordinate, reduce_replicas, describe, source,
)
from stage_textcraft_conditional_gdn import LOCAL


LAYERS = (6, 8)
TERMS = ('out_proj', 'norm_gate', 'remaining_input_FLA')
EFFECTS = ('mixer_input', 'mixer_output', 'gated_output', 'o_input', 'z_input')
FIELDS = TERMS + tuple('effect.'+name for name in EFFECTS) + (
    'mixer_gap', 'existing_primitive_mixer_gap', 'three_term_sum',
    'sum_minus_mixer', 'stored_closure', 'mixer_minus_existing_primitive',
    'recorded_sum_minus_recomputed', 'recorded_mixer_minus_recomputed',
    'native_d', 'effect_identity.out_proj', 'effect_identity.norm_gate',
    'effect_identity.remaining_input_FLA')


def _sign(value):
    return 'positive' if value > 0 else 'negative' if value < 0 else 'zero'


def _cancellation(row):
    magnitudes = sum(abs(row[name]) for name in TERMS)
    signed_sum = sum(row[name] for name in TERMS)
    return dict(sum_of_term_absolute_values=magnitudes, absolute_signed_sum=abs(signed_sum),
                cancelled_absolute_mass=magnitudes-abs(signed_sum))


def _summary(rows):
    cancellations = [_cancellation(row) for row in rows]
    return dict(n=len(rows), terms={name: dict(
        values=describe([row[name] for row in rows]),
        signs_relative_to_mixer=dict(Counter(
            _sign(row[name])+'|'+_sign(row['mixer_gap']) for row in rows)),
        same_nonzero_sign_count=sum(row[name]*row['mixer_gap'] > 0 for row in rows),
        opposite_sign_count=sum(row[name]*row['mixer_gap'] < 0 for row in rows)) for name in TERMS},
        mixer_gap=describe([row['mixer_gap'] for row in rows]),
        existing_primitive_mixer_gap=describe([row['existing_primitive_mixer_gap'] for row in rows]),
        three_term_closure=describe([row['sum_minus_mixer'] for row in rows]),
        maximum_absolute_three_term_closure=max(abs(row['sum_minus_mixer']) for row in rows),
        mixer_minus_existing_primitive=describe([row['mixer_minus_existing_primitive'] for row in rows]),
        maximum_absolute_mixer_difference=max(abs(row['mixer_minus_existing_primitive']) for row in rows),
        recorded_arithmetic_differences={name: describe([row[name] for row in rows]) for name in (
            'stored_closure', 'recorded_sum_minus_recomputed', 'recorded_mixer_minus_recomputed',
            'effect_identity.out_proj', 'effect_identity.norm_gate', 'effect_identity.remaining_input_FLA')},
        cancellation={name: describe([row[name] for row in cancellations])
                      for name in cancellations[0]},
        cancellation_scope='sum(abs(three terms))-abs(sum(three terms)) per row. For unique probes this is computed after the explicit replica mean of signed terms; no causal percentage.')


def build_gdn_analysis(input_dir, pack_path=PACK, matched_map_path=MATCHED_MAP):
    input_dir = Path(input_dir)
    boundary = build_analysis(input_dir, pack_path, matched_map_path)
    transport = {_coordinate(row): row for row in boundary['transport_observations']}
    paths = [input_dir/f'rank{rank}-readout.json' for rank in range(2)]
    runs = [json.loads(path.read_bytes()) for path in paths]
    rows, reports, full_metadata = [], [], []
    dtype_sets = dict(coefficients={}, native_fields={}, native_flat_norm={})

    def add_dtype(category, field, metadata):
        dtype_sets[category].setdefault(field, set()).add(metadata['dtype'])

    for rank, run in enumerate(runs):
        for case in run['cases']:
            index = case['original_owner_batch_index']
            if case['variant'] == 'full_response_eos':
                metadata = case['actual_coefficient_metadata']['conditional_gdn']
                assert metadata['captured_layer_indices'] == list(LAYERS)
                assert not metadata['diagnostics']
                for layer in metadata['layers'].values():
                    assert not layer['diagnostics'] and layer['norm_gate_calls'] == 1
                    for name, value in layer['coefficients'].items():
                        add_dtype('coefficients', name, value['original'])
                full_metadata.append(dict(rank=rank, original_owner_batch_index=index, metadata=metadata))
                continue
            probe = int(case['variant'].rsplit('_', 1)[1])
            primitive = case['conditional_boundary_contractions']['conditional_primitives']
            report = primitive['conditional_gdn']
            assert not report['diagnostics'] and report['score_calls'] == 1
            assert report['paired_root_capture_entries'] == 1
            assert report['recorded_layer_slot_scalars'] == report['expected_layer_slot_scalars']
            original_mixers = {(row['decoder_index'], row['original_slot']): row
                               for row in primitive['layers']}
            reports.append(dict(rank=rank, original_owner_batch_index=index, probe_index=probe, report=report))
            for metadata in report['capture_metadata'].values():
                for name, value in metadata['fields'].items():
                    add_dtype('native_fields', name, value['actual'])
                for name, value in metadata['norm_flat_fields'].items():
                    add_dtype('native_flat_norm', name, value)
            for gdn in report['layers']:
                slot, layer = gdn['original_slot'], gdn['decoder_index']
                assert layer in LAYERS
                original = transport[rank, index, slot, probe]
                original_mixer = original_mixers[layer, slot]
                assert gdn['prefix_cut'] == original_mixer['prefix_cut']
                assert gdn['suffix_length'] == original_mixer['suffix_length']
                assert gdn['coefficient_suffix_length'] == gdn['suffix_length']-gdn['capture_start']
                effects = gdn['effects']
                terms = gdn['three_terms']
                assert set(terms) == set(TERMS)
                recomputed = dict(out_proj=effects['gated_output']-effects['mixer_output'],
                    norm_gate=effects['o_input']+effects['z_input']-effects['gated_output'],
                    remaining_input_FLA=effects['mixer_input']-effects['o_input']-effects['z_input'])
                net = sum(terms.values())
                mixer = effects['mixer_input']-effects['mixer_output']
                existing = original_mixer['six_terms']['mixer']
                values = {name: float(terms[name]) for name in TERMS}
                values.update({'effect.'+name: float(effects[name]) for name in EFFECTS})
                values.update({'effect_identity.'+name: terms[name]-recomputed[name] for name in TERMS})
                values.update(mixer_gap=mixer, existing_primitive_mixer_gap=existing,
                    three_term_sum=net, sum_minus_mixer=net-mixer,
                    stored_closure=gdn['arithmetic_closure_residual'],
                    mixer_minus_existing_primitive=mixer-existing,
                    recorded_sum_minus_recomputed=gdn['three_term_sum']-net,
                    recorded_mixer_minus_recomputed=gdn['primitive_mixer_difference']-mixer,
                    native_d=original['native_d'])
                row = dict(rank=rank, original_owner_batch_index=index, slot=slot,
                    probe_index=probe, decoder_index=layer, traj_uid=original['traj_uid'],
                    source_step=original['source_step'], source_position=original['source_position'],
                    input_position=original['input_position'], token_id=original['token_id'],
                    prefix_cut=gdn['prefix_cut'], native_suffix_length=gdn['suffix_length'],
                    capture_start=gdn['capture_start'], coefficient_suffix_length=gdn['coefficient_suffix_length'],
                    **values)
                row['cancellation'] = _cancellation(row)
                rows.append(row)
    assert len(rows) == 52*len(LAYERS) and len(full_metadata) == 14
    assert len({(row['traj_uid'],row['source_step']) for row in rows}) == 21
    summaries, unique, start_groups = [], [], []
    for layer in LAYERS:
        selected = [row for row in rows if row['decoder_index'] == layer]
        pool = reduce_replicas(selected, ('traj_uid','source_step','source_position'), FIELDS)
        assert pool['unique_identities'] == 42 and pool['transport_slots'] == 52
        for row in pool['rows']:
            members = [member for member in selected if all(member[key] == row[key] for key in
                ('traj_uid','source_step','source_position'))]
            row['actual_replica_layouts'] = [{key: member[key] for key in (
                'rank','original_owner_batch_index','slot','probe_index','prefix_cut',
                'native_suffix_length','capture_start','coefficient_suffix_length')} for member in members]
        unique.append(dict(decoder_index=layer, **pool))
        summaries.append(dict(decoder_index=layer, transport_statistics=_summary(selected),
            unique_probe_statistics=_summary(pool['rows']), maximum_replica_ranges=pool['max_replica_ranges']))
        for start in sorted({row['capture_start'] for row in selected}):
            group = [row for row in selected if row['capture_start'] == start]
            start_groups.append(dict(decoder_index=layer, capture_start=start,
                unique_probe_identities=len({(row['traj_uid'],row['source_step'],row['source_position']) for row in group}),
                original_group_coordinates=sorted({(row['rank'],row['original_owner_batch_index']) for row in group}),
                native_suffix_lengths=sorted({row['native_suffix_length'] for row in group}),
                coefficient_suffix_lengths=sorted({row['coefficient_suffix_length'] for row in group}),
                transport_statistics=_summary(group)))
    dependencies = ('analyze_textcraft_conditional_primitives.py','analyze_textcraft_conditional_boundaries.py',
                    'analyze_textcraft_matched_layout.py','analyze_textcraft_native_readout.py',
                    'observe_textcraft_conditional_gdn.py','stage_textcraft_conditional_gdn.py')
    return dict(scope=__doc__, status='descriptive_actual_GDN_projection_not_production_repair',
        sources=boundary['sources']+[source(Path(__file__))]+[
            source(Path(__file__).with_name(name)) for name in dependencies],
        checkpoint=boundary['checkpoint'], actual_imported_sources=boundary['actual_imported_sources'],
        actual_owner_options=boundary['actual_owner_options'], dtype=boundary['dtype'], calls=boundary['calls'],
        observed_tensor_dtypes={category:{field:sorted(values) for field,values in fields.items()}
                               for category,fields in dtype_sets.items()},
        coverage=dict(transport_probes=52, unique_probes=42, successful_first_response_UIDs=21,
                      decoder_indices=list(LAYERS), transport_probe_layer_records=len(rows),
                      unique_probe_layer_records=42*len(LAYERS)),
        layer_summaries=summaries, capture_start_groups=start_groups,
        full_finite_GDN_metadata=full_metadata, actual_single_GDN_reports=reports,
        transport_rows=rows, unique_probe_replica_means=unique,
        limitations=[
            'Three-term closure is an observation arithmetic identity, not a token-credit accuracy or official kernel test.',
            'm/mo/mz are actual joint full-EOS coefficients. Native o/z/gated retain the same actual capture_start suffix; mixer input/output contractions retain the complete suffix.',
            'remaining_input_FLA includes the original convolution, projections, FLA and storage; it does not isolate a FLA kernel defect.',
            'The same-run primitive mixer difference is compared directly; no value is rescaled or forced to match it.',
            'Unique summaries use arithmetic means of signed replica values; original replicas and their capture_start layouts and ranges remain recorded.',
            'Twenty-one successful first-response UID and two fixed probes each are a dependent selected sample, not all training tokens or independent causal cases.',
            'Recorded dtypes cover actual captured interfaces only; native_mo after casting and recurrent states were not additionally captured.',
            'No numerical threshold, tolerance change, correction, global causal percentage, model call or training change.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=LOCAL)
    parser.add_argument('--pack', type=Path, default=PACK)
    parser.add_argument('--matched-map', type=Path, default=MATCHED_MAP)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build_gdn_analysis(args.input_dir, args.pack, args.matched_map)
    output = args.output or args.input_dir/'conditional-gdn-analysis.json'
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')
    compact_layers = []
    for layer in result['layer_summaries']:
        unique = layer['unique_probe_statistics']
        compact_layers.append(dict(decoder_index=layer['decoder_index'],
            terms={name:{key:spec['values'][key] for key in ('abs_mean','rms','positive','negative','zero')}
                   for name,spec in unique['terms'].items()},
            cancellation_abs_mean=unique['cancellation']['cancelled_absolute_mass']['abs_mean'],
            maximum_absolute_three_term_closure=unique['maximum_absolute_three_term_closure'],
            maximum_absolute_mixer_difference=unique['maximum_absolute_mixer_difference'],
            maximum_absolute_transport_closure=layer['transport_statistics']['maximum_absolute_three_term_closure'],
            maximum_absolute_transport_mixer_difference=layer['transport_statistics']['maximum_absolute_mixer_difference']))
    print(json.dumps(dict(output=source(output), coverage=result['coverage'], layers=compact_layers,
        capture_start_groups=[{key:group[key] for key in ('decoder_index','capture_start','unique_probe_identities')}
                              | {'transport_n':group['transport_statistics']['n']} for group in result['capture_start_groups']],
        actual_dtypes=result['observed_tensor_dtypes']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
