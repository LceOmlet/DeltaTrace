"""Aggregate actual native primitive projections without a numerical gate.

Reuse the original boundary parser and its exact transport/probe identities.
All replicas remain recorded; unique probe summaries use the same explicit
arithmetic replica means.  These are local conditional residuals, not causal
percentages, official kernel acceptance or a repaired learning signal.
"""
import argparse
import json
from pathlib import Path

from analyze_textcraft_conditional_boundaries import build_analysis, PACK, MATCHED_MAP, _coordinate
from analyze_textcraft_matched_layout import reduce_replicas
from analyze_textcraft_native_readout import describe, source


LOCAL = Path(__file__).parent / 'textcraft-degradation-20261005/conditional-primitives-20261006/v1'


def flatten(row):
    values = dict(Cin=row['Cin'], Cout=row['Cout'], Cin_minus_Cout=row['Cin_minus_Cout'],
                  six_term_sum=row['six_term_sum'], six_term_closure=row['arithmetic_closure_residual'])
    values.update({'six.'+key: value for key, value in row['six_terms'].items()})
    for name, norm in row['norms'].items():
        values.update({f'norm.{name}.{key}': value for key, value in norm['decomposition'].items()})
        values[f'norm.{name}.gap'] = norm['joint_native_gap']
        values[f'norm.{name}.closure'] = norm['arithmetic_closure_residual']
        values[f'norm.{name}.single_FP32_identity_residual'] = norm['single_pair_FP32_identity_residual']
    for name, addition in row['residual_additions'].items():
        values.update({f'add.{name}.{key}': value for key, value in addition['decomposition'].items()})
        values[f'add.{name}.gap'] = addition['branch_sum_minus_actual_output']
        values[f'add.{name}.closure'] = addition['arithmetic_closure_residual']
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=LOCAL)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    boundary = build_analysis(args.input_dir, PACK, MATCHED_MAP)
    boundary_path = args.input_dir / 'conditional-boundary-analysis.json'
    boundary_path.write_text(json.dumps(boundary, ensure_ascii=False, separators=(',', ':'),
        allow_nan=False)+'\n', encoding='utf-8')
    transport = {_coordinate(row): row for row in boundary['transport_observations']}
    runs = [json.loads((args.input_dir/f'rank{rank}-readout.json').read_bytes()) for rank in range(2)]
    rows, raw_reports, full_metadata = [], [], []
    for rank, run in enumerate(runs):
        for case in run['cases']:
            index = case['original_owner_batch_index']
            if case['variant'] == 'full_response_eos':
                meta = case['actual_coefficient_metadata']['conditional_primitives']
                assert meta['all_requested_layers_present'] and not meta['diagnostics']
                assert all(not layer['diagnostics'] for layer in meta['layers'].values())
                full_metadata.append(dict(rank=rank, original_owner_batch_index=index, metadata=meta))
                continue
            probe = int(case['variant'].rsplit('_', 1)[1])
            report = case['conditional_boundary_contractions']['conditional_primitives']
            assert not report['diagnostics'] and report['score_calls'] == 1
            assert report['paired_root_capture_entries'] == 1
            assert report['recorded_layer_slot_scalars'] == report['expected_layer_slot_scalars']
            raw_reports.append(dict(rank=rank, original_owner_batch_index=index, probe_index=probe, report=report))
            for primitive in report['layers']:
                key = (rank, index, primitive['original_slot'], probe)
                original = transport[key]
                layer = primitive['decoder_index']
                values = flatten(primitive)
                values['Cin_minus_boundary'] = primitive['Cin']-original['boundary_C'][layer]
                values['Cout_minus_boundary'] = primitive['Cout']-original['boundary_C'][layer+1]
                rows.append(dict(rank=rank, original_owner_batch_index=index,
                    slot=primitive['original_slot'], probe_index=probe,
                    decoder_index=layer, block_type=primitive['block_type'],
                    traj_uid=original['traj_uid'], source_step=original['source_step'],
                    source_position=original['source_position'], native_d=original['native_d'], **values))
    assert len(rows) == 52*3 and len(full_metadata) == 14
    fields = tuple(flatten(raw_reports[0]['report']['layers'][0])) + (
        'Cin_minus_boundary', 'Cout_minus_boundary', 'native_d')
    summaries, unique = [], []
    for layer in (6, 8, 11):
        selected = [row for row in rows if row['decoder_index'] == layer]
        pool = reduce_replicas(selected, ('traj_uid', 'source_step', 'source_position'), fields)
        assert pool['unique_identities'] == 42 and pool['transport_slots'] == 52
        unique.append(dict(decoder_index=layer, **pool))
        summaries.append(dict(decoder_index=layer, block_type=selected[0]['block_type'],
            transport_n=len(selected), unique_n=pool['unique_identities'],
            unique_probe_statistics={key: describe([row[key] for row in pool['rows']]) for key in fields},
            transport_statistics={key: describe([row[key] for row in selected]) for key in fields},
            maximum_replica_ranges=pool['max_replica_ranges']))
    output = args.output or args.input_dir/'conditional-primitive-analysis.json'
    result = dict(scope=__doc__, status='actual_primitive_residual_observed_not_production_repair',
        sources=[source(args.input_dir/f'rank{rank}-readout.json') for rank in range(2)] +
            [source(boundary_path), source(Path(__file__)), source(Path(__file__).with_name('observe_textcraft_conditional_primitives.py'))],
        coverage=dict(transport_records=156, unique_probe_layer_records=126, unique_probes=42,
            unique_successful_first_response_UIDs=21, decoder_indices=[6, 8, 11]),
        layer_summaries=summaries, full_finite_observation_metadata=full_metadata,
        actual_single_reports=raw_reports, transport_rows=rows, unique_probe_replica_means=unique,
        limitations=[
            'Six-term closure checks observation arithmetic only; it is not a credit-accuracy test.',
            'Norm joint-minus-single term includes arithmetic in actual saved joint coefficients, not solely curvature.',
            'MLP and mixer residuals have not separated joint approximation from low-precision arithmetic.',
            'CPU original norm/storage comparisons report GPU/CPU discrepancies separately.',
            'Dependent two-probe/21-successful-UID sample, not full training-token defect prevalence.',
            'No correction, threshold, clipping, normalization, PPO change or optimizer update.'])
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=source(output), coverage=result['coverage'], layer_summaries=summaries), indent=2))


if __name__ == '__main__':
    main()
