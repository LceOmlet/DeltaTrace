"""Independent stdlib review of actual symmetric-FLA order observations.

Read complete original rank JSONs and the literal matched-layout pack summary.
Do not import observation/analysis modules, Torch, Ray or a connector. Preserve
all transport replicas; unique identities use an explicit signed replica mean.
All distances and cancellation quantities are descriptive, without tolerances.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


LOCAL = Path(__file__).resolve().parent
DEG = LOCAL.parents[1]
PACK = DEG / 'native-minibatch-v4/native64-first-response-matched-layout-summary.json'
PACK_SHA = '2e8390747c02801bdb763e4eac76112c0ac4caf3f513893f8218e7124e560f27'
LAYERS = (6, 8)
INPUTS = ('q', 'k', 'v', 'beta', 'g')
NUMBERS = ('F0', 'F1', 'Favg', 'Y', 'mean_orders', 'F0_minus_Y', 'F1_minus_Y',
    'Favg_minus_Y', 'mean_orders_minus_Y', 'average_closure', 'head_field_closure_absmax',
    'F_field_closure_absmax', 'stored_order0_residual_difference', 'stored_order1_residual_difference',
    'stored_average_residual_difference', 'stored_mean_difference', 'stored_average_closure_difference',
    'stored_projection_cancellation_difference') + tuple(
        f'order{order}.{name}' for order in (0, 1) for name in INPUTS)


def provenance(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def describe(values):
    values = [float(value) for value in values]
    if not values:
        return dict(n=0)
    assert all(math.isfinite(value) for value in values)
    ordered = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
        min=min(values), max=max(values), abs_mean=statistics.mean(abs(value) for value in values),
        rms=math.sqrt(statistics.mean(value*value for value in values)),
        positive=sum(value > 0 for value in values), negative=sum(value < 0 for value in values),
        zero=sum(value == 0 for value in values),
        quantiles={str(q): ordered[round(q*(len(ordered)-1))] for q in (0, .1, .25, .5, .75, .9, 1)})


def sample_identity(sample):
    return {key: sample[key] for key in ('traj_uid', 'source_step', 'original_request_index',
        'source_start', 'source_end', 'probe_source_positions', 'probe_input_positions')}


def cancellation(row):
    projection_mean = (row['F0']+row['F1'])*.5
    error_mean = (row['F0_minus_Y']+row['F1_minus_Y'])*.5
    return dict(order_projection_opposite_signs=row['F0']*row['F1'] < 0,
        order_residual_opposite_signs=row['F0_minus_Y']*row['F1_minus_Y'] < 0,
        projection_cancelled_absolute_mass=(abs(row['F0'])+abs(row['F1']))*.5-abs(projection_mean),
        residual_cancelled_absolute_mass=(abs(row['F0_minus_Y'])+abs(row['F1_minus_Y']))*.5-abs(error_mean),
        order0_closer_to_same_Y=abs(row['F0_minus_Y']) < abs(row['F1_minus_Y']),
        order1_closer_to_same_Y=abs(row['F1_minus_Y']) < abs(row['F0_minus_Y']),
        same_distance_to_Y=abs(row['F0_minus_Y']) == abs(row['F1_minus_Y']))


def summarize(rows):
    nonlinear = [cancellation(row) for row in rows]
    return dict(n=len(rows), numeric_statistics={name: describe([row[name] for row in rows]) for name in NUMBERS},
        nonlinear_statistics={name: describe([row[name] for row in nonlinear]) for name in (
            'projection_cancelled_absolute_mass', 'residual_cancelled_absolute_mass')},
        sign_and_distance_counts={name: sum(row[name] for row in nonlinear) for name in (
            'order_projection_opposite_signs', 'order_residual_opposite_signs',
            'order0_closer_to_same_Y', 'order1_closer_to_same_Y', 'same_distance_to_Y')})


def build_review(input_dir, pack_path):
    input_dir, pack_path = Path(input_dir), Path(pack_path)
    pack = json.loads(pack_path.read_bytes())
    assert pack['original_full_pack']['sha256'] == PACK_SHA
    paths = [input_dir/f'rank{rank}-readout.json' for rank in (0, 1)]
    runs = [json.loads(path.read_bytes()) for path in paths]
    completed_path = input_dir/'completed.json'
    completed = json.loads(completed_path.read_bytes())
    assert [run['rank'] for run in runs] == [0, 1]
    assert all(run['phase'] == 'complete_native_readout' and run['input_sha256'] == PACK_SHA for run in runs)
    assert [len(run['cases']) for run in runs] == [21, 21]
    assert runs[0]['checkpoint'] == runs[1]['checkpoint']
    assert all(run['optimizer_steps'] == run['scheduler_steps'] == run['backward_calls'] == 0 for run in runs)
    full_metadata, native_metadata, rows = [], [], []
    calls = []
    owner_sources = []
    state_differences = []
    dtype_sets = defaultdict(set)

    for rank, run in enumerate(runs):
        owner_sources.append(dict(rank=rank, sources=run['sources']))
        calls.append(dict(rank=rank, **{name:run[name] for name in ('finite_trace_calls', 'finite_seed_calls',
            'native_forward_calls', 'native_root_calls', 'native_prefix_calls', 'optimizer_steps',
            'scheduler_steps', 'backward_calls')}))
        full_by_group = {}
        groups = {group['original_owner_batch_index']:group for group in pack['rank_groups'][rank]}
        for case in run['cases']:
            index = case['original_owner_batch_index']
            assert [sample_identity(s) for s in case['samples']] == [sample_identity(s) for s in groups[index]['samples']]
            if case['variant'] != 'full_response_eos':
                continue
            original = case['actual_coefficient_metadata']['conditional_fla']
            orders = original['conditional_orders']
            assert not original['diagnostics'] and not orders['diagnostics']
            assert orders['recorded_original_avg_returns'] == orders['expected_original_avg_returns']
            assert sorted(int(layer) for layer in orders['layers']) == list(LAYERS)
            full_by_group[index] = original
            full_metadata.append(dict(rank=rank, original_owner_batch_index=index, metadata=original))
            for layer in original['layers'].values():
                for group in layer['groups']:
                    dtype_sets['joint.do'].add(group['incoming_do']['dtype'])
                    dtype_sets['joint.h'].add(group['incoming_state']['dtype'])
                    state_differences.extend(group['incoming_state_pair_absmax'].values())
            for layer in orders['layers'].values():
                for group in layer['groups']:
                    for order in group['orders']:
                        for name, field in order['fields'].items():
                            dtype_sets[f'order{order["order"]}.{name}'].add(field['original']['dtype'])

        assert len(full_by_group) == 7
        for case in run['cases']:
            if case['variant'] == 'full_response_eos':
                continue
            index = case['original_owner_batch_index']
            probe = int(case['variant'].rsplit('_', 1)[1])
            fla = case['conditional_boundary_contractions']['conditional_primitives']['conditional_gdn']['conditional_fla']
            orders = fla['conditional_orders']
            assert not fla['diagnostics'] and not orders['diagnostics']
            assert orders['recorded_layer_slot_scalars'] == orders['expected_layer_slot_scalars']
            assert orders['reused_average_contraction_calls'] == orders['expected_reused_average_contraction_calls']
            assert orders['added_CPU_order_contractions'] == 2*orders['reused_average_contraction_calls']
            actuals = {(row['decoder_index'], row['original_slot']):row for row in fla['layers']}
            native_metadata.append(dict(rank=rank, original_owner_batch_index=index, probe_index=probe,
                coordinates=fla['coordinates'], capture_metadata=fla['capture_metadata'], orders_report=orders))
            for metadata in fla['capture_metadata'].values():
                for name, field in metadata['fields'].items():
                    dtype_sets['native.'+name].add(field['dtype'])
                if metadata['actual_public_initial_state'] is not None:
                    dtype_sets['native.h0'].add(metadata['actual_public_initial_state']['dtype'])
            for row in orders['layers']:
                layer, slot = row['decoder_index'], row['original_slot']
                sample = case['samples'][slot]
                assert layer in LAYERS and sample['source_step'] == 0
                assert row['paired_rows'] == [2*slot, 2*slot+1]
                actual = actuals[layer, slot]
                assert row['coordinates'] == actual['coordinates']
                assert row['group_layouts'] == actual['group_layouts']
                assert row['actual_F_average'] == actual['F_input_projection']
                assert row['Y_public_output_projection'] == actual['Y_public_output_projection']
                assert [order['order'] for order in row['orders']] == [0, 1]
                head_differences, F_differences = [], []
                for order in row['orders']:
                    assert set(order['fields']) == set(INPUTS)
                    assert len(order['head_groups']) == len(row['group_layouts'])
                    for name in INPUTS:
                        head_differences.append(order['fields'][name]-sum(head[name] for head in order['head_groups']))
                    F_differences.append(row['F_order'+str(order['order'])]-sum(order['fields'].values()))
                F0, F1, avg, Y = (row[name] for name in ('F_order0', 'F_order1', 'actual_F_average', 'Y_public_output_projection'))
                mean = (F0+F1)*.5
                values = dict(F0=F0, F1=F1, Favg=avg, Y=Y, mean_orders=mean,
                    F0_minus_Y=F0-Y, F1_minus_Y=F1-Y, Favg_minus_Y=avg-Y,
                    mean_orders_minus_Y=mean-Y, average_closure=avg-mean,
                    head_field_closure_absmax=max(abs(value) for value in head_differences),
                    F_field_closure_absmax=max(abs(value) for value in F_differences),
                    stored_order0_residual_difference=row['F_order0_minus_Y']-(F0-Y),
                    stored_order1_residual_difference=row['F_order1_minus_Y']-(F1-Y),
                    stored_average_residual_difference=row['actual_average_minus_Y']-(avg-Y),
                    stored_mean_difference=row['order_projection_mean']-mean,
                    stored_average_closure_difference=row['actual_average_closure_difference']-(avg-mean),
                    stored_projection_cancellation_difference=row['absolute_projection_cancellation']-((abs(F0)+abs(F1))*.5-abs(mean)))
                for order in row['orders']:
                    values.update({f'order{order["order"]}.{name}':order['fields'][name] for name in INPUTS})
                saved = sample['saved_probe_credit'][probe]
                assert saved['source_position'] == sample['probe_source_positions'][probe]
                assert saved['input_position'] == sample['probe_input_positions'][probe]
                record = dict(rank=rank, original_owner_batch_index=index, original_slot=slot, probe_index=probe,
                    decoder_index=layer, traj_uid=sample['traj_uid'], source_step=sample['source_step'],
                    source_position=sample['probe_source_positions'][probe], input_position=sample['probe_input_positions'][probe],
                    token_id=saved['token_id'], saved_DT_d=saved['d'], saved_DT_d_scope=saved['d_scope'],
                    coordinates=row['coordinates'], group_layouts=row['group_layouts'], raw_orders=row['orders'], **values)
                record['nonlinear_descriptions'] = cancellation(record)
                rows.append(record)

    assert len(rows) == 104 and len(full_metadata) == 14 and len(native_metadata) == 28
    unique, summaries = [], []
    for layer in LAYERS:
        selected = [row for row in rows if row['decoder_index'] == layer]
        pools = defaultdict(list)
        for row in selected:
            pools[row['traj_uid'], row['source_step'], row['source_position']].append(row)
        assert len(selected) == 52 and len(pools) == 42
        reduced = []
        for identity, members in sorted(pools.items()):
            replicas = {name:dict(mean=statistics.mean(member[name] for member in members),
                min=min(member[name] for member in members), max=max(member[name] for member in members),
                range=max(member[name] for member in members)-min(member[name] for member in members)) for name in NUMBERS}
            row = dict(traj_uid=identity[0], source_step=identity[1], source_position=identity[2], decoder_index=layer,
                replica_count=len(members), replica_values=replicas,
                transport_locations=[{name:member[name] for name in ('rank', 'original_owner_batch_index', 'original_slot',
                    'probe_index', 'input_position', 'coordinates', 'group_layouts')} for member in members],
                **{name:replicas[name]['mean'] for name in NUMBERS})
            row['nonlinear_descriptions'] = cancellation(row)
            reduced.append(row)
        summaries.append(dict(decoder_index=layer, transport_statistics=summarize(selected),
            unique_statistics=summarize(reduced), replica_count_histogram=dict(Counter(row['replica_count'] for row in reduced)),
            maximum_replica_ranges={name:max(row['replica_values'][name]['range'] for row in reduced) for name in NUMBERS}))
        unique.append(dict(decoder_index=layer, rows=reduced))

    return dict(scope=__doc__, status='independent_descriptive_raw_review_not_kernel_tolerance_acceptance',
        sources=[provenance(path) for path in (*paths, completed_path, pack_path, Path(__file__))],
        input_literal_pack=pack['original_full_pack'], original_owner_sources=owner_sources,
        checkpoint=runs[0]['checkpoint'], completed=completed, calls=calls,
        coverage=dict(raw_cases=42, full_traces=14, single_root_reports=28,
            transport_probes_per_layer=52, unique_probes_per_layer=42, decoder_indices=list(LAYERS),
            transport_layer_records=len(rows), unique_layer_records=sum(len(layer['rows']) for layer in unique),
            successful_first_response_UIDs=len({row['traj_uid'] for row in rows})),
        observed_dtypes={name:sorted(values) for name,values in dtype_sets.items()},
        actual_fixed_incoming_state_pair_differences=describe(state_differences),
        layer_summaries=summaries, transport_rows=rows, unique_signed_replica_means=unique,
        full_original_order_metadata=full_metadata, single_native_order_metadata=native_metadata,
        limitations=[
            'Independent direct raw parsing and arithmetic; no observation/analyzer modules, Torch, model or connector imported.',
            'F0/F1 are the original joint full-EOS order coefficients projected on fixed single-EOS native operands, not fresh single-token finite coefficients or world counterfactual values.',
            'Both orders share the original do and orientation of factual-minus-reference operands; order1 is not negated.',
            'Distances to the same Y and order cancellation are descriptive. They do not identify a correct order or establish an official FLA kernel tolerance failure.',
            'Unique summaries first average signed primitive values over transport replicas, then compute absolute distances/sign/cancellation. All raw replicas and ranges remain visible.',
            'Only 21 successful first-response UID with two fixed probes each; no full-training-token prevalence or degradation main-cause claim.',
            'No tolerance, rescaling, correction, alternative reference, optimizer update or production change.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=LOCAL)
    parser.add_argument('--pack', type=Path, default=PACK)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build_review(args.input_dir, args.pack)
    output = args.output or args.input_dir/'independent-orders-review.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(output=provenance(output), coverage=result['coverage'],
        actual_dtypes=result['observed_dtypes'], layers=result['layer_summaries']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
