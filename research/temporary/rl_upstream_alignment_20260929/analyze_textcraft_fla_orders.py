"""Describe the two existing symmetric-FLA order projections on saved data.

Reuse the prior original-owner FLA parser, identities, replica reduction and
descriptive statistics. The output compares the existing average with each
order against the same actual native public-output projection Y. It selects
no replacement rule and adds no model call, tolerance, scaling or correction.
"""
import argparse
import json
from pathlib import Path

from analyze_textcraft_conditional_fla import (
    build_fla_analysis, LAYERS, PACK, MATCHED_MAP, reduce_replicas, describe, source,
    _row_key, INPUTS,
)


LOCAL = Path(__file__).parent / 'textcraft-degradation-20261005/fla-orders-20261006/v1'
FIELDS = ('F0', 'F1', 'F_average', 'Y', 'R0', 'R1', 'R_average', 'order_mean',
    'order_abs_mean', 'order_cancellation', 'average_closure_difference',
    'F0_minus_input_field_sum', 'F1_minus_input_field_sum',
    'recorded_order_mean_minus_recomputed', 'recorded_average_closure_minus_recomputed',
    'recorded_order_abs_mean_minus_recomputed', 'recorded_order_cancellation_minus_recomputed',
    'recorded_R0_minus_recomputed', 'recorded_R1_minus_recomputed',
    'recorded_R_average_minus_recomputed', 'average_minus_existing_F', 'Y_minus_existing_Y',
    'native_d') + tuple(f'order{order}.{name}' for order in (0,1) for name in INPUTS)


def statistics(rows):
    return {name:describe([row[name] for row in rows]) for name in FIELDS}


def comparisons(rows):
    """Compare actual distances/signs; exact ties are reported, not a tolerance."""
    def relation(average, order):
        return 'closer' if average < order else 'farther' if average > order else 'equal'
    pairs = {name:{value:0 for value in ('closer','farther','equal')} for name in ('order0','order1')}
    opposite = []
    cancellation = []
    differences = {'average_abs_error_minus_order0':[], 'average_abs_error_minus_order1':[]}
    closer_both = farther_both = 0
    for row in rows:
        f0, f1, average, Y = (row[name] for name in ('F0','F1','F_average','Y'))
        errors = (abs(f0-Y),abs(f1-Y),abs(average-Y))
        pairs['order0'][relation(errors[2],errors[0])] += 1
        pairs['order1'][relation(errors[2],errors[1])] += 1
        closer_both += errors[2] < errors[0] and errors[2] < errors[1]
        farther_both += errors[2] > errors[0] and errors[2] > errors[1]
        differences['average_abs_error_minus_order0'].append(errors[2]-errors[0])
        differences['average_abs_error_minus_order1'].append(errors[2]-errors[1])
        cancellation.append((abs(f0)+abs(f1))*0.5-abs((f0+f1)*0.5))
        if f0*f1 < 0:
            opposite.append(row)
    return dict(n=len(rows), opposite_order_sign_count=len(opposite),
        opposing_order0_absolute_mass=sum(abs(row['F0']) for row in opposite),
        opposing_order1_absolute_mass=sum(abs(row['F1']) for row in opposite),
        cancellation_from_current_projection_values=describe(cancellation),
        average_distance_comparison=pairs, average_closer_than_both=closer_both,
        average_farther_than_both=farther_both,
        absolute_error_difference={name:describe(values) for name,values in differences.items()},
        scope='Distances are relative to the same native public-output projection Y, not real-world reward or a new acceptance criterion. Unique comparisons use replica-mean F0/F1/average/Y.')


def build_orders_analysis(input_dir, pack_path=PACK, matched_map_path=MATCHED_MAP):
    base = build_fla_analysis(input_dir, pack_path, matched_map_path)
    originals = {_row_key(row):row for row in base['transport_rows']}
    full, reports, rows, payloads = [], [], [], []
    dtype_sets = dict(order_coefficients={}, selected_CPU_coefficients={}, original_do={})

    def dtype(category, name, metadata):
        dtype_sets[category].setdefault(name,set()).add(metadata['dtype'])

    for entry in base['full_finite_FLA_metadata']:
        metadata = entry['metadata']['conditional_orders']
        assert not metadata['diagnostics']
        assert metadata['recorded_original_avg_returns'] == metadata['expected_original_avg_returns']
        full.append(dict(rank=entry['rank'], original_owner_batch_index=entry['original_owner_batch_index'],metadata=metadata))
        payloads.append(dict(rank=entry['rank'],original_owner_batch_index=entry['original_owner_batch_index'],
            retained_cpu_tensor_bytes=metadata['retained_cpu_tensor_bytes']))
        for layer in metadata['layers'].values():
            for group in layer['groups']:
                dtype('original_do','seed',group['original_incoming_do'])
                for order in group['orders']:
                    for name, field in order['fields'].items():
                        dtype('order_coefficients',f"order{order['order']}.{name}",field['original'])
                        for value in field['selected'].values():
                            dtype('selected_CPU_coefficients',f"order{order['order']}.{name}",value)

    for entry in base['actual_single_FLA_reports']:
        report = entry['report']['conditional_orders']
        assert not report['diagnostics']
        assert report['recorded_layer_slot_scalars'] == report['expected_layer_slot_scalars']
        assert report['reused_average_contraction_calls'] == report['expected_reused_average_contraction_calls']
        assert report['added_CPU_order_contractions'] == 2*report['reused_average_contraction_calls']
        reports.append(dict(rank=entry['rank'],original_owner_batch_index=entry['original_owner_batch_index'],
            probe_index=entry['probe_index'],report=report))
        for item in report['layers']:
            key = (entry['rank'],entry['original_owner_batch_index'],item['original_slot'],entry['probe_index'],item['decoder_index'])
            original = originals[key]
            assert item['coordinates']['prefix_cut'] == original['prefix_cut']
            assert item['coordinates']['suffix_length'] == original['native_suffix_length']
            assert item['group_layouts'] == original['actual_head_group_layouts']
            order = {int(part['order']):part for part in item['orders']}
            assert sorted(order) == [0,1]
            f0,f1,average,Y = (float(item[name]) for name in ('F_order0','F_order1','actual_F_average','Y_public_output_projection'))
            mean = (f0+f1)*0.5
            absolute_mean = (abs(f0)+abs(f1))*0.5
            cancellation = absolute_mean-abs(mean)
            values = dict(F0=f0,F1=f1,F_average=average,Y=Y,R0=f0-Y,R1=f1-Y,R_average=average-Y,
                order_mean=mean,order_abs_mean=absolute_mean,order_cancellation=cancellation,
                average_closure_difference=average-mean,
                F0_minus_input_field_sum=f0-sum(order[0]['fields'][name] for name in INPUTS),
                F1_minus_input_field_sum=f1-sum(order[1]['fields'][name] for name in INPUTS),
                recorded_order_mean_minus_recomputed=item['order_projection_mean']-mean,
                recorded_average_closure_minus_recomputed=item['actual_average_closure_difference']-(average-mean),
                recorded_order_abs_mean_minus_recomputed=item['mean_absolute_order_projection']-absolute_mean,
                recorded_order_cancellation_minus_recomputed=item['absolute_projection_cancellation']-cancellation,
                recorded_R0_minus_recomputed=item['F_order0_minus_Y']-(f0-Y),
                recorded_R1_minus_recomputed=item['F_order1_minus_Y']-(f1-Y),
                recorded_R_average_minus_recomputed=item['actual_average_minus_Y']-(average-Y),
                average_minus_existing_F=average-original['F'],Y_minus_existing_Y=Y-original['Y'],
                native_d=original['native_d'])
            assert item['opposing_signs'] == (f0*f1 < 0)
            for order_index in (0,1):
                for name in INPUTS:
                    values[f'order{order_index}.{name}'] = float(order[order_index]['fields'][name])
            row = {name:original[name] for name in (
                'rank','original_owner_batch_index','slot','probe_index','decoder_index',
                'traj_uid','source_step','source_position','input_position','token_id',
                'prefix_cut','native_suffix_length','capture_start','coefficient_suffix_length')}
            row.update(values)
            row['actual_head_group_layouts'] = original['actual_head_group_layouts']
            row['actual_order_head_group_projections'] = item['orders']
            rows.append(row)

    assert len(full) == 14 and len(rows) == 52*len(LAYERS)
    summaries, unique, cuts = [], [], []
    for layer in LAYERS:
        selected = [row for row in rows if row['decoder_index'] == layer]
        pool = reduce_replicas(selected,('traj_uid','source_step','source_position'),FIELDS)
        assert pool['transport_slots'] == 52 and pool['unique_identities'] == 42
        for row in pool['rows']:
            members = [member for member in selected if all(member[name] == row[name]
                for name in ('traj_uid','source_step','source_position'))]
            row['actual_replica_layouts'] = [{name:member[name] for name in (
                'rank','original_owner_batch_index','slot','probe_index','prefix_cut',
                'native_suffix_length','capture_start','coefficient_suffix_length','actual_head_group_layouts')} for member in members]
        unique.append(dict(decoder_index=layer,**pool))
        summaries.append(dict(decoder_index=layer,transport_n=len(selected),unique_n=pool['unique_identities'],
            transport_statistics=statistics(selected),unique_probe_statistics=statistics(pool['rows']),
            transport_order_comparisons=comparisons(selected),
            unique_probe_order_comparisons=comparisons(pool['rows']),maximum_replica_ranges=pool['max_replica_ranges']))
        for start in sorted({row['capture_start'] for row in selected}):
            part = [row for row in selected if row['capture_start'] == start]
            cuts.append(dict(decoder_index=layer,capture_start=start,transport_n=len(part),
                original_group_coordinates=sorted({(row['rank'],row['original_owner_batch_index']) for row in part}),
                transport_statistics=statistics(part),transport_order_comparisons=comparisons(part)))

    return dict(scope=__doc__,status='descriptive_original_order_projection_not_a_rule_change',
        sources=base['sources']+[source(Path(__file__)),source(Path(__file__).with_name('observe_textcraft_fla_orders.py'))],
        checkpoint=base['checkpoint'],calls=base['calls'],coverage=base['coverage'],
        actual_imported_sources=base['actual_imported_sources'],actual_owner_options=base['actual_owner_options'],
        observed_original_FLA_tensor_dtypes=base['observed_tensor_dtypes'],
        observed_order_tensor_dtypes={category:{name:sorted(values) for name,values in fields.items()}
            for category,fields in dtype_sets.items()},
        order_layer_summaries=summaries,capture_start_groups=cuts,
        base_FLA_layer_summaries=base['layer_summaries'],
        full_finite_FLA_metadata=base['full_finite_FLA_metadata'],full_finite_order_metadata=full,
        actual_single_FLA_reports=base['actual_single_FLA_reports'],actual_single_order_reports=reports,
        CPU_original_average_bank_payloads=base['CPU_coefficient_bank_payloads'],
        CPU_order_coefficient_bank_payloads=payloads,
        CPU_order_coefficient_bank_payload_statistics=describe([row['retained_cpu_tensor_bytes'] for row in payloads]),
        transport_rows=rows,unique_probe_replica_means=unique,
        limitations=base['limitations']+[
            'Two native reverse seeds read different endpoint rows but use the same incoming do; the original average and both original calls remain unchanged.',
            'Sign opposition/cancellation is descriptive; it does not show that either order is more accurate or justify changing the accepted rule.',
            'Closer/farther means distance to the same native FP16 public-output projection Y, not quality of the complete token credit or real-world value.',
            'Unique sign/distance comparisons are recomputed from replica-mean projections. Stored order_abs_mean/order_cancellation statistics instead retain the mean of per-transport nonlinear quantities.',
            'Original head/cut/dtype metadata and replica ranges are retained; observed bank bytes are not process RSS or peak GPU memory.',
            'Coincident official-gradient tolerance checks do not establish noncoincident single-token deletion accuracy.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=LOCAL)
    parser.add_argument('--pack',type=Path,default=PACK)
    parser.add_argument('--matched-map',type=Path,default=MATCHED_MAP)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = build_orders_analysis(args.input_dir,args.pack,args.matched_map)
    output = args.output or args.input_dir/'fla-orders-analysis.json'
    output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=source(output),coverage=result['coverage'],
        layers=[dict(decoder_index=layer['decoder_index'],transport_n=layer['transport_n'],unique_n=layer['unique_n'],
            residuals={name:{key:layer['unique_probe_statistics'][name][key] for key in ('abs_mean','rms','positive','negative','zero')}
                for name in ('R0','R1','R_average')},
            comparisons=layer['unique_probe_order_comparisons'],
            average_closure=layer['unique_probe_statistics']['average_closure_difference']) for layer in result['order_layer_summaries']],
        order_dtypes=result['observed_order_tensor_dtypes'],payload=result['CPU_order_coefficient_bank_payload_statistics']),ensure_ascii=False))


if __name__ == '__main__':
    main()
