"""Independent stdlib review of literal GDN diagnostic rows.

No analysis module, Torch, model, remote connection, tolerance or production
formula is imported. Arithmetic below only checks the recorded three-term
observation against its recorded effects and the same-run primitive mixer.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean


DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_PACK = DEFAULT_DIR.parents[1] / 'native-layout-20261006/native64-first-response-matched-layout-pack.json'
LAYERS = (6, 8)
TERMS = ('out_proj', 'norm_gate', 'remaining_input_FLA')
GDN_OWNER_SHA = 'fcbfd9e74a6c40c3bf970d7d24ec42e0f08e5d513ff49eb1ade6a5fcd52e9fca'


def identity(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def maximum_absolute(values):
    return max((abs(value) for value in values), default=None)


def shape_check(actual, expected, errors, label):
    if list(actual) != list(expected):
        errors.append(dict(where=label, actual=list(actual), expected=list(expected)))


def review(input_dir, pack_path):
    input_dir, pack_path = Path(input_dir), Path(pack_path)
    pack = json.loads(pack_path.read_bytes())
    prepared_path = input_dir / 'prepared.json'
    prepared = json.loads(prepared_path.read_bytes())
    paths = [input_dir / f'rank{rank}-readout.json' for rank in range(2)]
    runs = [json.loads(path.read_bytes()) for path in paths]
    errors, diagnostics, sources, full_metadata, rows = [], [], [], [], []
    dtypes = defaultdict(set)
    calls = []
    expected_sources = dict(prepared['sources'])
    expected_sources.update(prepared['diagnostic_dependencies'])
    expected_sources[prepared['diagnostic_source']['path']] = prepared['diagnostic_source']['sha256']

    for rank, run in enumerate(runs):
        if run['rank'] != rank or run['phase'] != 'complete_native_readout':
            errors.append(dict(where='rank_completion', rank=rank, phase=run['phase']))
        if run['input_sha256'] != identity(pack_path)['sha256']:
            errors.append(dict(where='literal_pack_identity', rank=rank))
        if run['checkpoint'] != prepared['checkpoint']:
            errors.append(dict(where='checkpoint', rank=rank))
        calls.append(dict(rank=rank, **{key: run[key] for key in (
            'finite_trace_calls', 'finite_seed_calls', 'native_root_calls',
            'native_prefix_calls', 'optimizer_steps', 'scheduler_steps', 'backward_calls')}))
        for name, spec in run['sources'].items():
            expected = expected_sources.get(spec['path'])
            sources.append(dict(rank=rank, name=name, actual=spec,
                                prepared_sha256=expected,
                                matches_prepared=None if expected is None else expected == spec['sha256']))
            if expected is not None and expected != spec['sha256']:
                errors.append(dict(where='actual_source_binding', rank=rank, name=name))
        full = {case['original_owner_batch_index']: case for case in run['cases']
                if case['variant'] == 'full_response_eos'}
        for group_index, case in full.items():
            group = pack['rank_groups'][rank][group_index]
            slots = [slot for slot, sample in enumerate(group['samples']) if sample['source_step'] == 0]
            metadata = case['actual_coefficient_metadata']['conditional_gdn']
            diagnostics.extend(dict(rank=rank, group=group_index, scope='full', detail=value)
                               for value in metadata['diagnostics'])
            if metadata['captured_layer_indices'] != list(LAYERS):
                errors.append(dict(where='full_GDN_layers', rank=rank, group=group_index))
            if metadata['original_gdn_pullback']['sha256'] != GDN_OWNER_SHA:
                errors.append(dict(where='original_gdn_pullback_identity', rank=rank, group=group_index,
                                   actual=metadata['original_gdn_pullback']))
            if not metadata['actual_compile_gdn_scalar_rules']:
                errors.append(dict(where='actual_compiled_gate_seam', rank=rank, group=group_index))
            native_t = case['selection']['length']
            for layer in LAYERS:
                meta = metadata['layers'][str(layer)]
                diagnostics.extend(dict(rank=rank, group=group_index, layer=layer, scope='full', detail=value)
                                   for value in meta['diagnostics'])
                start = meta['capture_start']
                coeff_t = native_t - start
                if meta['norm_gate_calls'] != 1 or meta['native_suffix_length'] != native_t or meta['coefficient_suffix_length'] != coeff_t:
                    errors.append(dict(where='actual_full_suffix', rank=rank, group=group_index, layer=layer))
                for name, value in meta['coefficients'].items():
                    shape_check(value['original']['shape'], [4, coeff_t, 32, 128], errors,
                                f'rank{rank}/group{group_index}/layer{layer}/{name}')
                    dtypes['coefficient.' + name].add(value['original']['dtype'])
                    for slot in slots:
                        shape_check(value['slots'][str(slot)]['shape'], [1, coeff_t, 4096], errors,
                                    f'rank{rank}/group{group_index}/layer{layer}/{name}/slot{slot}')
                for name in ('o', 'z'):
                    shape_check(meta['actual_native_' + name]['shape'], [8, coeff_t, 32, 128], errors,
                                f'rank{rank}/group{group_index}/layer{layer}/full_{name}')
                    dtypes['full_gate_input.' + name].add(meta['actual_native_' + name]['dtype'])
                full_metadata.append(dict(rank=rank, original_owner_batch_index=group_index,
                    decoder_index=layer, original_slots=slots, native_suffix_length=native_t,
                    capture_start=start, coefficient_suffix_length=coeff_t,
                    norm_gate_calls=meta['norm_gate_calls'], native_o=meta['actual_native_o'],
                    native_z=meta['actual_native_z']))
        for case in run['cases']:
            if case['variant'] == 'full_response_eos':
                continue
            probe = int(case['variant'].rsplit('_', 1)[1])
            group_index = case['original_owner_batch_index']
            group = pack['rank_groups'][rank][group_index]
            expected_slots = [slot for slot, sample in enumerate(group['samples']) if sample['source_step'] == 0]
            primitive = case['conditional_boundary_contractions']['conditional_primitives']
            report = primitive['conditional_gdn']
            diagnostics.extend(dict(rank=rank, group=group_index, probe=probe, scope='single', detail=value)
                               for value in report['diagnostics'])
            if report['score_calls'] != 1 or report['paired_root_capture_entries'] != 1:
                errors.append(dict(where='single_root_call_scope', rank=rank, group=group_index, probe=probe))
            expected_rows = {(layer, slot) for layer in LAYERS for slot in expected_slots}
            actual_rows = {(row['decoder_index'], row['original_slot']) for row in report['layers']}
            if expected_rows != actual_rows or len(report['layers']) != len(expected_rows):
                errors.append(dict(where='single_layer_slot_coverage', rank=rank, group=group_index, probe=probe))
            prim = {(row['decoder_index'], row['original_slot']): row for row in primitive['layers']}
            for layer, meta in report['capture_metadata'].items():
                t = case['selection']['length']
                start = full[group_index]['actual_coefficient_metadata']['conditional_gdn']['layers'][layer]['capture_start']
                if meta['capture_start'] != start or meta['native_suffix_length'] != t:
                    errors.append(dict(where='single_capture_start', rank=rank, group=group_index, layer=layer))
                for name, value in meta['norm_flat_fields'].items():
                    shape_check(value['shape'], [8*t*32, 128], errors,
                                f'rank{rank}/group{group_index}/layer{layer}/flat_{name}')
                    dtypes['single_norm.' + name].add(value['dtype'])
                for name, value in meta['fields'].items():
                    shape_check(value['actual']['shape'], [8, t, 4096], errors,
                                f'rank{rank}/group{group_index}/layer{layer}/{name}')
                    dtypes['single_native.' + name].add(value['actual']['dtype'])
                    for slot in expected_slots:
                        selected_t = t if name in ('input', 'output') else t-start
                        shape_check(value['slots'][str(slot)]['shape'], [2, selected_t, 4096], errors,
                                    f'rank{rank}/group{group_index}/layer{layer}/{name}/slot{slot}')
            for row in report['layers']:
                layer, slot = row['decoder_index'], row['original_slot']
                sample = group['samples'][slot]
                actual_sample = case['samples'][slot]
                for name in ('traj_uid', 'source_step', 'probe_source_positions', 'probe_input_positions'):
                    if sample[name] != actual_sample[name]:
                        errors.append(dict(where='literal_sample_mapping', rank=rank, group=group_index, slot=slot, field=name))
                old = prim[layer, slot]
                effects, terms = row['effects'], row['three_terms']
                recalculated = dict(out_proj=effects['gated_output']-effects['mixer_output'],
                    norm_gate=effects['o_input']+effects['z_input']-effects['gated_output'],
                    remaining_input_FLA=effects['mixer_input']-effects['o_input']-effects['z_input'])
                if row['suffix_length'] != old['suffix_length'] or row['prefix_cut'] != old['prefix_cut']:
                    errors.append(dict(where='original_primitive_layout', rank=rank, group=group_index, slot=slot, layer=layer))
                mixer = effects['mixer_input']-effects['mixer_output']
                rows.append(dict(rank=rank, original_owner_batch_index=group_index, slot=slot,
                    probe_index=probe, decoder_index=layer, traj_uid=sample['traj_uid'],
                    source_step=sample['source_step'], source_position=sample['probe_source_positions'][probe],
                    input_position=sample['probe_input_positions'][probe], prefix_cut=row['prefix_cut'],
                    native_suffix_length=row['suffix_length'], capture_start=row['capture_start'],
                    coefficient_suffix_length=row['coefficient_suffix_length'], terms=terms, effects=effects,
                    term_arithmetic_differences={name: terms[name]-recalculated[name] for name in TERMS},
                    recorded_mixer_difference=row['primitive_mixer_difference'], mixer=mixer,
                    existing_primitive_mixer=old['six_terms']['mixer'],
                    mixer_difference_vs_existing=mixer-old['six_terms']['mixer'],
                    term_sum_minus_mixer=sum(terms.values())-mixer,
                    recorded_closure=row['arithmetic_closure_residual'],
                    native_endpoint_difference=case['target_log_probs'][2*slot+1]-case['target_log_probs'][2*slot]))
    groups = defaultdict(list)
    for row in rows:
        groups[row['decoder_index'], row['traj_uid'], row['source_step'], row['source_position']].append(row)
    summaries, unique_rows = [], []
    for layer in LAYERS:
        selected = [row for row in rows if row['decoder_index'] == layer]
        members = {key: replicas for key, replicas in groups.items() if key[0] == layer}
        for key, replicas in members.items():
            unique_rows.append(dict(decoder_index=layer, traj_uid=key[1], source_step=key[2], source_position=key[3],
                replica_count=len(replicas),
                replica_coordinates=[{name: row[name] for name in ('rank', 'original_owner_batch_index', 'slot', 'probe_index', 'prefix_cut', 'native_suffix_length', 'capture_start')} for row in replicas],
                signed_replica_mean={name: fmean(row['terms'][name] for row in replicas) for name in TERMS},
                term_replica_ranges={name: max(row['terms'][name] for row in replicas)-min(row['terms'][name] for row in replicas) for name in TERMS},
                mixer_replica_mean=fmean(row['mixer'] for row in replicas),
                mixer_replica_range=max(row['mixer'] for row in replicas)-min(row['mixer'] for row in replicas)))
        summaries.append(dict(decoder_index=layer, transport_rows=len(selected), unique_probe_identities=len(members),
            unique_UIDs=len({row['traj_uid'] for row in selected}),
            replica_count_histogram=dict(Counter(len(value) for value in members.values())),
            term_arithmetic_max_abs={name: maximum_absolute(row['term_arithmetic_differences'][name] for row in selected) for name in TERMS},
            three_term_closure_max_abs=maximum_absolute(row['term_sum_minus_mixer'] for row in selected),
            same_run_primitive_mixer_difference_max_abs=maximum_absolute(row['mixer_difference_vs_existing'] for row in selected),
            term_signed_mean={name: fmean(row['terms'][name] for row in selected) for name in TERMS},
            term_abs_mean={name: fmean(abs(row['terms'][name]) for row in selected) for name in TERMS},
            capture_start_counts=dict(Counter(row['capture_start'] for row in selected))))
    return dict(scope=__doc__, status='independent_raw_descriptive_review',
        provenance=[identity(path) for path in [*paths, pack_path, prepared_path, Path(__file__)]],
        sources=sources, source_binding_unmatched_means_not_in_prepared_index=True,
        original_GDN_owner_sha256=GDN_OWNER_SHA, model_modes=[dict(rank=rank, model_dtype=run['model_dtype'],
            native_fla_fp16=run['native_fla_fp16'], event_attention_backend=run['event_attention_backend']) for rank, run in enumerate(runs)],
        calls=calls, structural_errors=errors, actual_observer_diagnostics=diagnostics,
        actual_dtypes={name: sorted(values) for name, values in dtypes.items()},
        coverage=dict(transport_layer_records=len(rows), unique_layer_probe_identities=len(groups),
            transport_probes_per_layer={str(row['decoder_index']): row['transport_rows'] for row in summaries},
            unique_probes_per_layer={str(row['decoder_index']): row['unique_probe_identities'] for row in summaries}),
        layer_summaries=summaries, full_GDN_capture_metadata=full_metadata,
        original_transport_rows=rows, explicit_replica_means=unique_rows,
        limitations=['No numerical pass threshold or FA/FLA tolerance is defined.',
            'Recorded arithmetic closure does not establish world counterfactual accuracy or kernel failure.',
            'm/mo/mz are joint full-span coefficients; native interfaces are actual single-deletion roots.',
            'All original transport rows and duplicate variation remain visible; means are explicitly labelled.',
            'The after-cast native_mo FLA seed is not additionally captured.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--pack', type=Path, default=DEFAULT_PACK)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = review(args.input_dir, args.pack)
    output = args.output or args.input_dir / 'independent-raw-gdn-review.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=identity(output), coverage=result['coverage'],
        structural_errors=result['structural_errors'], actual_observer_diagnostics=result['actual_observer_diagnostics'],
        layers=result['layer_summaries'], dtypes=result['actual_dtypes']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
