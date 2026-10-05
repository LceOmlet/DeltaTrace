"""Independent stdlib review of saved original-owner FLA scalar observations.

Reads the two complete raw rank files and the fixed matched input pack (or
its identity/geometry summary). It does not import the observation/analyzer
modules, model, Torch, Ray or any remote connector. Arithmetic residuals are
reported descriptively, without a numerical acceptance threshold.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEG = HERE.parents[1]
DEFAULT_PACK = DEG / 'native-minibatch-v4/native64-first-response-matched-layout-summary.json'
TERMS = ('input_chain', 'FLA_projection', 'native_cast_bridge')
FIELDS = ('q', 'k', 'v', 'beta', 'g', 'o')
METRICS = TERMS + ('F', 'Y', 'old_remaining', 'cancellation', 'closure',
    'F_field_sum_difference', 'Y_field_difference', 'input_chain_formula_difference',
    'FLA_formula_difference', 'cast_formula_difference', 'same_run_GDN_remaining_difference')


def source(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def describe(values):
    values = list(values)
    if not values:
        return dict(count=0)
    return dict(count=len(values), mean=math.fsum(values)/len(values),
        mean_abs=math.fsum(abs(v) for v in values)/len(values),
        rms=math.sqrt(math.fsum(v*v for v in values)/len(values)),
        max_abs=max(abs(v) for v in values), minimum=min(values), maximum=max(values),
        positive=sum(v > 0 for v in values), negative=sum(v < 0 for v in values),
        zero=sum(v == 0 for v in values), nonfinite=sum(not math.isfinite(v) for v in values))


def stats(rows):
    return {name: describe(row[name] for row in rows) for name in METRICS}


def identity(sample, probe):
    return dict(traj_uid=sample['traj_uid'], source_step=int(sample['source_step']),
        source_position=int(sample['probe_source_positions'][probe]),
        input_position=int(sample['probe_input_positions'][probe]),
        original_request_index=int(sample['original_request_index']))


def verify_sample(actual, expected):
    for name in ('traj_uid', 'source_step', 'original_request_index', 'source_start', 'source_end',
                 'context_tokens', 'compute_tokens', 'probe_source_positions', 'probe_input_positions'):
        if actual[name] != expected[name]:
            raise ValueError('Raw sample differs from fixed input pack: '+name)


def build(directory, pack_path):
    pack = read(pack_path)
    full_pack_identity = pack.get('original_full_pack', source(pack_path))
    expected_input_sha = full_pack_identity['sha256']
    raw = []
    diagnostics = []
    layouts = []
    native_dtypes = defaultdict(set)
    callback_dtypes = defaultdict(set)
    rank_sources = []
    original_sources = []
    rank_counters = []
    observation_counts = []
    input_matches = []
    for rank in (0, 1):
        path = directory / f'rank{rank}-readout.json'
        data = read(path)
        rank_sources.append(source(path))
        original_sources.append(dict(rank=rank, sources=data['sources'],
            input_sources=data['input_sources'], input_path=data['input_path'], input_sha256=data['input_sha256']))
        if int(data['rank']) != rank or len(data['cases']) != 21 or data['phase'] != 'complete_native_readout':
            raise ValueError('Require both original complete 21-case rank outputs.')
        if data['input_sha256'] != expected_input_sha:
            raise ValueError('Raw rank output does not bind the fixed matched input pack.')
        rank_counters.append(dict(rank=rank, checkpoint=data['checkpoint'],
            **{name:data[name] for name in ('optimizer_steps', 'scheduler_steps', 'backward_calls',
                'finite_trace_calls', 'finite_seed_calls', 'paired_root_entries',
                'native_forward_calls', 'native_root_calls', 'native_prefix_calls')}))
        groups = {int(group['original_owner_batch_index']):group for group in pack['rank_groups'][rank]}
        full_cases = {int(case['original_owner_batch_index']):case for case in data['cases']
                      if case['variant'] == 'full_response_eos'}
        if sorted(full_cases) != list(range(7)):
            raise ValueError('Original seven full-EOS groups are missing or duplicated.')
        for group_index, case in sorted(full_cases.items()):
            group = groups[group_index]
            for actual, expected in zip(case['samples'], group['samples'], strict=True):
                verify_sample(actual, expected)
            metadata = case['actual_coefficient_metadata']['conditional_fla']
            diagnostics.append(dict(rank=rank, group=group_index, variant=case['variant'],
                kind='full_FLA', diagnostics=metadata['diagnostics']))
            for layer, layer_meta in metadata['layers'].items():
                for head_index, head in enumerate(layer_meta['groups']):
                    layout = head['layout']
                    shape = head['incoming_do']['shape']
                    native_start = int(head['native_time_start'])
                    compact_t = int(head['coordinates']['suffix_length'])-native_start
                    if native_start != int(layout['capture_start'])+int(layout['cut']) or shape[1] != compact_t:
                        raise ValueError('Actual callback time coordinates do not align.')
                    layouts.append(dict(rank=rank, group=group_index, layer=int(layer), head_group=head_index,
                        layout=layout, native_time_start=native_start, incoming_do=head['incoming_do'],
                        head_count=head['head_count'], head_width=head['head_width'],
                        coordinates=head['coordinates'], incoming_state=head['incoming_state'],
                        incoming_state_pair_absmax=head['incoming_state_pair_absmax'],
                        actual_scale=head['actual_scale']))
                    callback_dtypes['do'].add(head['incoming_do']['dtype'])
                    callback_dtypes['incoming_state'].add(head['incoming_state']['dtype'])
                    for name, field in head['fields'].items():
                        callback_dtypes[name+'_coefficient'].add(field['coefficient']['dtype'])
                        callback_dtypes[name+'_endpoint'].add(field['endpoint']['dtype'])
        for case_index, case in enumerate(data['cases']):
            if case['variant'] == 'full_response_eos':
                continue
            probe = {'single_eos_0':0, 'single_eos_1':1}[case['variant']]
            group_index = int(case['original_owner_batch_index'])
            group, full = groups[group_index], full_cases[group_index]
            if case['fixed_cut'] != full['observed_prefix_length'] or case['observed_prefix_length'] != full['observed_prefix_length']:
                raise ValueError('Single-EOS cut differs from the same original full-EOS layout.')
            for actual, expected in zip(case['samples'], group['samples'], strict=True):
                verify_sample(actual, expected)
            boundary = case['conditional_boundary_contractions']
            primitive = boundary['conditional_primitives']
            gdn = primitive['conditional_gdn']
            fla = gdn['conditional_fla']
            for name, report in (('boundaries',boundary), ('primitives',primitive), ('GDN',gdn), ('FLA',fla)):
                diagnostics.append(dict(rank=rank, group=group_index, variant=case['variant'], kind=name,
                    diagnostics=report['diagnostics']))
            observation_counts.append(dict(rank=rank, group=group_index, variant=case['variant'],
                expected=fla['expected_layer_slot_scalars'], recorded=fla['recorded_layer_slot_scalars'],
                paired_root_entries=fla['paired_root_capture_entries'], score_calls=fla['score_calls']))
            gdn_rows = {(int(row['decoder_index']),int(row['original_slot'])):row for row in gdn['layers']}
            expected_slots = [i for i,sample in enumerate(group['samples']) if int(sample['source_step']) == 0]
            if sorted(case['changed_first_response_slots']) != expected_slots:
                raise ValueError('Single-EOS changed slots differ from original first-response slots.')
            if len(fla['layers']) != 2*len(expected_slots):
                raise ValueError('Raw FLA rows do not cover original selected slots at two layers.')
            for layer, capture in fla['capture_metadata'].items():
                for name, field in capture['fields'].items():
                    native_dtypes[name].add(field['dtype'])
                if set(capture['fields']) != set(FIELDS):
                    raise ValueError('Native original FLA field observations are incomplete.')
                input_matches.append(dict(rank=rank, group=group_index, probe=probe, layer=int(layer),
                    fields=capture['fields'], stage_returns=capture['stage_returns'],
                    public_returns=capture['public_returns'], actual_stage_scale=capture['actual_stage_scale'],
                    actual_public_scale_argument=capture['actual_public_scale_argument'],
                    actual_qk_normalization=capture['actual_qk_normalization'],
                    actual_public_initial_state=capture['actual_public_initial_state']))
            for row in fla['layers']:
                layer, slot = int(row['decoder_index']), int(row['original_slot'])
                if slot not in expected_slots or row['paired_rows'] != [2*slot,2*slot+1]:
                    raise ValueError('Raw selected paired rows differ from fixed pack slot identity.')
                sample = group['samples'][slot]
                expected_position = int(sample['source_start'])+int(sample['probe_source_positions'][probe])
                if expected_position != int(sample['probe_input_positions'][probe]):
                    raise ValueError('Fixed source-relative and absolute probe positions differ.')
                previous = gdn_rows[layer, slot]
                effects = row['original_GDN_effects']
                F, Y = float(row['F_input_projection']), float(row['Y_public_output_projection'])
                terms = {name:float(row['remaining_three_terms'][name]) for name in TERMS}
                old = float(previous['three_terms']['remaining_input_FLA'])
                f_sum = math.fsum(float(row['fields'][name]) for name in FIELDS[:-1])
                native_lp = float(case['target_log_probs'][2*slot+1])-float(case['target_log_probs'][2*slot])
                record = dict(rank=rank, group=group_index, case_index=case_index, probe=probe,
                    original_slot=slot, layer=layer, **identity(sample, probe),
                    context_tokens=sample['context_tokens'], observed_return=sample['observed_return'],
                    F=F, Y=Y, old_remaining=old, **terms,
                    cancellation=math.fsum(abs(v) for v in terms.values())-abs(math.fsum(terms.values())),
                    closure=math.fsum(terms.values())-old,
                    F_field_sum_difference=F-f_sum, Y_field_difference=Y-float(row['fields']['o']),
                    input_chain_formula_difference=terms['input_chain']-(effects['mixer_input']-effects['z_input']-F),
                    FLA_formula_difference=terms['FLA_projection']-(F-Y),
                    cast_formula_difference=terms['native_cast_bridge']-(Y-effects['o_input']),
                    same_run_GDN_remaining_difference=float(row['original_GDN_remaining'])-old,
                    original_GDN_effects=effects, native_fields=row['fields'], native_single_logprob_difference=native_lp,
                    group_layouts=row['group_layouts'], coordinates=row['coordinates'])
                raw.append(record)
    replicas = defaultdict(list)
    for row in raw:
        replicas[row['layer'],row['traj_uid'],row['source_step'],row['source_position']].append(row)
    unique = []
    for key, rows in sorted(replicas.items()):
        layer, uid, step, position = key
        unique.append(dict(layer=layer, traj_uid=uid, source_step=step, source_position=position,
            replica_count=len(rows), transport_rows=[dict(rank=r['rank'],group=r['group'],slot=r['original_slot'],probe=r['probe']) for r in rows],
            **{name:math.fsum(r[name] for r in rows)/len(rows) for name in METRICS},
            replica_spread={name:dict(minimum=min(r[name] for r in rows),maximum=max(r[name] for r in rows),
                range=max(r[name] for r in rows)-min(r[name] for r in rows)) for name in METRICS}))
    summaries = {}
    for layer in (6,8):
        layer_raw = [r for r in raw if r['layer'] == layer]
        layer_unique = [r for r in unique if r['layer'] == layer]
        summaries[str(layer)] = dict(raw_transport_count=len(layer_raw), unique_probe_count=len(layer_unique),
            unique_trajectory_count=len({r['traj_uid'] for r in layer_unique}),
            raw_stats=stats(layer_raw), replica_mean_stats=stats(layer_unique),
            duplicate_spread={name:describe(r['replica_spread'][name]['range'] for r in layer_unique) for name in METRICS})
    return dict(scope=__doc__, input_pack_summary=source(pack_path), original_full_pack=full_pack_identity,
        raw_rank_sources=rank_sources, independent_script=source(Path(__file__)),
        original_owner_sources=original_sources, original_rank_counters=rank_counters,
        coverage=dict(raw_layer_rows=len(raw),unique_layer_rows=len(unique),
            per_layer=summaries, fixed_pack_coverage=pack.get('coverage')),
        observer_diagnostics=dict(report_count=len(diagnostics),
            item_count=sum(len(row['diagnostics']) for row in diagnostics), reports=diagnostics),
        observation_counts=observation_counts, native_field_dtypes={k:sorted(v) for k,v in native_dtypes.items()},
        callback_field_dtypes={k:sorted(v) for k,v in callback_dtypes.items()},
        actual_callback_layouts=layouts, native_input_observations=input_matches,
        raw_transport_rows=raw, unique_replica_mean_rows=unique,
        limitations=[
            '42 unique probes are 21 successful first responses at checkpoint25; they are not independent trajectories or the whole training distribution.',
            'Duplicate transport contexts are retained; the unique summary averages them and separately reports their spread.',
            'FLA_projection includes joint finite conditional projection and native storage; it does not establish an official FLA kernel error.',
            'native_cast_bridge combines actual do and public-output dtype boundaries; it is not a correction.',
            'Arithmetic closure is descriptive and is not an additional model or whole-DT tolerance.',
            'Source mapping uses the original matched pack identity summary when literal input arrays are not present locally.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=HERE)
    parser.add_argument('--input-pack', type=Path, default=DEFAULT_PACK)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.directory / 'independent-raw-fla-review.json'
    if output.exists():
        raise FileExistsError('Preserve existing review: '+str(output))
    result = build(args.directory, args.input_pack)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output.resolve()), sources=result['raw_rank_sources'],
        observer_diagnostic_items=result['observer_diagnostics']['item_count'],
        native_dtypes=result['native_field_dtypes'],
        layers={layer:dict(raw=summary['raw_transport_count'],unique=summary['unique_probe_count'],
            terms={name:summary['replica_mean_stats'][name] for name in TERMS},
            closure_max=summary['raw_stats']['closure']['max_abs'])
            for layer,summary in result['coverage']['per_layer'].items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
