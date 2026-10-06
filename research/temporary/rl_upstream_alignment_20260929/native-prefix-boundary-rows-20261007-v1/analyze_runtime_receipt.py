"""Read saved row-storage diagnostic JSON only; no numerical acceptance policy."""
import argparse
import hashlib
import json
import math
from pathlib import Path


VARIANTS = ('shared_cold', 'shared_warm',
            'shared_boundary_rows_cold', 'shared_boundary_rows_warm')


def source(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def difference(before, after):
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    return dict(OFF=before, ON=after, removed=before-after,
                removed_fraction=(before-after)/before if before else None)


def compact_phase(value):
    return {key: value[key] for key in (
        'rank', 'phase', 'variant', 'observed_unix', 'total_wall_seconds',
        'shared_preparation', 'shared_bank_already_prepared') if key in value}


def read_phase_events(path, errors):
    """Read existing structured events or the owner's JSON stdout lines."""
    events = []
    if path.suffix == '.json':
        try:
            value = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(dict(path=str(path), error=str(error)))
            return events
        if isinstance(value, dict):
            value = next((value[key] for key in ('phases', 'phase_events', 'events')
                          if isinstance(value.get(key), list)), [])
        if isinstance(value, list):
            events = [compact_phase(row) for row in value
                      if isinstance(row, dict) and 'phase' in row]
        return events
    with path.open('r', encoding='utf8', errors='replace') as handle:
        for line_number, line in enumerate(handle, 1):
            if '"phase"' not in line:
                continue
            start = line.find('{')
            if start < 0:
                continue
            try:
                value = json.loads(line[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and 'phase' in value:
                events.append(dict(line=line_number, **compact_phase(value)))
    return events


def row_check_summary(inventory):
    checks = inventory.get('same_capture_row_checks', [])
    kinds = {}
    for kind in sorted({check.get('kind', 'missing') for check in checks}):
        rows = [check for check in checks if check.get('kind', 'missing') == kind]
        kinds[kind] = dict(
            export_records=len(rows),
            original_rows=sum(len(row.get('original_rows', [])) for row in rows),
            recorded_equal_true=sum(row.get('equal') is True for row in rows),
            recorded_equal_false=sum(row.get('equal') is False for row in rows),
            recorded_equal_missing=sum('equal' not in row for row in rows),
            logical_element_bytes_compared=sum(row.get('logical_element_bytes_compared', 0)
                                               for row in rows),
            observer_seconds=math.fsum(row.get('observer_seconds', 0) for row in rows),
            dtypes=sorted({str(row.get('dtype')) for row in rows}),
            layers=sorted({row.get('layer_index') for row in rows
                           if isinstance(row.get('layer_index'), int)}))
    return dict(raw_records=len(checks), by_kind=kinds,
                owner_summary=inventory.get('same_capture_row_check_summary'),
                scope='Recorded exact logical element bytes at the same capture export; '
                      'does not establish whole-DT numerical correctness or a floating tolerance.')


def inventory_summary(inventory):
    return dict(
        variant=inventory.get('variant'),
        original_request_path=inventory.get('original_request_path'),
        original_request_sha256=inventory.get('original_request_sha256'),
        original_requests=inventory.get('original_requests'),
        original_consumer_batches=inventory.get('original_consumer_batches'),
        source_slots=inventory.get('source_slots'),
        distinct_artifacts=inventory.get('distinct_artifacts'),
        distinct_consumed_artifact_boundary_rows=inventory.get(
            'distinct_consumed_artifact_boundary_rows'),
        totals=inventory.get('totals'),
        lease_source_checks=inventory.get('lease_source_checks'),
        original_preparation=inventory.get('original_preparation'),
        host_before_prepare={key: inventory.get('host_before_prepare', {}).get(key)
                             for key in ('pss_bytes', 'scope')},
        host_after_prepare={key: inventory.get('host_after_prepare', {}).get(key)
                            for key in ('pss_bytes', 'scope')},
        same_capture=row_check_summary(inventory))


def consumption_signature(inventory):
    return sorted((artifact['artifact_index'], row['prefix_length'], row['row'])
                  for artifact in inventory.get('artifacts', [])
                  for row in artifact.get('consumed_boundary_rows', []))


def build_summary(directory, phase_paths=()):
    inputs, errors = {}, []

    def read(name):
        path = directory / name
        if not path.exists():
            return None
        inputs[name] = source(path)
        try:
            return json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(dict(path=str(path), error=str(error)))
            return None

    prepared, job = read('prepared.json'), read('job.json')
    result, completed = read('result.json'), read('completed.json')
    events = []
    candidates = list(phase_paths) or [directory / name for name in (
        'phase-events.json', 'phase-records.json', 'phases.json', 'probe.log')]
    for path in candidates:
        if path.exists():
            inputs[str(path)] = source(path)
            events.extend(read_phase_events(path, errors))
    ranks = []
    for rank in (0, 1):
        value = read(f'rank{rank}.json') or {}
        variants = {}
        for label, report in value.get('reports', {}).items():
            if label not in VARIANTS:
                continue
            variants[label] = {key: report.get(key) for key in (
                'total_wall_seconds', 'original_attribute_wall_seconds',
                'original_runner_phase_seconds', 'original_runner_phase_counts',
                'peak_torch_allocated_bytes', 'physical_free_bytes', 'pss_bytes')}
            variants[label]['original_readout_summary'] = {key: report.get(
                'original_readout_report', {}).get(key) for key in (
                    'seconds', 'finite_trace_calls', 'event_contrasts', 'shared_native_prefix')}
        original = read(f'prefix-bank-inventory-shared_cold-rank{rank}.json')
        compact = read(f'prefix-bank-inventory-shared_boundary_rows_cold-rank{rank}.json')
        inventory = {}
        for label, data in (('OFF', original), ('ON', compact)):
            if data is not None:
                inventory[label] = inventory_summary(data)
        comparison = None
        if original is not None and compact is not None:
            before, after = original.get('totals', {}), compact.get('totals', {})
            comparison = dict(
                request_sha256_same=original.get('original_request_sha256') == compact.get(
                    'original_request_sha256'),
                consumed_artifact_boundary_original_rows_same=(
                    consumption_signature(original) == consumption_signature(compact)),
                storage={key: difference(before.get(key), after.get(key)) for key in (
                    'logical_tensor_bytes', 'distinct_live_storage_bytes', 'gdn_conv_bytes',
                    'gdn_state_bytes', 'gdn_unconsumed_conv_bytes', 'gdn_unconsumed_state_bytes',
                    'gdn_stored_boundary_rows', 'gdn_unconsumed_boundary_rows',
                    'fa_keys_bytes', 'fa_values_bytes', 'input_ids_bytes')},
                capture_preparation_seconds=difference(
                    original.get('original_preparation', {}).get('capture_and_preparation_seconds'),
                    compact.get('original_preparation', {}).get('capture_and_preparation_seconds')),
                scope='Storage counts for the full originally constructed88-request bank; '
                      'both variants select one original B4 afterwards. Distinct storage bytes '
                      'are artifact inventory, not process RSS or pinned allocator counters.')
        rank_events = [event for event in events if event.get('rank') == rank]
        event_completions = [event for event in rank_events
                             if event.get('phase') == 'native_prefix_lease_variant_complete']
        for event in event_completions:
            label = event.get('variant')
            if label in VARIANTS and label not in variants:
                variants[label] = dict(total_wall_seconds=event.get('total_wall_seconds'),
                                       original_readout_summary=dict(shared_native_prefix=
                                           event.get('shared_preparation')),
                                       scope='Only original phase-completion event available; '
                                             'full owner phase report not yet saved.')
        warm_difference = difference(variants.get('shared_warm', {}).get('total_wall_seconds'),
                                     variants.get('shared_boundary_rows_warm', {}).get('total_wall_seconds'))
        ranks.append(dict(rank=rank, phase=value.get('phase'), observed_unix=value.get('observed_unix'),
            imported_sources=value.get('imported_sources'), actual_owner_sources=value.get('sources'),
            native_conv_initial_states=value.get('native_conv_initial_states'),
            variants=variants, warm_total_wall_difference=warm_difference,
            inventories=inventory, OFF_ON_inventory_comparison=comparison,
            raw_value_observations=value.get('raw_value_observations'),
            original_phase_events=rank_events,
            missing_variants=[label for label in VARIANTS if label not in variants]))
    candidate = (prepared or {}).get('boundary_row_storage_candidate')
    return dict(
        status=('rank_diagnostics_complete' if all(rank['phase'] ==
                'native_prefix_lease_diagnostic_complete' for rank in ranks) else 'incomplete'),
        scope='Read existing JSON/log receipts only. No torch, tensor artifact load, '
              'model, environment, forward, DT, backward, optimizer, or remote action.',
        analyzer=source(Path(__file__)), directory=str(directory.resolve()), inputs=inputs,
        prepared_candidate=candidate,
        prepared_scope={key: (prepared or {}).get(key) for key in (
            'role', 'devices', 'rows_per_rank', 'current_formal_source', 'original_scalar_ledger_only')},
        job={key: (job or {}).get(key) for key in (
            'pid', 'birth', 'pid_birth', 'started_unix', 'start_time', 'devices', 'log')},
        completed_marker=completed,
        result_marker=(None if result is None else {key: result.get(key) for key in (
            'completed_unix', 'returncode', 'status')}) if isinstance(result, dict) else None,
        ranks=ranks, read_errors=errors,
        interpretation_limits=[
            'Same-capture byte comparisons validate declared exported rows only; '
            'raw Q/V/A differences are descriptive and have no new acceptance threshold.',
            'ON cold preparation includes recorded byte-check observer time. '
            'Observer seconds are a measured subset, not a separate wall-time saving; '
            'warm attribute timing excludes new bank capture.',
            'Cold OFF/ON captures are distinct native forwards; numerical variation '
            'between captures is not silently attributed to storage selection.',
            'Per-rank walls run in parallel and are not summed as whole job wall time.',
            'Artifact storage bytes do not explain all PSS/cgroup memory or prove a leak.',
            'Rank completion alone does not prove driver exit, deployment,32k capacity '
            'or improved training effectiveness.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--phases', type=Path, action='append', default=[])
    args = parser.parse_args()
    output = args.output or args.directory / 'runtime-summary.json'
    summary = build_summary(args.directory, args.phases)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(path=str(output.resolve()), sha256=source(output)['sha256'],
                         status=summary['status'], ranks=len(summary['ranks']))))


if __name__ == '__main__':
    main()
