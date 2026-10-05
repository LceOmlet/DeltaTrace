"""Read existing Chrome traces to locate memcpy/synchronization CPU APIs.

No Torch import, GPU call, profiler replacement, source patch or timing gate.
CPU runtime durations and correlated device copies remain separate quantities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


NATIVE = re.compile(r'^DT_native_layer_(\d+)_pass_(\d+)$')
PARAMETER = re.compile(r'^DT_native_(prepare|release)_finite_layer_(\d+)$')
APIS = {'mcMemcpyAsync', 'mcStreamSynchronize',
        'cudaMemcpyAsync', 'cudaStreamSynchronize'}


def duration(event):
    return float(event.get('dur', 0))


def external_id(event):
    value = event.get('args', {}).get('External id')
    return None if value is None else str(value)


def correlation(event):
    value = event.get('args', {}).get('correlation')
    return None if value is None else str(value)


class SameThreadIntervals:
    """Sweep original intervals at monotonically ordered runtime timestamps."""
    def __init__(self, events):
        self.regions = sorted(events, key=lambda event: (float(event['ts']), duration(event)))
        self.cursor = 0
        self.active = []

    def at(self, timestamp):
        self.active = [event for event in self.active
                       if float(event['ts']) + duration(event) > timestamp]
        while self.cursor < len(self.regions) and float(self.regions[self.cursor]['ts']) <= timestamp:
            event = self.regions[self.cursor]
            self.cursor += 1
            if float(event['ts']) + duration(event) > timestamp:
                self.active.append(event)
        return sorted(self.active, key=lambda event: (duration(event), -float(event['ts'])))


def union_seconds(events):
    intervals = sorted((float(event['ts']), float(event['ts']) + duration(event))
                       for event in events)
    result, end = 0.0, None
    for start, stop in intervals:
        if end is None or start > end:
            result += stop - start
        elif stop > end:
            result += stop - end
        end = stop if end is None else max(end, stop)
    return result / 1e6


def copied_direction(event):
    name = event.get('name', '')
    return next((kind for kind in ('HtoD', 'DtoH', 'DtoD') if kind in name), 'other_memcpy')


def analyze(path, top=20):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    events = data.get('traceEvents', []) if isinstance(data, dict) else data
    cpu_regions, annotation_regions = defaultdict(list), defaultdict(list)
    cpu_by_external = defaultdict(list)
    runtime_events, gpu_copies = [], []
    runtime_by_correlation, runtime_copy_by_external = defaultdict(list), defaultdict(list)
    gpu_by_correlation, gpu_by_external = defaultdict(list), defaultdict(list)
    for event in events:
        if event.get('ph') != 'X':
            continue
        category = event.get('cat', '')
        thread = (event.get('pid'), event.get('tid'))
        if category == 'cpu_op':
            cpu_regions[thread].append(event)
            if external_id(event) is not None:
                cpu_by_external[external_id(event)].append(event)
        if category == 'user_annotation' or event.get('name', '').startswith('FSDP::'):
            annotation_regions[thread].append(event)
        if category == 'gpu_memcpy':
            gpu_copies.append(event)
            if correlation(event) is not None:
                gpu_by_correlation[correlation(event)].append(event)
            if external_id(event) is not None:
                gpu_by_external[external_id(event)].append(event)
        if event.get('name') in APIS and category not in ('kernel', 'gpu_memcpy', 'cpu_op', 'user_annotation'):
            runtime_events.append(event)
            if correlation(event) is not None:
                runtime_by_correlation[correlation(event)].append(event)
            if 'Memcpy' in event['name'] and external_id(event) is not None:
                runtime_copy_by_external[external_id(event)].append(event)
    cpu_indices = {thread: SameThreadIntervals(regions) for thread, regions in cpu_regions.items()}
    annotation_indices = {thread: SameThreadIntervals(regions)
                          for thread, regions in annotation_regions.items()}
    links, tables, details = Counter(), {}, []
    for key in ('runtime_by_scope', 'runtime_by_torch_stack'):
        tables[key] = defaultdict(lambda: [0, 0.0, 0, 0, 0, 0.0])
    runtime_totals, by_thread = defaultdict(list), defaultdict(list)

    for event in sorted(runtime_events, key=lambda value: float(value['ts'])):
        timestamp = float(event['ts'])
        thread = (event.get('pid'), event.get('tid'))
        ops = cpu_indices[thread].at(timestamp) if thread in cpu_indices else []
        annotations = annotation_indices[thread].at(timestamp) if thread in annotation_indices else []
        op = ops[0] if ops else None
        native = next((value for value in annotations if NATIVE.fullmatch(value.get('name', ''))), None)
        parameter = next((value for value in annotations if PARAMETER.fullmatch(value.get('name', ''))), None)
        fsdp = next((value for value in annotations if value.get('name', '').startswith('FSDP::')), None)
        cpu_candidates = cpu_by_external.get(external_id(event), [])
        external_cpu = cpu_candidates[0] if len(cpu_candidates) == 1 else None
        if op is None:
            links['no_same_thread_cpu_op'] += 1
        else:
            links['same_thread_cpu_op'] += 1
            links['runtime_end_inside_innermost_cpu_op' if timestamp + duration(event) <=
                  float(op['ts']) + duration(op) else 'runtime_end_outside_innermost_cpu_op'] += 1
        links['external_cpu_unique' if external_cpu is not None else
              'external_cpu_ambiguous' if cpu_candidates else 'external_cpu_missing'] += 1
        if external_cpu is not None and op is not None:
            links['external_cpu_is_innermost' if external_cpu is op else 'external_cpu_differs_from_innermost'] += 1
        copies, copy_link = [], 'not_a_memcpy_API'
        if 'Memcpy' in event['name']:
            corr = correlation(event)
            if corr is not None and len(runtime_by_correlation[corr]) == 1 and gpu_by_correlation[corr]:
                copies, copy_link = gpu_by_correlation[corr], 'unique_runtime_correlation'
            elif (external_id(event) is not None and
                  len(runtime_copy_by_external[external_id(event)]) == 1 and
                  len(gpu_by_external[external_id(event)]) == 1):
                copies, copy_link = gpu_by_external[external_id(event)], 'one_to_one_external_id'
            else:
                copy_link = 'no_unique_device_copy_link'
            links[copy_link] += 1
        directions = sorted({copied_direction(value) for value in copies})
        copy_scope = '+'.join(directions) if directions else 'no_correlated_device_copy'
        scope = (event['name'], op.get('name') if op else 'outside_cpu_op',
                 fsdp.get('name') if fsdp else 'outside_fsdp_annotation',
                 native.get('name') if native else 'outside_native_layer',
                 parameter.get('name') if parameter else 'outside_parameter_range', copy_scope)
        stack = (event['name'], tuple(value.get('name', '') for value in ops),
                 tuple(value.get('name', '') for value in annotations), copy_scope)
        for table, key in (('runtime_by_scope', scope), ('runtime_by_torch_stack', stack)):
            values = tables[table][key]
            values[0] += 1
            values[1] += duration(event)
            values[2] += len(copies)
            values[3] += sum(int(value.get('args', {}).get('bytes', 0)) for value in copies)
            values[4] += sum('bytes' not in value.get('args', {}) for value in copies)
            values[5] += sum(duration(value) for value in copies)
        runtime_totals[(event['name'], event.get('cat', ''))].append(event)
        by_thread[thread].append(event)
        details.append(dict(runtime_event=event,
            same_thread_cpu_op_stack=[dict(name=value.get('name'), ts=value['ts'],
                dur=duration(value), external_id=external_id(value)) for value in ops],
            same_thread_annotation_stack=[dict(name=value.get('name'), ts=value['ts'],
                dur=duration(value)) for value in annotations],
            unique_external_cpu_op=(None if external_cpu is None else dict(
                name=external_cpu.get('name'), pid=external_cpu.get('pid'), tid=external_cpu.get('tid'),
                ts=external_cpu['ts'], dur=duration(external_cpu))),
            device_copy_link_rule=copy_link, correlated_device_copies=copies))
    result = dict(source_path=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_trace_source=data.get('source') if isinstance(data, dict) else None,
        trace_event_count=len(events), analyzed_runtime_apis=sorted(APIS),
        runtime_event_count=len(runtime_events), observed_gpu_copy_count=len(gpu_copies),
        scope='Existing CPU runtime intervals only. CPU inclusive duration, device copy duration and other CPU op inclusive duration overlap; never add them as elapsed time or call every synchronization a specific GPU wait.',
        cpu_owner_rule='At runtime CPU start, original same-pid/tid innermost cpu_op and annotation intervals; full actual ancestor chains retained. External-id owner is reported separately; missing/ambiguous or differing links are not silently replaced.',
        copy_link_rule='Unique runtime correlation to original gpu_memcpy events, otherwise exact one-to-one memcpy API/device External id; no size, direction or waiting-work guesses.',
        link_counts=dict(links),
        runtime_totals=[dict(api=api, category=category, count=len(values),
            cpu_inclusive_seconds=sum(duration(value) for value in values)/1e6)
            for (api, category), values in sorted(runtime_totals.items())],
        runtime_union_by_thread=[dict(pid=thread[0], tid=thread[1],
            cpu_runtime_union_seconds=union_seconds(values)) for thread, values in by_thread.items()],
        top_runtime_events={api: sorted((value for value in details if value['runtime_event']['name'] == api),
            key=lambda value: -duration(value['runtime_event']))[:top] for api in sorted(APIS)})
    for key, table in tables.items():
        result[key] = sorted([dict(group=list(group), count=values[0],
            cpu_inclusive_seconds=values[1]/1e6, correlated_device_copy_count=values[2],
            correlated_device_copy_bytes=values[3], device_copies_without_bytes=values[4],
            correlated_device_copy_seconds=values[5]/1e6) for group, values in table.items()],
            key=lambda value: -value['cpu_inclusive_seconds'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    encoded = json.dumps(dict(traces=[analyze(path, args.top) for path in args.traces]),
                         ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + '\n', encoding='utf8')
        print(json.dumps(dict(output=str(args.output.resolve()), traces=len(args.traces))))
    else:
        print(encoded)


if __name__ == '__main__':
    main()
