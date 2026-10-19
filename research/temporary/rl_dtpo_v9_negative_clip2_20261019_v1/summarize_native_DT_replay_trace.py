"""Reduce the original native Torch profiler trace without new device work."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'


def intervals(events):
    result = []
    for start, end in sorted((e['ts'], e['ts'] + e['dur']) for e in events):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def length(ranges):
    return sum(end - start for start, end in ranges)


def intersection(a, b):
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def main():
    receipt = json.loads((AUDIT / 'native-DT-replay-trace-receipt.json').read_bytes())
    assert receipt['status'] == 'completed_native_replay_trace'
    assert receipt['restored_original_methods'] and not receipt['state']['errors']
    path = Path(receipt['trace'])
    trace = json.loads(path.read_bytes())
    events = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and e.get('dur', 0) > 0]
    envelope, = [e for e in events if e.get('cat') == 'Trace']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    copies = [e for e in events if e.get('cat') == 'gpu_memcpy']
    empty = [e for e in events if e.get('cat') == 'cpu_op' and e['name'] == 'aten::empty_strided']
    communication = [e for e in kernels if 'mccl' in e['name'].lower()]
    device_ranges = intervals(kernels + copies)
    empty_ranges = intervals(empty)
    names = collections.Counter()
    for event in kernels:
        names[event['name']] += event['dur']
    result = dict(status='completed_native_replay_trace_reduction', original_trace=str(path),
        original_trace_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(), original_trace_bytes=path.stat().st_size,
        original_profile_receipt=receipt, profile_envelope_seconds=envelope['dur'] / 1e6,
        kernel_union_seconds=length(intervals(kernels)) / 1e6,
        recorded_device_event_union_seconds=length(device_ranges) / 1e6,
        CPU_empty_strided_union_seconds=length(empty_ranges) / 1e6,
        CPU_empty_without_recorded_device_event_seconds=(length(empty_ranges) - intersection(empty_ranges, device_ranges)) / 1e6,
        communication_events=[dict(name=e['name'], start_from_profile_seconds=(e['ts'] - envelope['ts']) / 1e6,
            seconds=e['dur'] / 1e6, input_elements=e['args'].get('In msg nelems'), output_elements=e['args'].get('Out msg nelems'),
            dtype=e['args'].get('dtype'), group_size=e['args'].get('Group size')) for e in communication],
        kernel_sums=[dict(name=name, seconds=duration / 1e6) for name, duration in names.most_common(12)],
        caveats=[
            'A single rank and first FA/GDN pair in one long B4; startup allocations and cross-rank arrival skew are present. This is not a full32-layer profile.',
            'CPU allocation time can include waits; empty_strided alone does not identify which allocator or prove all time is allocation work.',
            'Recorded device interval coverage is not native GPU-Util or a fraction of peak FLOPS. Nested CPU/GPU ranges must not be added.',
            'The first collective3.041s and next1.446ms do not establish a large per-layer communication cost; no prefetch change is deployed.',
        ],
        next_candidate=dict(status='one_same_input_small_B4_owner_option_comparison_pending', option='offload_replay_mixer=False',
            scope='Bounded eligible small input only, original output returned and flags/RNG restored; full long-input production path remains selected.',
            source='check_existing_DT_gpu_resident_pair.py'),
        model_calls=0,DT_calls=0,optimizer_calls=0,arithmetic_changes=0)
    target= AUDIT / 'native-DT-replay-trace-summary.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(receipt=str(target),envelope=result['profile_envelope_seconds'],
        recorded_device_seconds=result['recorded_device_event_union_seconds'],
        CPU_empty_without_device_seconds=result['CPU_empty_without_recorded_device_event_seconds'],
        collective_seconds=[e['seconds'] for e in result['communication_events']])))


if __name__ == '__main__':
    main()
