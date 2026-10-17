"""Join native dmon rows to the existing process-phase observations.

This reads physical CSV and existing owner observations only. Training metrics
continue to come from the original collect/read_job_logs/parse_logs entry.
"""
import argparse
import bisect
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'
HOST = ROOT / 'receipts/appworld-formal-training-20261017-v1/native-host-RSS-PSS-XTT-timeseries-v1.jsonl'
PHYSICAL = AUDIT / 'native-mx-smi-dmon-timestamped.jsonl'


def summary(values):
    return dict(samples=len(values), mean=statistics.mean(values),
        median=statistics.median(values), minimum=min(values), maximum=max(values))


def phase(row):
    jobs = [j for j in row['jobs'] if j['alive'] and j.get('version') and
        '-8gpu-200-20261017-v1' in j['version']]
    if len(jobs) != 1:
        return ('unresolved_job',)
    job, = jobs
    workers = [p for p in job['processes'] if 'ray::WorkerDict.' in p['phase']]
    phases = {p['phase'] for p in workers}
    if len(workers) != 8 or len(phases) != 1:
        return ('partial_or_mixed_worker_phase', job['version'])
    return (job['task'], job['method'], job['version'], phases.pop())


def main(window='all'):
    lower, upper = None, None
    if window == 'before_allocator_pair':
        lower = 0
        upper = min(json.loads((AUDIT / f'one-natural-DT-native-allocator-pair-v2-rank{rank}.json').read_bytes())['started']
                    for rank in range(8))
    elif window == 'after_native_configuration':
        lower = json.loads((AUDIT / 'native-pinned-host-register-20261017-v1-completed.json').read_bytes())['collected']
    rows = []
    with PHYSICAL.open() as stream:
        for line in stream:
            value = json.loads(line)
            if lower is not None and value['received_unix'] < lower:
                continue
            if upper is not None and value['received_unix'] >= upper:
                continue
            fields = next(csv.reader([value['native_csv_line']], skipinitialspace=True))
            if len(fields) != 9:
                continue
            try:
                numbers = list(map(float, fields))
            except ValueError:
                continue
            rows.append((value['received_unix'], int(numbers[0]), numbers[2], numbers[8]))
    assert rows, 'No native physical rows; missing measurements are not zero'
    first, last = rows[0][0], rows[-1][0]
    host = []
    with HOST.open() as stream:
        for line in stream:
            value = json.loads(line)
            if first-20 <= value['collected'] <= last+20:
                host.append((value['collected'], phase(value)))
    host.sort(key=lambda value: value[0])
    times = [r[0] for r in host]
    groups = {}
    for stamp, device, utilization, power in rows:
        index = bisect.bisect_right(times, stamp)
        label = ('phase_boundary_or_missing_bracket',)
        if 0 < index < len(host):
            left, right = host[index-1], host[index]
            if left[1] == right[1] and stamp-left[0] <= 15 and right[0]-stamp <= 15:
                label = left[1]
        entry = groups.setdefault(label, {})
        entry.setdefault(device, {'utilization': [], 'board_power': []})['utilization'].append(utilization)
        entry[device]['board_power'].append(power)
    report = dict(collected=time.time(), status='native_physical_rows_joined_to_existing_phase_observations',
        source=__file__, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_physical_source=str(PHYSICAL), original_phase_source=str(HOST),
        window=dict(condition=window, lower_unix=lower, upper_unix=upper,
                    first_received=first, last_received=last),
        phase_alignment='Both adjacent original ten-second observations must agree and be within15 seconds; boundaries/mixed/partial observations remain explicit',
        physical_utilization_meaning='Native device activity percentage, not a measured fraction of peak FLOPS',
        missing_values='Never filled with zero',
        diagnostic_caveat='Bounded profiling and same-input allocator-pair intervals remain in the raw record; inspect their separate receipts before using these windows as fair steady-state throughput',
        phases=[dict(identity=list(label), devices={str(device): dict(
            gpu_utilization_percent=summary(values['utilization']),
            board_power_W=summary(values['board_power'])) for device, values in sorted(devices.items())})
            for label, devices in groups.items()],
        model_calls=0, training_mutations=0)
    suffix = '' if window == 'all' else '-'+window.replace('_', '-')
    destination = AUDIT / ('native-utilization-phase-summary'+suffix+'.json')
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(destination)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', choices=('all', 'before_allocator_pair', 'after_native_configuration'), default='all')
    main(parser.parse_args().window)
