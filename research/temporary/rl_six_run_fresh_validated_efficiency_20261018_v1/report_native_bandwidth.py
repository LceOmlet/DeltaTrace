"""Summarize original mx-smi bandwidth text with the existing phase owner.

The installed native JSON interface omits bandwidth fields. This adapter reads
the original timestamped text; it is not a training logger or log parser.
"""
import bisect
import datetime
import hashlib
import json
from pathlib import Path
import re
import time


ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-fresh-validated-efficiency-20261018-v1'
OWNER = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/summarize_native_utilization_v2.py'
OWNER_SHA = 'e6f3b17198f34ca0b5946637dcbd3ee4920bb08b9783bbc374ffe85bb14199e3'


def native_samples(path, kind):
    # Only records followed by another native timestamp are complete prefixes.
    # The last record may be in the middle of the native writer's append.
    parts = re.split(r'(?m)^Timestamp\s+:\s*([^\n]+)$', path.read_text())
    zone = datetime.timezone(datetime.timedelta(hours=8))
    for index in range(1, len(parts) - 2, 2):
        stamp = datetime.datetime.strptime(parts[index].strip(), '%a %b %d %H:%M:%S %Y').replace(tzinfo=zone).timestamp()
        devices, device, direction = {}, None, None
        for line in parts[index + 1].splitlines():
            match = re.match(r'^GPU#(\d+)\s', line)
            if match:
                device = int(match[1])
                devices[device] = {}
                direction = None
            if device is None:
                continue
            if line.strip() in ('MetaXLink RX Bandwidth', 'MetaXLink TX Bandwidth'):
                direction = line.strip().split()[1]
            match = re.match(r'^\s*(throughput|RX|TX|metaxlink\d+)\s*:\s*(\d+(?:\.\d+)?)\s+MBytes/s\s*$', line)
            if match:
                key, value = match[1], float(match[2])
                if key.startswith('metaxlink'):
                    key = direction + ':' + key
                devices[device][key] = value
        for device, fields in devices.items():
            if kind == 'metaxlink':
                # Preserve each port and each direction; do not double-count
                # TX+RX as one global interconnect throughput.
                values = {key: value for key, value in fields.items()}
            else:
                values = fields
            yield stamp, device, values


def main():
    assert hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA
    owner = {'__name__': '_original_native_phase_owner', '__file__': str(OWNER)}
    code = OWNER.read_text().replace('-8gpu-200-20261017-v1', '-8gpu-200-20261018-v5')
    exec(compile(code, str(OWNER), 'exec'), owner)
    streams = {kind: AUDIT / f'native-{kind}-bandwidth-v1.raw.log'
               for kind in ('hbm', 'pcie', 'metaxlink')}
    rows = [(kind, *row) for kind, path in streams.items() for row in native_samples(path, kind)]
    if not rows:
        print(json.dumps({'status': 'waiting_for_complete_native_bandwidth_records'}))
        return
    first, last = min(row[1] for row in rows), max(row[1] for row in rows)
    host = []
    with owner['HOST'].open() as stream:
        for line in stream:
            value = json.loads(line)
            if first - 20 <= value['collected'] <= last + 20:
                host.append((value['collected'], owner['phase'](value)))
    host.sort(key=lambda value: value[0])
    times = [value[0] for value in host]
    groups, missing = {}, []
    for kind, stamp, device, fields in rows:
        # Same bracketing rule as the pinned existing utilization report:
        # both adjacent original observations agree and are within15s.
        index = bisect.bisect_right(times, stamp)
        label = ('phase_boundary_or_missing_bracket',)
        if 0 < index < len(host):
            left, right = host[index - 1], host[index]
            if left[1] == right[1] and stamp - left[0] <= 15 and right[0] - stamp <= 15:
                label = left[1]
        expected = ('throughput',) if kind == 'hbm' else ('RX', 'TX') if kind == 'pcie' else ()
        absent = [key for key in expected if key not in fields]
        if absent:
            missing.append(dict(kind=kind, timestamp=stamp, device=device, fields=absent))
        entry = groups.setdefault((kind, label), {}).setdefault(device, {})
        for key, value in fields.items():
            entry.setdefault(key, []).append(value)
    report = dict(
        collected=time.time(), status='native_bandwidth_samples_joined_to_existing_phase_observations',
        source=str(Path(__file__)), source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        phase_owner=str(OWNER), phase_owner_SHA256=OWNER_SHA,
        native_streams={key: str(value) for key, value in streams.items()},
        original_phase_source=str(owner['HOST']), first_native_timestamp=first, last_native_timestamp=last,
        units='Original mx-smi MBytes/s. No peak-FLOPS, peak-bandwidth or ten-second-average conversion.',
        missing_values='Missing are not zero; native printed0 remains an actual observed0.',
        missing_fields=missing,
        coverage='Bandwidth starts during TextCraft DTPOv6 iteration8; preceding bandwidth is unavailable. Use matched observed windows.',
        phases=[dict(kind=kind, identity=list(label), devices={str(device): {
            key: owner['summary'](values) for key, values in fields.items()}
            for device, fields in sorted(devices.items())})
            for (kind, label), devices in groups.items()],
        model_calls=0, training_mutations=0,
    )
    destination = AUDIT / 'native-bandwidth-phase-summary-v1.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(destination)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
