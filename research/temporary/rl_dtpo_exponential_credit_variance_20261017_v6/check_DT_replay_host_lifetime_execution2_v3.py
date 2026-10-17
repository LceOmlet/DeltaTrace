"""Retry the same bounded resource comparison after its proven sensor error.

The first execution ended before the original DT call. Keep that execution and
its source intact; correct only physical-readout handling and execution paths.
The DT resource candidate, original fixture and native worker remain unchanged.
"""
import hashlib
from pathlib import Path
import time

legacy = Path(__file__).resolve().parent / 'check_DT_replay_host_lifetime_v3.py'
assert hashlib.sha256(legacy.read_bytes()).hexdigest() == '8794f5234857e56ff781796d03dcee2872c4d8c31641df2231b0c6d168873e09'
code = legacy.read_text()


def physical_host_snapshot():
    result = dict(collected=time.time())
    try:
        result['MemAvailable_KiB'] = int(next(line.split()[1] for line in
            Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    except (OSError, ValueError, StopIteration) as exc:
        result['MemAvailable_read_error'] = repr(exc)
    try:
        # Same native sysfs units as the existing campaign resource observer.
        result['XTT_single_pool_KiB'] = int(Path(
            '/sys/class/drm/card1/device/mem_info_xtt_used').read_text().split()[0])
    except (OSError, ValueError, IndexError) as exc:
        # Optional sensor metadata must not discard a valid native DT result.
        result['XTT_read_error'] = repr(exc)
    return result


old = """            before_physical = {
                'collected': time.time(),
                'MemAvailable_KiB': int(next(x.split()[1] for x in
                    Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:'))),
                'XTT_single_pool_KiB': int(Path(
                    '/sys/class/drm/card1/device/mem_info_xtt_used').read_text()),
            }
"""
assert code.count(old) == 1
code = code.replace(old, "            before_physical = physical_host_snapshot()\n", 1)
replacements = [
    ("AUDIT = ROOT / 'receipts/dt-native-replay-host-lifetime-20261017-v3'",
     "AUDIT = ROOT / 'receipts/dt-native-replay-host-lifetime-20261017-v3/execution2'", 1),
    ("_temp_dir='/tmp/ray-DT-host-lifetime-v3'",
     "_temp_dir='/tmp/ray-DT-host-lifetime-v3-execution2'", 1),
    ("original_actor_verification_method_reused=True",
     "original_actor_verification_method_reused=True, physical_sample=physical_host_snapshot()", 1),
]
for old, new, count in replacements:
    assert code.count(old) == count, old
    code = code.replace(old, new)
EXECUTED_SOURCE_SHA256 = hashlib.sha256(code.encode()).hexdigest()
exec(compile(code, str(legacy), 'exec'), globals())
