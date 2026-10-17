"""Reuse the native counter observer once on an unchanged natural DT call."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/observe_native_host_allocation_time.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '2368842e65e043ad98bd44b811a246e73ea90ad7dd28959ebd2787000ec01606'
code = source.read_text()
code = code.replace('receipts/six-run-native-utilization-20261017-v1',
                    'receipts/six-run-fresh-after-efficiency-20261017-v1')
code = code.replace('runs/textcraft-dtpo-v6-8gpu-200-20261017-v1',
                    'runs/textcraft-dtpo-v6-8gpu-200-20261017-v2')
anchor = "    assert delegate.__func__.__name__ == 'observed_attribution', 'Keep the already installed existing-timing observer'"
assert code.count(anchor) == 1
code = code.replace(anchor, '''    import hashlib
    native_source = Path(delegate.__func__.__code__.co_filename)
    assert hashlib.sha256(native_source.read_bytes()).hexdigest() == '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
''')
exec(compile(code, str(source), 'exec'), globals())
