"""Observe one natural original DT producer batch, retaining its opaque result."""
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
code = code.replace('runner = actor._deltatrace_producer.runner', 'runner = actor._deltatrace_producer')
code = code.replace('runner.attribute', 'runner.attribute_prepared_batch')
anchor = "    assert delegate.__func__.__name__ == 'observed_attribution', 'Keep the already installed existing-timing observer'"
assert code.count(anchor) == 1
code = code.replace(anchor, '''    import hashlib
    native_source = Path(delegate.__func__.__code__.co_filename)
    assert hashlib.sha256(native_source.read_bytes()).hexdigest() == '2aa5f55252a793458ff5bd569a5e203e18a99094e4fa3d3cd896b70c43b6b73e'
''')
anchor = '        record = dict(pid=os.getpid(), rank=actor.rank, started=time.time(),'
assert code.count(anchor) == 1
code = code.replace(anchor, '        started_perf = time.perf_counter()\n'+anchor)
anchor = "            record['owner_seconds'] = result[1].get('complete_attribution_seconds_with_diagnostics')"
assert code.count(anchor) == 1
code = code.replace(anchor, "            record['owner_seconds'] = time.perf_counter()-started_perf")
anchor = "            record['owner_calls'] = result[1].get('calls')"
assert code.count(anchor) == 1
code = code.replace(anchor, "            record['scope'] = 'Original attribute_prepared_batch including all internal B4 calls; excludes outer worker pre/postprocessing'")
code = code.replace('one-DT-native-host-counters', 'one-batch-DT-native-host-counters')
code = code.replace('host-counter-observer', 'batch-host-counter-observer')
exec(compile(code, str(source), 'exec'), globals())
