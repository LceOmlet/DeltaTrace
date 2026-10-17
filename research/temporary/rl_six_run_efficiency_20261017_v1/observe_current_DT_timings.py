"""Bind the existing self-restoring timing observer to the actual eight ranks.

It retains info already returned by one naturally scheduled DT RPC. It does
not call another model, finite propagation, tensor operation or optimizer.
"""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'
source = AUDIT / 'source/original_existing_DT_timings.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '68bbf90081a2e3d64d7ca35a4b881c581951f8469b390465d2ae319f01c559b4'
code = source.read_text()


def replace(before, after):
    global code
    assert code.count(before) == 1, before
    code = code.replace(before, after)


replace("RUN = ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'",
        "RUN = ROOT/'runs/textcraft-dtpo-v6-8gpu-200-20261017-v1'")
replace("DEST = ROOT/'receipts/textcraft-efficiency-20261015-v1'",
        "DEST = ROOT/'receipts/six-run-native-utilization-20261017-v1/existing-one-native-DT-RPC'")
replace('WORKERS = {3151306: 1791993626.33, 3153050: 1791993643.98}',
        """actual=json.loads((RUN/'runtime-overrides/actual-worker-imports-after-first-DT-v1.json').read_bytes())
WORKERS={row['pid']:row['birth'] for row in actual['results']}
assert len(WORKERS)==8""")
replace('assert len(names)==2', 'assert len(names)==8')
replace('existing-DT-timings-efficiency-20261015-v1.json', 'existing-DT-timings-eight-GPU-20261017-v1.json')
exec(compile(code, str(source), 'exec'), globals())
