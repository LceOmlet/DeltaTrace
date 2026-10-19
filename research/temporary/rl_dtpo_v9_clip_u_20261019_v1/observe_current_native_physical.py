"""Resume the accepted global dmon stream with a distinct v9 observer receipt.

Append original raw lines/timestamps; preserve earlier sessions. This is the
same read-only hardware producer, with no model or training instrumentation.
"""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/observe_native_gpu_utilization.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'f03bc4dfbea4bb0a728a27674cb5b57f6a7a705ceed6ee2955bb99d3ee53c55c'
code = owner.read_text()
anchor = "job_path = AUDIT / 'physical-observer-job.json'"
assert code.count(anchor) == 1
code = code.replace(anchor,
    "job_path = ROOT / 'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2/physical-observer-job.json'")
for name in ('native-mx-smi-dmon.csv','native-mx-smi-dmon-timestamped.jsonl','native-mx-smi-dmon.stderr'):
    old = repr(name) + ").open('x')"
    assert code.count(old) == 1
    code = code.replace(old, repr(name) + ").open('a')")
assert code.count('eight-gpu-sequential-20261017-v1') == 1
code = code.replace('eight-gpu-sequential-20261017-v1','eight-gpu-v9-clip-u-20261019-v2')
EXECUTED_SOURCE_SHA256 = hashlib.sha256(code.encode()).hexdigest()
exec(compile(code,str(owner),'exec'),globals())
