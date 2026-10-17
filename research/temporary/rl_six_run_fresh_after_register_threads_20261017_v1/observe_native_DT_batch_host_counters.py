"""Reuse the verified opaque native producer observer for the v3 full batch."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1/source/observe_fresh_DT_batch_host_counters.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'a268fb7eece2580e0a6800a96f18b83b566da7423dab8c8f8d4728ec95e45fa6'
code = source.read_text().replace('six-run-fresh-after-efficiency-20261017-v1',
                                'six-run-fresh-after-register-threads-20261017-v1')
code = code.replace('runs/textcraft-dtpo-v6-8gpu-200-20261017-v2',
                    'runs/textcraft-dtpo-v6-8gpu-200-20261017-v3')
exec(compile(code, str(source), 'exec'), globals())
