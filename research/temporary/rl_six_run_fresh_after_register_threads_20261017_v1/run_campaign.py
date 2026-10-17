"""Compose the pinned original six-run process owner with v3 identities."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1/source/run_fresh_queue.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '60af705bf7633a8351982585c238f1a4372c7dc513b1d885b6f34bbed31bf691'
code = source.read_text().replace('receipts/six-run-fresh-after-efficiency-20261017-v1',
    'receipts/six-run-fresh-after-register-threads-20261017-v1')
exec(compile(code, str(source), 'exec'), globals())
