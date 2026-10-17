"""Read the same native worker metadata for the v3 campaign; zero model calls."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1/source/inspect_fresh_workers.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '467b37d880c572e5278e3b66214bc7065fe5b13903361539bd4389ef95ad935d'
code = source.read_text().replace("'textcraft-eight-gpu-sequential-20261017-v2'",
                                "'textcraft-eight-gpu-sequential-20261017-v3'")
code = code.replace('fresh-eight-GPU-initial-v1', 'threads8-eight-GPU-initial-v1')
exec(compile(code, str(source), 'exec'), globals())
