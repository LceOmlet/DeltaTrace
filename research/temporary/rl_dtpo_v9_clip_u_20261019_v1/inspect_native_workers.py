"""Reuse the original eight-rank metadata RPC; no model or optimizer calls."""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/inspect_native_workers.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='7f3c940abc65da9317a9e5f74182f5080cae066a116ddd6beaebc7476b36bb1d'
code=owner.read_text().replace('textcraft-eight-gpu-sequential-20261017-v1',
    'textcraft-eight-gpu-v9-clip-u-20261019-v1')
code=code.replace("else 'eight-GPU-initial-v1')","else 'v9-clip-u-eight-GPU-initial-v1')")
exec(compile(code,str(owner),'exec'),globals())
