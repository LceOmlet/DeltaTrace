"""Reuse the accepted eight-rank metadata RPC for the actual new v5 run."""
import hashlib
from pathlib import Path
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/inspect_native_workers.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='7f3c940abc65da9317a9e5f74182f5080cae066a116ddd6beaebc7476b36bb1d'
code=owner.read_text().replace('textcraft-eight-gpu-sequential-20261017-v1',
    'textcraft-v5-eight-gpu-20261019-v1')
code=code.replace("else 'eight-GPU-initial-v1')", "else 'v5-eight-GPU-initial-v1')")
exec(compile(code,str(owner),'exec'),globals())
