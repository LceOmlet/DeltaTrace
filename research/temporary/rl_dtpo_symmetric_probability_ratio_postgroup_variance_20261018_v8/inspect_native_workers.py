"""Reuse the accepted eight-rank read-only import RPC for the current v7 run."""
import hashlib
import os
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
task=os.environ.get('RL_CAMPAIGN_INSPECT_TASK','TextCraft')
assert task in ('TextCraft','AppWorld')
source=ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/inspect_native_workers.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='7f3c940abc65da9317a9e5f74182f5080cae066a116ddd6beaebc7476b36bb1d'
code=source.read_text().replace('textcraft-eight-gpu-sequential-20261017-v1',
    task.lower()+'-eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8')
code=code.replace("else 'eight-GPU-initial-v1')", "else 'symmetric-v8-initial-v1')")
if task=='AppWorld':
    code=code.replace('TextCraft','AppWorld')
exec(compile(code,str(source),'exec'),globals())
