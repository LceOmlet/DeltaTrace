"""Reuse prepared-only execution commit pinning; method source remains a88d479c."""
import hashlib
from pathlib import Path
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8/source/pin_execution_commit.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='a9d2316b63c1930c259e6ce3fe6f640b2dba321e672cb06ef282ef958ad0c243'
code=owner.read_text().replace("'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'",
    "'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'")
exec(compile(code,str(owner),'exec'),globals())
