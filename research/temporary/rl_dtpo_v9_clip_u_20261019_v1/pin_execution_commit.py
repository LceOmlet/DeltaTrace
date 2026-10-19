"""Reuse the original prepared-only commit resolution owner for this queue."""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8/source/pin_execution_commit.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='a9d2316b63c1930c259e6ce3fe6f640b2dba321e672cb06ef282ef958ad0c243'
code=owner.read_text().replace("'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'",
    "'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1'")
code=code.replace('method_verified_commit=previous,','method_verified_commit=target,preparation_record_commit=previous,')
code=code.replace('method_commit=previous,','method_commit=target,preparation_record_commit=previous,')
exec(compile(code,str(owner),'exec'),globals())
