"""Reuse the original occupancy, identity, and no-duplicate launch checks."""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1/source/launch_campaign.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='fb2013fc2d8c18b1eb58d18a05f9c2dd637b76dd140f6236451d9e5bc0310cff'
code=owner.read_text().replace("'receipts/six-run-fresh-validated-efficiency-20261018-v1'",
    "'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v3'")
exec(compile(code,str(owner),'exec'),globals())
