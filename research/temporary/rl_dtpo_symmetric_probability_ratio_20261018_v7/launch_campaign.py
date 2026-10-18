"""Use the accepted no-duplicate launch owner with the v7 queue identity."""
import hashlib
from pathlib import Path
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1/source/launch_campaign.py'
code=source.read_text()
expected='fb2013fc2d8c18b1eb58d18a05f9c2dd637b76dd140f6236451d9e5bc0310cff'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
code=code.replace("'receipts/six-run-fresh-validated-efficiency-20261018-v1'","'receipts/dtpo-symmetric-probability-ratio-20261018-v7'")
exec(compile(code,str(source),'exec'),globals())
