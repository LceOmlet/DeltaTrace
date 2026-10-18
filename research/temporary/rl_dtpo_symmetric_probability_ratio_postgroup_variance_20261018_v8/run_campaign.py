"""Use the accepted native six-run queue; replace only the manifest identity."""
import hashlib
from pathlib import Path
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1/source/run_campaign.py'
code=source.read_text()
expected='c36704c5dc2d1e517a5d4419dfb65efe020219f91d1aa107096f5a1baff1dc24'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
code=code.replace("'receipts/six-run-fresh-validated-efficiency-20261018-v1'","'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'")
exec(compile(code,str(source),'exec'),globals())
