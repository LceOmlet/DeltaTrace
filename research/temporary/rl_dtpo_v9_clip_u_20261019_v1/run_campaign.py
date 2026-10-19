"""Reuse the pinned original six-run completion and transition owner."""
import hashlib
from pathlib import Path
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1/source/run_campaign.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='c36704c5dc2d1e517a5d4419dfb65efe020219f91d1aa107096f5a1baff1dc24'
code=owner.read_text().replace("'receipts/six-run-fresh-validated-efficiency-20261018-v1'",
    "'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1'")
anchor="        selected = json.loads(Path(plan['selected_native_resources']).read_bytes())"
assert code.count(anchor)==1
code=code.replace(anchor,"""        assert json.loads((AUDIT/'CPU-behavior-verification.json').read_bytes())['status']=='passed'
        assert json.loads((AUDIT/'native-verification-summary.json').read_bytes())['status']=='passed_native_v9_DT_and_exact32768_B4_per_rank'
"""+anchor)
exec(compile(code,str(owner),'exec'),globals())
