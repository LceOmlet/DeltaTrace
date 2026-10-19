"""Compose the same six-run owner with the verified allocation configuration."""
import hashlib
from pathlib import Path
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1/source/run_campaign.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='7b6abc3b613f647244cea165e5732a5c07945e70f0de599530a7140f0715a15a'
code=owner.read_text().replace('postgroup-variance-20261019-v1','postgroup-variance-20261019-v2')
anchor="        assert json.loads((AUDIT/'CPU-behavior-verification.json').read_bytes())['status']=='passed'"
assert code.count(anchor)==1
code=code.replace(anchor,"        assert json.loads((AUDIT/'allocation-owner-verification.json').read_bytes())['status']=='passed_original_capture_copy_configured_native_hostalloc'\n"+anchor)
exec(compile(code,str(owner),'exec'),globals())
