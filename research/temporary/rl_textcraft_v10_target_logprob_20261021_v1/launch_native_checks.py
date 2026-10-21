"""Configure the original bounded native-check launcher for v10, physical GPU2/3."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-v10-target-logprob-20261021-v1'
owner=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5/source/launch_native_checks.py'
bindings=json.loads((AUDIT/'owner-source-bindings.json').read_bytes())
assert hashlib.sha256(owner.read_bytes()).hexdigest()==bindings[str(owner)]
code=owner.read_text()
for before,after in {
    "AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'":
        "AUDIT=ROOT/'receipts/textcraft-v10-target-logprob-20261021-v1'",
    "AUDIT/'source/run_native_behavior_v5.py'":"AUDIT/'source/run_native_behavior.py'",
    "status='submitted_native_interface_and_capacity'":"status='submitted_native_B4_target_endpoint_integration'",
}.items():
    assert code.count(before)==1,before
    code=code.replace(before,after)
exec(compile(code,str(owner),'exec'),globals())
