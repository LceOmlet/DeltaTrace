"""Reuse the original bounded two-rank native DT check launcher for negative2."""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5/source/launch_native_checks.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='01fbcc415148df869ba64479c4cce976c2e4b89e4395d9a050543b73e8b16e8e'
code=owner.read_text().replace("'receipts/textcraft-reward-target-exponential-20261016-v5'",
    "'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v3'")
code=code.replace('source/run_native_behavior_v5.py','source/run_native_behavior.py')
exec(compile(code,str(owner),'exec'),globals())
