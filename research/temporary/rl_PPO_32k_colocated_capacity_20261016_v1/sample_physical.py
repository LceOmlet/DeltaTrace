"""Reuse the original bounded raw physical sampler for this capacity run."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/source/sample_physical.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'ad5b15e388422430f3f2e8ed9022ca64488c2e4274a4be93d4ac718ed7120d63'
code = owner.read_text().replace("AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2'",
                                "AUDIT=ROOT/'receipts/PPO-32k-colocated-capacity-20261016-v1'")
code = code.replace("AUDIT/'same-session-status.json'", "AUDIT/'status.json'")
exec(compile(code,str(owner),'exec'),globals())
