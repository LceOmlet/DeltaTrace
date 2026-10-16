"""Reuse the existing read-only native restore verification for recovery v2."""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/appworld-formal-training-20261017-v1/source/verify_TextCraft_PPO_restore.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='c4867366e7fbf6f0dbd1d27956df4345cbd8964650f33898d42221d090947fda'
code=source.read_text()
replacements={
    'textcraft-PPO-global-OOM-recovery-20261017-v1':'textcraft-PPO-global-OOM-recovery-20261017-v2',
    'TextCraft-PPO-native-OOM-recovery-input.json':'TextCraft-PPO-native-OOM-recovery-input-v2.json',
    'TextCraft-PPO-native-OOM-recovery-state-readback-raw.json':'TextCraft-PPO-native-OOM-recovery-state-readback-raw-v2.json',
    'native-PPO-restore-global-OOM-20261017-v1.json':'native-PPO-restore-global-OOM-20261017-v2.json',
}
for old,new in replacements.items():
    assert old in code,old
    code=code.replace(old,new)
exec(compile(code,str(source),'exec'),globals())
