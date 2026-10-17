"""Reuse the original four role/rank native restore metadata verification."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/appworld-formal-training-20261017-v1/source/verify_TextCraft_PPO_restore.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'c4867366e7fbf6f0dbd1d27956df4345cbd8964650f33898d42221d090947fda'
code = source.read_text()
for before, after in {
    'textcraft-PPO-global-OOM-recovery-20261017-v1': 'textcraft-PPO-host-memory-recovery-20261017-v3',
    'TextCraft-PPO-native-OOM-recovery-input.json': 'TextCraft-PPO-native-OOM-recovery-input-v2.json',
    'TextCraft-PPO-native-OOM-recovery-state-readback-raw.json': 'TextCraft-PPO-host-memory-recovery-state-readback-raw-v3.json',
    'native-PPO-restore-global-OOM-20261017-v1.json': 'native-PPO-restore-host-memory-20261017-v3.json',
    "manifest['TextCraft_PPO_global_OOM_recovery_20261017_v1'].update(": "manifest.setdefault('TextCraft_PPO_host_memory_recovery_20261017_v3',{}).update(",
}.items():
    assert before in code, before
    code = code.replace(before, after)
exec(compile(code, str(source), 'exec'), globals())
