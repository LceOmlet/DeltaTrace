"""Repeat the unchanged cost measurement using VERL's existing checkpoint file cache."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v4/source/run_performance.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '5e0ef17663b583e502b27b69846458967fe39f2c94a946d7e82d67c8a0f81112'
code = owner.read_text()
code = code.replace("AUDIT = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v4/performance'",
                    "AUDIT = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v4/performance-exec2'")
code = code.replace('/tmp/ray-PPO32k-native-fast-dispatch-v4', '/tmp/ray-PPO32k-native-fast-dispatch-v4-shm-exec2')
anchor = "checkpoint = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/numerical-warm-state-original-VERL'"
addition = '''
from verl.utils.fs import copy_to_local
original_checkpoint = checkpoint
write(AUDIT/'staging.json', dict(status='original_VERL_checkpoint_cache_copy', source=str(checkpoint),
      mechanism='Original copy_to_local(use_shm=True); no new checkpoint or model/optimizer loader', started=time.time()))
checkpoint = Path(copy_to_local(str(checkpoint), use_shm=True))
write(AUDIT/'staging.json', dict(status='cached', source=str(original_checkpoint), cached=str(checkpoint),
      mechanism='Original copy_to_local(use_shm=True); original native load_checkpoint still owns all restores', finished=time.time()))
'''
assert code.count(anchor) == 1
code = code.replace(anchor, anchor+addition)
exec(compile(code, str(owner), 'exec'), globals())
