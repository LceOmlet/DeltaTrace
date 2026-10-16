"""Keep the same native-cache timing test with a short AF_UNIX-safe Ray directory."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v4/source/run_performance_shm.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '91c1761d539f5fa6a2630c6b4fd502cdb806c95076c09e589f05e29528172365'
code = owner.read_text().replace('performance-exec2', 'performance-exec3')
code = code.replace('/tmp/ray-PPO32k-native-fast-dispatch-v4-shm-exec2', '/tmp/ray-PPO4e3')
exec(compile(code, str(owner), 'exec'), globals())
