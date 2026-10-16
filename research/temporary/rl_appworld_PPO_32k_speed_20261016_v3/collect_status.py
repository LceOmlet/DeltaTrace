"""Reuse the recorded native Ray session linkage and original log reader."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT/'receipts/appworld-sql-native-speed-20261016-v3/source/collect_status.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '0b4714e54f54dbe2c8b2a49ec430f664fc7e27206d26b840f4ae648baf8071a6'
code = owner.read_text().replace('appworld-sql-native-speed-20261016-v3', 'appworld-native-PPO-32k-speed-20261016-v3')
exec(compile(code, str(owner), 'exec'), globals())
