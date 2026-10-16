"""Reuse the original trial log reader without another training log parser."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT / 'receipts/appworld-sql-native-speed-20261016-v1/source/collect_status.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '8e1e9754033645b87aa205b91f632bce48d67702d6d6d54f494878c29a71bad6'
code = owner.read_text().replace('appworld-sql-native-speed-20261016-v1', 'appworld-sql-native-speed-20261016-v2')
exec(compile(code, str(owner), 'exec'), globals())
