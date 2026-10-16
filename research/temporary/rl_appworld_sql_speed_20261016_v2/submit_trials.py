"""Reuse the existing single-submission owner for the independent GPU6/7 queue."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT / 'receipts/appworld-sql-native-speed-20261016-v1/source/submit_trials.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '7647f046472eb7f045a9337d15ae885a980ba65a8e2a9227cedc47783881c6fc'
code = owner.read_text()
for old, new in [
    ('appworld-sql-native-speed-20261016-v1', 'appworld-sql-native-speed-20261016-v2'),
    ('[2,3]', '[6,7]'),
]:
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())
