"""Reuse the same bounded native trial owner after the evidenced interface fixes."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT / 'receipts/appworld-sql-native-speed-20261016-v1/source/run_trials.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'ff15b26471618671f86cee95a07eb41a82b8c0a8b5ef5cb82636b6241b21b001'
code = owner.read_text()
for old, new in [('appworld-sql-native-speed-20261016-v1', 'sql-native-format-observation-20261016-v1'),
                 ('[2,3]', '[6,7]'), ("'2,3'", "'6,7'"), ('(?:2|3)', '(?:6|7)'), ('GPU2/3', 'GPU6/7')]:
    assert old in code
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())
