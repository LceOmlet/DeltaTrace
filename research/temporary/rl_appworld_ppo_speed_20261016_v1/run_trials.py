"""Reuse the frozen native bounded queue on GPU2/3."""
import hashlib
from pathlib import Path
owner=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-sql-native-speed-20261016-v1/source/run_trials.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='ff15b26471618671f86cee95a07eb41a82b8c0a8b5ef5cb82636b6241b21b001'
code=owner.read_text().replace("appworld-sql-native-speed-20261016-v1",'appworld-native-ppo-speed-20261016-v1')
exec(compile(code,str(owner),"exec"),globals())
