"""Reuse the frozen native bounded queue on GPU2/3."""
import hashlib
from pathlib import Path
owner=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-sql-native-speed-20261016-v1/source/submit_trials.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='7647f046472eb7f045a9337d15ae885a980ba65a8e2a9227cedc47783881c6fc'
code=owner.read_text().replace("appworld-sql-native-speed-20261016-v1",'appworld-native-ppo-speed-20261016-v1')
exec(compile(code,str(owner),"exec"),globals())
