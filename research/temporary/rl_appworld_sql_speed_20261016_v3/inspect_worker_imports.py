"""Reuse the native read-only worker RPC for the corrected version."""
from pathlib import Path

owner = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-sql-native-speed-20261016-v2/source/inspect_worker_imports.py')
code = owner.read_text().replace('appworld-sql-native-speed-20261016-v2', 'appworld-sql-native-speed-20261016-v3')
exec(compile(code, str(owner), 'exec'), globals())
