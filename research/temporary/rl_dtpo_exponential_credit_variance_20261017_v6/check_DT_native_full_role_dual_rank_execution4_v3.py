"""Retry the unchanged dual-rank capacity check with a valid Ray socket path.

Execution3 failed before worker/model initialization because the diagnostic's
temporary directory exceeded Ray's AF_UNIX socket pathname limit. Preserve it;
only the output identity and short Ray temporary directory change here.
"""
import hashlib
from pathlib import Path

legacy = Path(__file__).resolve().parent / 'check_DT_native_full_role_dual_rank_execution3_v3.py'
assert hashlib.sha256(legacy.read_bytes()).hexdigest() == '10f1d54728c849a08c0fd4046736923af3ce1b7fa2dc47eb8aad1d6ec71cf29f'
code = legacy.read_text()
old = 'receipts/dt-native-replay-host-lifetime-20261017-v3/execution3-full-role-dual-rank'
new = 'receipts/dt-native-replay-host-lifetime-20261017-v3/execution4-full-role-dual-rank'
assert code.count(old) == 1
code = code.replace(old, new, 1)
old = '/tmp/ray-DT-host-lifetime-v3-execution3-full-role-dual-rank'
assert code.count(old) == 1
code = code.replace(old, '/tmp/ray-dth4', 1)
EXECUTED_SOURCE_SHA256 = hashlib.sha256(code.encode()).hexdigest()
exec(compile(code, str(legacy), 'exec'), globals())
