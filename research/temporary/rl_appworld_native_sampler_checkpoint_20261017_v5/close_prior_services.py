"""Reuse native orphan cleanup for the two stopped, prior AppWorld executions."""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/appworld-environment-readiness-20261017-v4/source/close_failed_run_services.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='94653e682a9aa7dbb7e0ee26589abd5faa1cead5bbee66fa62da89139dc46141'
code=source.read_text()
for old,new in {
    "AUDIT=ROOT/'receipts/appworld-environment-readiness-20261017-v4'":
        "AUDIT=ROOT/'receipts/appworld-native-sampler-checkpoint-20261017-v5'",
    'appworld-dtpo-reward-target-exponential-formal-20261017-v3':
        'appworld-dtpo-reward-target-exponential-formal-20261017-v4',
    'appworld-native-PPO-formal-20261017-v1':'appworld-native-PPO-formal-20261017-v2',
    'failed-run-server-cleanup.json':'prior-execution-server-cleanup.json',
    'failed_versions=versions':'prior_executions=versions',
}.items():
    assert old in code,old
    code=code.replace(old,new)
exec(compile(code,str(source),'exec'),globals())
