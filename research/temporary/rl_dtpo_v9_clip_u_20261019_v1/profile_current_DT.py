"""Reuse the existing 20-second nonblocking DT call-site/physical sampler."""
from pathlib import Path


owner=Path(__file__).resolve().parent.parent/'rl_six_run_efficiency_20261017_v1/profile_current_DT.py'
code=owner.read_text(encoding='utf-8')
for old,new in [
    ('receipts/two-task-eight-gpu-sequential-20261017-v2/six-combination-manifest.json',
     'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2/six-combination-manifest.json'),
    ('runtime-overrides/actual-worker-imports-after-first-DT-v1.json',
     'runtime-overrides/actual-worker-imports-after-first-update-v1.json'),
    ('receipts/six-run-native-utilization-20261017-v1',
     'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'),
]:
    assert old in code, old
    code=code.replace(old,new)
exec(compile(code,str(owner),'exec'),globals())
