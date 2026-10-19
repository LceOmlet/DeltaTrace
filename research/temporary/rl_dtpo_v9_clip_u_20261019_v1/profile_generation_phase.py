"""Reuse the native bounded sampler for the current generation and its caller."""
from pathlib import Path

owner=Path(__file__).resolve().parent.parent/'rl_six_run_efficiency_20261017_v1/profile_generation_phase.py'
code=owner.read_text(encoding='utf-8')
old="source = Path(__file__).with_name('profile_current_DT.py')"
assert code.count(old)==1
code=code.replace(old,"source = Path(__file__).resolve().parent.parent/'rl_six_run_efficiency_20261017_v1/profile_current_DT.py'")
anchor='code = source.read_text()'
assert code.count(anchor)==1
code=code.replace(anchor,anchor+'''
code=code.replace('receipts/two-task-eight-gpu-sequential-20261017-v2/six-combination-manifest.json',
    'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2/six-combination-manifest.json')
code=code.replace('runtime-overrides/actual-worker-imports-after-first-DT-v1.json',
    'runtime-overrides/actual-worker-imports-after-first-update-v1.json')
code=code.replace('receipts/six-run-native-utilization-20261017-v1',
    'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2')
''')
exec(compile(code,str(owner),'exec'),globals())
