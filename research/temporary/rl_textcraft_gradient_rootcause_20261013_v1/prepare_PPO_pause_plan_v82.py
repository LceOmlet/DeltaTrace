"""Reuse the verified complete PPO25+ state for the prepared causal probe.

The previous checkpoint30 wait was an extra scheduling choice. Reusing a
complete, verified25 checkpoint does not manufacture a new checkpoint or
change native resume mathematics. This prepared helper revision is v84.
"""
from pathlib import Path

source = Path(__file__).with_name('prepare_PPO_pause_plan_v79.py')
code = source.read_text()
changes = {
    'assert step >= 25,': 'assert step >= 25,',
    'PPO-native-pause-for-control-v78-plan-v79.json':
        'PPO-native-pause-for-partition-v81-plan-v82.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
