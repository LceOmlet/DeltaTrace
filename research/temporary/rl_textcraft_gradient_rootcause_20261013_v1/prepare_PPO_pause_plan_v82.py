"""Reuse the native full-state pause plan at completed PPO30 or later."""
from pathlib import Path

source = Path(__file__).with_name('prepare_PPO_pause_plan_v79.py')
code = source.read_text()
changes = {
    'assert step >= 25,': 'assert step >= 30,',
    'PPO-native-pause-for-control-v78-plan-v79.json':
        'PPO-native-pause-for-partition-v81-plan-v82.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
