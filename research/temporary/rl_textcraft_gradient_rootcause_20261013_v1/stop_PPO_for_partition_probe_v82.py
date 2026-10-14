"""Reuse the authorized native checkpoint pause for the partition probe."""
from pathlib import Path

source = Path(__file__).with_name('stop_PPO_for_control_probe_v79.py')
code = source.read_text()
changes = {
    'PPO-native-pause-for-control-v78-plan-v79.json': 'PPO-native-pause-for-partition-v81-plan-v82.json',
    'native-PPO-resume-after-control-v78-v79-prepared.json': 'native-PPO-resume-after-partition-v81-v82-prepared.json',
    'PPO-temporary-stop-for-control-v78-v79.json': 'PPO-temporary-stop-for-partition-v81-v82.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
