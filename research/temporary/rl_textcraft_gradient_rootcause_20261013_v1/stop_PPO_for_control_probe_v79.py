"""Reuse the authorized native checkpoint pause, protecting the live jobs."""
from pathlib import Path

source = Path(__file__).with_name('stop_PPO_for_saved_v3_v70.py')
text = source.read_text()
changes = {
    'PPO-native-pause-for-v69-plan-v70.json': 'PPO-native-pause-for-control-v78-plan-v79.json',
    'native-PPO-resume-after-v69-v70-prepared.json': 'native-PPO-resume-after-control-v78-v79-prepared.json',
    'PPO-temporary-stop-for-v69-v70.json': 'PPO-temporary-stop-for-control-v78-v79.json',
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
exec(compile(text, str(source), 'exec'), globals())
