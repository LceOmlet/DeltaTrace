"""Reuse the authorized native PPO process-group pause after restore is ready."""
from pathlib import Path

source = Path(__file__).with_name('stop_PPO_for_saved_v3_v70.py')
code = source.read_text()
changes = {
    'PPO-native-pause-for-v69-plan-v70.json': 'PPO-native-pause-for-same-session-v113-plan-v114.json',
    'native-PPO-resume-after-v69-v70-prepared.json': 'native-PPO-resume-after-same-session-v113-v114-prepared.json',
    'PPO-temporary-stop-for-v69-v70.json': 'PPO-temporary-stop-for-same-session-v113-v114.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
