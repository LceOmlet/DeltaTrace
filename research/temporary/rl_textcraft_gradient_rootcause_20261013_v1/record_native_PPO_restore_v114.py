"""Verify actual native full PPO state at the dynamically planned checkpoint."""
from pathlib import Path

source = Path(__file__).with_name('record_native_PPO_restore_v79.py')
code = source.read_text()
changes = {
    'PPO-native-pause-for-control-v78-plan-v79.json': 'PPO-native-pause-for-same-session-v113-plan-v114.json',
    'native-PPO-resume-after-control-v78-v79': 'native-PPO-resume-after-same-session-v113-v114',
    'native-PPO-bindings-v79.json': 'native-PPO-bindings-v114.json',
    'resume_PPO_after_control_probe_v79.py': 'resume_PPO_after_same_session_probe_v114.py',
    'native-PPO-restore-verification-v79.json': 'native-PPO-restore-verification-v114.json',
    'resume-v79-job.json': 'resume-v114-job.json',
    'before-verified-restore-v79.json': 'before-verified-restore-v114.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
