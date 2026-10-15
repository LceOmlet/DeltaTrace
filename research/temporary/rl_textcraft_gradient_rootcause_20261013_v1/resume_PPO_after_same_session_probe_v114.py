"""Use original VERL full-state resume after the same-session adjoint readout."""
from pathlib import Path

source = Path(__file__).with_name('resume_PPO_after_control_probe_v79.py')
code = source.read_text()
changes = {
    'PPO-native-pause-for-control-v78-plan-v79.json': 'PPO-native-pause-for-same-session-v113-plan-v114.json',
    'PPO-temporary-stop-for-control-v78-v79.json': 'PPO-temporary-stop-for-same-session-v113-v114.json',
    'native-PPO-resume-after-control-v78-v79': 'native-PPO-resume-after-same-session-v113-v114',
    'native-trained-control-propagation-v78': 'native-same-session-saved-update-feedback-v113',
    '-v79.stdout.log': '-v114.stdout.log',
    '-v79.stderr.log': '-v114.stderr.log',
    'before-resume-v79.json': 'before-resume-v114.json',
    'resume-v79-job.json': 'resume-v114-job.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
