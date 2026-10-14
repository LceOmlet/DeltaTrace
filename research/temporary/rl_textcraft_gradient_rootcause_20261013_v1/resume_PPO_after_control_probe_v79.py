"""Use original VERL full-checkpoint resume after the read-only control probe."""
from pathlib import Path

source = Path(__file__).with_name('resume_native_PPO_after_diagnostic_v55.py')
text = source.read_text()
old = "stop=json.loads((OUT/'PPO-temporary-stop-for-v47-v54.json').read_bytes())\nassert stop['status']=='temporarily_stopped_native_full_restore_required'"
new = """if '--prepare' in sys.argv:
    stop=json.loads((OUT/'PPO-native-pause-for-control-v78-plan-v79.json').read_bytes())
    assert stop['status']=='planned_only_PPO_still_running'
else:
    stop=json.loads((OUT/'PPO-temporary-stop-for-control-v78-v79.json').read_bytes())
    assert stop['status']=='temporarily_stopped_native_full_restore_required'"""
assert old in text
text = text.replace(old, new)
changes = {
    'native-PPO-resume-after-v47-v55': 'native-PPO-resume-after-control-v78-v79',
    'native-common-update-feedback-v47': 'native-trained-control-propagation-v78',
    '[(1419849,1791977138.31),(3174448,1791793629.73)]':
        '[(3140624,1791993570.92),(3174448,1791793629.73)]',
    '-v55.stdout.log': '-v79.stdout.log', '-v55.stderr.log': '-v79.stderr.log',
    'before-resume-v55.json': 'before-resume-v79.json',
    'resume-v55-job.json': 'resume-v79-job.json',
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
exec(compile(text, str(source), 'exec'), globals())
