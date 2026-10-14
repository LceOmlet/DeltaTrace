"""Resume the native full checkpoint after the diagnostic partition probe."""
from pathlib import Path

source = Path(__file__).with_name('resume_PPO_after_control_probe_v79.py')
code = source.read_text()
changes = {
    'PPO-native-pause-for-control-v78-plan-v79.json': 'PPO-native-pause-for-partition-v81-plan-v82.json',
    'PPO-temporary-stop-for-control-v78-v79.json': 'PPO-temporary-stop-for-partition-v81-v82.json',
    'native-PPO-resume-after-control-v78-v79': 'native-PPO-resume-after-partition-v81-v82',
    'native-trained-control-propagation-v78': 'native-upper-adapter-partition-v81',
    '-v79.stdout.log': '-v82.stdout.log', '-v79.stderr.log': '-v82.stderr.log',
    'before-resume-v79.json': 'before-resume-v82.json', 'resume-v79-job.json': 'resume-v82-job.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
