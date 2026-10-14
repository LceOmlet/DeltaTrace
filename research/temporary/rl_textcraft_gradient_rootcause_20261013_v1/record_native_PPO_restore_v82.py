"""Record original actor/critic states at the next native restored checkpoint."""
from pathlib import Path

source = Path(__file__).with_name('record_native_PPO_restore_v79.py')
code = source.read_text()
changes = {
    'PPO-native-pause-for-control-v78-plan-v79.json': 'PPO-native-pause-for-partition-v81-plan-v82.json',
    'native-PPO-resume-after-control-v78-v79': 'native-PPO-resume-after-partition-v81-v82',
    'native-PPO-bindings-v79.json': 'native-PPO-bindings-v82.json',
    'resume_PPO_after_control_probe_v79.py': 'resume_PPO_after_partition_probe_v82.py',
    'native-PPO-restore-verification-v79.json': 'native-PPO-restore-verification-v82.json',
    'resume-v79-job.json': 'resume-v82-job.json',
    'before-verified-restore-v79.json': 'before-verified-restore-v82.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
