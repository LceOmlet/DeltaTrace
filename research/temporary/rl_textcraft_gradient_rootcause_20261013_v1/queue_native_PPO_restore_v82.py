"""Queue the existing native state reader after the partition probe restore."""
from pathlib import Path

source = Path(__file__).with_name('queue_native_PPO_restore_v79.py')
code = source.read_text()
assert 'v79' in code
code = code.replace('v79', 'v82').replace('native-PPO-resume-after-control-v78-v82',
                                       'native-PPO-resume-after-partition-v81-v82')
exec(compile(code, str(source), 'exec'), globals())
