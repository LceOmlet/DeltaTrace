"""Queue the unchanged native actor/critic binding reader for this restore."""
from pathlib import Path

source = Path(__file__).with_name('queue_native_PPO_restore_v79.py')
code = source.read_text()
assert 'v79' in code and 'native-PPO-resume-after-control-v78-v79' in code
code = code.replace('v79', 'v114').replace(
    'native-PPO-resume-after-control-v78-v114', 'native-PPO-resume-after-same-session-v113-v114')
exec(compile(code, str(source), 'exec'), globals())
