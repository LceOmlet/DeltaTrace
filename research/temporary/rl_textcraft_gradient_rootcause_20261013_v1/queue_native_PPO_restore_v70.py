"""Queue the unchanged native binding/state reader for the new full restore."""
import sys
from pathlib import Path

if len(sys.argv) == 1:
    sys.argv.append('v70')
source = Path(__file__).with_name('queue_native_PPO_restore_read_v55.py')
text = source.read_text()
changes = {
    "('v55','v56')": "('v70',)",
    'native-PPO-resume-after-v47-v55-job.json': 'native-PPO-resume-after-v69-v70-job.json',
    "version=='v56'": "version=='v70'",
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
exec(compile(text, str(source), 'exec'), globals())
