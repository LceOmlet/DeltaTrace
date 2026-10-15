"""Queue the existing native actor/critic restore reader for the same run."""
from pathlib import Path
source = Path(__file__).with_name('queue_native_PPO_restore_v91.py')
code = source.read_text().replace('saved-update-v90', 'natural-PG-v99').replace('v91', 'v100')
exec(compile(code, str(source), 'exec'), globals())
