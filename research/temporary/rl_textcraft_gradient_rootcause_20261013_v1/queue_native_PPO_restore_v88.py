"""Queue the original actual actor/critic state reader after this restore."""
from pathlib import Path

source = Path(__file__).with_name('queue_native_PPO_restore_v82.py')
code = source.read_text().replace('partition-v81', 'direction-v86').replace('v82', 'v88')
exec(compile(code, str(source), 'exec'), globals())
