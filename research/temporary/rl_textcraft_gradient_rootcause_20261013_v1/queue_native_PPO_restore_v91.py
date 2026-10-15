"""Queue the unchanged actor/critic native state reader for this restore."""
from pathlib import Path
source = Path(__file__).with_name('queue_native_PPO_restore_v82.py')
code = source.read_text().replace('partition-v81', 'saved-update-v90').replace('v82', 'v91')
exec(compile(code, str(source), 'exec'), globals())
