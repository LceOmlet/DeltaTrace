"""Use the existing authorized complete native PPO pause, without changes."""
from pathlib import Path
source = Path(__file__).with_name('stop_PPO_for_partition_probe_v82.py')
code = source.read_text().replace('partition-v81', 'saved-update-v90').replace('v82', 'v91')
exec(compile(code, str(source), 'exec'), globals())
