"""Reuse the authorized PPO pause after native full-state preparation."""
from pathlib import Path

source = Path(__file__).with_name('stop_PPO_for_partition_probe_v82.py')
code = source.read_text().replace('partition-v81', 'direction-v86').replace('v82', 'v88')
exec(compile(code, str(source), 'exec'), globals())
