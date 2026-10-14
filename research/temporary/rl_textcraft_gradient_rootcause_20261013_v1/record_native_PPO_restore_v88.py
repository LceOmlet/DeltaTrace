"""Record the complete original actor/critic state at the restored checkpoint."""
from pathlib import Path

source = Path(__file__).with_name('record_native_PPO_restore_v82.py')
code = source.read_text().replace('partition-v81', 'direction-v86').replace('v82', 'v88')
code = code.replace('resume_PPO_after_partition_probe_v88.py', 'resume_PPO_after_direction_probe_v88.py')
exec(compile(code, str(source), 'exec'), globals())
