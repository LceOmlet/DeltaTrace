"""Record actual original model/optimizer/scheduler/RNG/data restoration."""
from pathlib import Path
source = Path(__file__).with_name('record_native_PPO_restore_v82.py')
code = source.read_text().replace('partition-v81', 'saved-update-v90').replace('v82', 'v91')
code = code.replace('resume_PPO_after_partition_probe_v91.py', 'resume_PPO_after_saved_update_readout_v91.py')
exec(compile(code, str(source), 'exec'), globals())
