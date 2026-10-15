"""Verify actual original model/optimizer/scheduler/RNG/data restoration."""
from pathlib import Path
source = Path(__file__).with_name('record_native_PPO_restore_v91.py')
code = source.read_text().replace('saved-update-v90', 'natural-PG-v99').replace('v91', 'v100')
code = code.replace('resume_PPO_after_saved_update_readout_v100.py', 'resume_PPO_after_natural_PG_v100.py')
exec(compile(code, str(source), 'exec'), globals())
