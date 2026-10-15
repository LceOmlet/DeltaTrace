"""Use the already authorized temporary PPO pause, with full restore prepared."""
from pathlib import Path
source = Path(__file__).with_name('stop_PPO_for_saved_update_readout_v91.py')
code = source.read_text().replace('saved-update-v90', 'natural-PG-v99').replace('v91', 'v100')
exec(compile(code, str(source), 'exec'), globals())
