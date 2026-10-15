"""Restore the same native full PPO checkpoint after the actual PG readout."""
from pathlib import Path
source = Path(__file__).with_name('resume_PPO_after_saved_update_readout_v91.py')
code = source.read_text().replace('native-full-policy-saved-update-feedback-v90', 'native-saved-update-natural-PG-v99')
code = code.replace('saved-update-v90', 'natural-PG-v99').replace('v91', 'v100')
exec(compile(code, str(source), 'exec'), globals())
