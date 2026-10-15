"""Restore the original full PPO checkpoint after the saved-update readout."""
from pathlib import Path
source = Path(__file__).with_name('resume_PPO_after_partition_probe_v82.py')
code = source.read_text().replace('native-upper-adapter-partition-v81', 'native-full-policy-saved-update-feedback-v90')
code = code.replace('partition-v81', 'saved-update-v90').replace('v82', 'v91')
exec(compile(code, str(source), 'exec'), globals())
