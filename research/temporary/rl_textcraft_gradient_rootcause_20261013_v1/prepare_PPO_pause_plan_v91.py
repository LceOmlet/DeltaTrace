"""Reuse the original complete native PPO state for the saved-update readout."""
from pathlib import Path
source = Path(__file__).with_name('prepare_PPO_pause_plan_v82.py')
code = source.read_text().replace('PPO-native-pause-for-partition-v81-plan-v82.json', 'PPO-native-pause-for-saved-update-v90-plan-v91.json')
exec(compile(code, str(source), 'exec'), globals())
