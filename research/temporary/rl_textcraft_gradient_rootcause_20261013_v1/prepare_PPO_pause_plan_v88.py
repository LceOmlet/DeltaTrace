"""Reuse the complete native PPO pause plan for the full-loss direction readout."""
from pathlib import Path

source = Path(__file__).with_name('prepare_PPO_pause_plan_v82.py')
code = source.read_text().replace('PPO-native-pause-for-partition-v81-plan-v82.json',
    'PPO-native-pause-for-direction-v86-plan-v88.json')
exec(compile(code, str(source), 'exec'), globals())
