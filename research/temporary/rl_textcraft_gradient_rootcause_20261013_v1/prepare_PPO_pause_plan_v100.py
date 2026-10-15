"""Prepare the existing authorized full native pause before actual PG readout."""
from pathlib import Path
source = Path(__file__).with_name('prepare_PPO_pause_plan_v91.py')
code = source.read_text().replace('saved-update-v90-plan-v91', 'natural-PG-v99-plan-v100')
exec(compile(code, str(source), 'exec'), globals())
