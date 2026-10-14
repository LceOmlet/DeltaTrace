"""Reuse the accepted initial-state and passive-record installer for v3."""
from pathlib import Path
source=Path(__file__).with_name('verify_DT_group_workers_v40.py')
text=source.read_text()
assert "OUT/'DT-group-multiplier-v40-job.json'" in text
text=text.replace("OUT/'DT-group-multiplier-v40-job.json'","OUT/'DTPO-variance-v3-job.json'")
text=text.replace('initial-state-and-existing-passive-records-v40.json','initial-state-and-existing-passive-records-v3.json')
exec(compile(text,str(source),'exec'),globals())
