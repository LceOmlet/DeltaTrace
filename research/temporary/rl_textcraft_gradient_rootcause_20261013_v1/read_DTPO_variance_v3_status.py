"""Use the existing native-log/status reader for the current v3 job."""
from pathlib import Path
source=Path(__file__).with_name('read_DT_group_status_v40.py')
text=source.read_text()
assert "OUT/'DT-group-multiplier-v40-job.json'" in text
text=text.replace("OUT/'DT-group-multiplier-v40-job.json'","OUT/'DTPO-variance-v3-job.json'")
text=text.replace('DT-group-multiplier-observation-v40.json','DTPO-variance-v3-observation.json')
exec(compile(text,str(source),'exec'),globals())
