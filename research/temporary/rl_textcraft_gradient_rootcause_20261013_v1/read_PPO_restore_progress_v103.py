"""Reuse the specific original PPO restore progress reader for v100."""
from pathlib import Path
source = Path(__file__).with_name('read_PPO_restore_progress_v94.py')
code = source.read_text().replace('saved-update-v90', 'natural-PG-v99').replace('v91', 'v100')
code = code.replace("workers=[dict(pid=x.pid,birth=x.create_time(),phase=x.name())",
    "workers=[dict(pid=x.pid,birth=x.create_time(),phase=x.name(),PSS=x.memory_full_info().pss,read_bytes=x.io_counters().read_bytes,cpu_user=x.cpu_times().user)")
exec(compile(code, str(source), 'exec'), globals())
