"""Reuse original-reader snapshot sampling for the corrected task interfaces."""
import contextlib
import io
import json
from pathlib import Path

owner = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-sql-native-speed-20261016-v2/source/sample_status.py')
code = owner.read_text().replace('appworld-sql-native-speed-20261016-v2', 'sql-native-multiturn-speed-20261016-v1')
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(code, str(owner), 'exec'), globals())
summary = json.loads(captured.getvalue())
full = json.loads((AUDIT / 'latest-trial-snapshot.json').read_bytes())
events = {}
for job in full['jobs']:
    found = {}
    if job.get('ray_session'):
        for path in (Path(job['ray_session']) / 'logs').glob('worker-*.out'):
            for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
                for marker in ('[loop_transport]', '[loop_trajectory]', '[DeltaTrace readout]'):
                    if marker in line:
                        found[(str(path), marker)] = dict(path=str(path), line=number, text=line[:1800])
    events[job['task']] = list(found.values())
for job in summary['jobs']:
    job['tail'] = [line for line in job['tail'] if any(marker in line for marker in
        ('Training Progress:', '[loop_transport]', '[loop_trajectory]', 'Error:', 'Traceback'))]
summary['original_structured_owner_progress'] = events
(AUDIT / ('native-owner-progress-' + str(int(full['collected_unix'])) + '.json')).write_text(json.dumps(events, indent=2) + '\n')
print(json.dumps(summary, indent=2))
