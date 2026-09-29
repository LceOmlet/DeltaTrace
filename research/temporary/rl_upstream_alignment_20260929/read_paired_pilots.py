"""Read bounded pilots' original Ray logs and checkpoint markers, without polling."""
import json
from pathlib import Path
import re
import time

audit = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/upstream-alignment-20260929')
report = {'checked_unix': time.time(), 'jobs': []}
for job in json.loads((audit / 'two-gpu-pilots/manifest.json').read_text()):
    directory = Path(job['directory'])
    children = Path(f'/proc/{job["pid"]}/task/{job["pid"]}/children')
    pids = children.read_text().split() if children.exists() else []
    sessions = [path for path in Path('/tmp/ray').glob('session_*')
                if any(path.name.endswith('_'+pid) for pid in pids)]
    # Once a job exits, retain its observed session rather than use session_latest.
    previous = directory / 'observed-ray-sessions.json'
    if sessions:
        previous.write_text(json.dumps([str(path) for path in sessions]))
    elif previous.exists():
        sessions = [Path(path) for path in json.loads(previous.read_text())]
    logs = [directory / 'train.log']
    for session in sessions:
        logs.extend((session / 'logs').glob('worker-*.out'))
        logs.extend((session / 'logs').glob('worker-*.err'))
    entries, errors, metrics = [], [], []
    for log in logs:
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-512*1024))
            lines = stream.read().decode(errors='replace').replace('\r', '\n').splitlines()
        selected = []
        for line in lines:
            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            if '[DT EOS minimum]' in line or '[DT EOS failed minibatch]' in line:
                continue  # exact replay inputs remain in the source log
            if any(key in line for key in ('[DT rollout]', '[DT EOS plan]', '[DT EOS minibatch]',
                                          'Training Progress:', '[DeltaTrace] token advantages')):
                selected.append(line[:1600])
            if any(key in line for key in ('Traceback (most recent call last)', 'OutOfMemoryError',
                                          'Nonfinite DT coefficients', 'ValueError:', 'RuntimeError:')):
                errors.append({'source': str(log), 'line': line[:1600]})
            if 'actor/grad_norm' in line and 'step:' in line:
                metrics.append({'source': str(log), 'line': line})
        if selected:
            entries.append({'source': str(log), 'last_write_unix': log.stat().st_mtime,
                            'tail': selected[-4:]})
    marker = directory / 'checkpoints/latest_checkpointed_iteration.txt'
    exit_path = directory / 'exit-code'
    item = {'task': job['task'], 'method': job['method'], 'devices': job['devices'],
            'elapsed_seconds': time.time()-job['started_unix'],
            'shell_exists': children.exists(), 'ray_sessions': list(map(str, sessions)),
            'checkpoint_marker': marker.read_text().strip() if marker.exists() else None,
            'exit_code': exit_path.read_text().strip() if exit_path.exists() else None,
            'phase_log_entries': entries, 'errors': errors, 'completed_metrics': metrics}
    stop_record = directory / 'dataloader-stall-stop.json'
    if stop_record.exists():
        item['intentional_stop'] = json.loads(stop_record.read_text())
    report['jobs'].append(item)
report['cgroup_usage_bytes'] = int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
(audit / 'two-gpu-pilots/latest-observation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
