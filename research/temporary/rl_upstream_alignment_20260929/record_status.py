"""Correct stale status only after checking the recorded processes and markers."""
import json
from pathlib import Path
import time

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
audit = root / 'receipts/upstream-alignment-20260929'
report = {'checked_at': time.time(), 'tasks': []}
for name in ('formal-training.json', 'active-training.json'):
    path = root / name
    old = path.read_text()
    state = json.loads(old)
    for job in state.get('jobs', []):
        pids = {key: job.get(key) for key in ('pid', 'driver_pid', 'worker_pid', 'task_runner_pid')}
        alive = {key: pid for key, pid in pids.items() if pid and Path(f'/proc/{pid}').exists()}
        if alive:
            raise RuntimeError(f'Recorded PID exists; verify identity before changing status: {alive}')
        run = Path(job['run_dir'])
        marker = run / 'checkpoints/latest_checkpointed_iteration.txt'
        exit_file = run / 'exit-code'
        job['status'] = 'not_running_requires_alignment'
        job['last_checked'] = report['checked_at']
        job['current_observation'] = {
            'recorded_processes_alive': False,
            'completed_checkpoint_marker': marker.read_text().strip() if marker.exists() else None,
            'exit_code': exit_file.read_text().strip() if exit_file.exists() else None,
            'scope': 'Filesystem/process observation; checkpoint contents and task performance not revalidated.'}
        if name == 'formal-training.json':
            report['tasks'].append({'task': job['task'], **job['current_observation']})
    backup = audit / (name + '.before-status-correction')
    if not backup.exists():
        backup.write_text(old)
    state['status'] = 'not_running_requires_alignment'
    state['last_checked'] = report['checked_at']
    path.write_text(json.dumps(state, indent=2) + '\n')
(audit / 'restored-status.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
