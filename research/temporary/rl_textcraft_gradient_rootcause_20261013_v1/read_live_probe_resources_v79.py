"""Read the live native PPO checkpoint/phase before the authorized diagnostic."""
import json
from pathlib import Path
import psutil
import runpy
import subprocess
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
job = json.loads((OUT / 'native-PPO-control-v9-job.json').read_bytes())
p = psutil.Process(job['pid'])
assert p.create_time() == job['birth'] and p.status() != psutil.STATUS_ZOMBIE
run = Path(job['run'])
marker = run / 'textcraft-ppo/checkpoints/latest_checkpointed_iteration.txt'
session = list(Path('/tmp/ray').glob('session_*_' + str(job['pid'])))
assert len(session) == 1
owner = runpy.run_path(str(OUT / 'original_plot_training_progress.py'), run_name='read_existing_control_phase')
lines, errors = owner['read_job_logs'](dict(ray_session=str(session[0])))
metrics, phases = owner['parse_logs'](lines)
workers = [dict(pid=c.pid, birth=c.create_time(), phase=c.name(), status=c.status())
           for c in p.children(recursive=True) if c.name().startswith('ray::WorkerDict')]
report = dict(unix=time.time(), status='original_PPO_confirmed_live_not_stopped',
    pid=p.pid, birth=p.create_time(), latest_completed_checkpoint=int(marker.read_text()),
    completed_iteration=max((m['step'] for m in metrics), default=0),
    latest_original_metric=metrics[-1] if metrics else None, latest_original_phases=phases[-5:],
    workers=workers, log_errors=errors, ray_session=str(session[0]),
    physical_mx_smi=subprocess.check_output(['mx-smi'], text=True, timeout=30),
    model_backward_optimizer_scheduler_calls=0, signals_sent=0)
path = OUT / f'live-probe-preparation-resource-v79-{time.time_ns()}.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))
