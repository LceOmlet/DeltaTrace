"""Read both current formal AppWorld runs with the unchanged owner log reader."""
import hashlib
import json
from pathlib import Path
import runpy
import time
import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-formal-training-20261017-v1'
parser_path = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(parser_path.read_bytes()).hexdigest() == 'c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner = runpy.run_path(str(parser_path), run_name='current_appworld_original_log_reader')
manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
jobs = [j for j in manifest['jobs'] if j.get('task') == 'AppWorld' and j.get('method') in ('dt','ppo')]
snapshot = dict(collected=time.time(), jobs=[], parser=str(parser_path), formal_training_mutations=0,
                host_MemAvailable_GiB=psutil.virtual_memory().available/2**30)
for job in jobs:
    current = dict(job)
    processes = []
    alive = False
    try:
        process = psutil.Process(job['pid'])
        alive = process.create_time() == job['birth'] and process.status() != psutil.STATUS_ZOMBIE
        if alive:
            children = process.children(recursive=True)
            gcs = next((p for p in children if p.name() == 'gcs_server'), None)
            if gcs:
                log_dir = next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--log_dir='))
                current['ray_session'] = str(Path(log_dir).parent)
            processes = [dict(pid=p.pid,birth=p.create_time(),phase=p.name()) for p in children if 'WorkerDict' in p.name()]
    except psutil.NoSuchProcess:
        pass
    records, errors = owner['read_job_logs'](current)
    metrics, phases = owner['parse_logs'](records)
    checkpoint = Path(job['checkpoints'])
    marker = checkpoint / 'latest_checkpointed_iteration.txt'
    latest = None if not marker.exists() else int(marker.read_text().strip())
    completed_files = []
    if latest is not None:
        directory = checkpoint / f'global_step_{latest}'
        completed_files = [dict(path=str(p),bytes=p.stat().st_size) for p in directory.rglob('*') if p.is_file()]
    row = dict(task='AppWorld',method=job['method'],version=job['version'],pid=job['pid'],birth=job['birth'],
               alive=alive,workers=processes,ray_session=current.get('ray_session'),
               completed_iterations=max((m['step'] for m in metrics),default=0),total_iterations=200,
               metrics=metrics,phases=phases,errors=errors,checkpoint_path=str(checkpoint),checkpoint_interval=2,
               latest_checkpointed_iteration=latest,latest_completed_checkpoint_files=completed_files,
               source_receipt=job['source_receipt'])
    snapshot['jobs'].append(row)
(AUDIT / 'latest-native-snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
print(json.dumps({**snapshot,'jobs':[{k:v for k,v in row.items() if k not in ('metrics','phases','errors','latest_completed_checkpoint_files')}
                                   | dict(latest_metrics=row['metrics'][-1:],error_marker_count=len(row['errors']))
                                   for row in snapshot['jobs']]},indent=2))
