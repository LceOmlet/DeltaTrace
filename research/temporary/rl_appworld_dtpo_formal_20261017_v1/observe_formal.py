"""Read this formal run through the existing log parser and worker observer."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'appworld-dtpo-reward-target-exponential-formal-20261017-v1'
AUDIT = ROOT / 'receipts' / VERSION
RUN = ROOT / 'runs' / VERSION
job = json.loads((RUN / 'launch.json').read_bytes())

if '--client' not in sys.argv:
    env = dict(os.environ)
    env.update(job['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('MACA_VISIBLE_DEVICES', None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'], __file__, '--client'] + sys.argv[1:], env=env))

import psutil

process = psutil.Process(job['pid'])
assert process.create_time() == job['birth']
gcs = next(p for p in process.children(recursive=True) if p.name() == 'gcs_server')
log_dir = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--log_dir='))
session = str(Path(log_dir).parent)
owner_path = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
owner = runpy.run_path(str(owner_path), run_name='formal_appworld_original_log_reader')
records, errors = owner['read_job_logs'](dict(job, ray_session=session))
metrics, phases = owner['parse_logs'](records)
checkpoint = Path(job['checkpoints'])
marker = checkpoint / 'latest_checkpointed_iteration.txt'
snapshot = dict(
    version=VERSION, collected_unix=time.time(), pid=process.pid, birth=process.create_time(),
    completed_iterations=max((m['step'] for m in metrics), default=0), total_iterations=200,
    metrics=metrics, phases=phases, errors=errors, ray_session=session,
    workers=[dict(pid=p.pid, birth=p.create_time(), phase=p.name())
             for p in process.children(recursive=True) if 'WorkerDict' in p.name()],
    checkpoint_path=str(checkpoint), checkpoint_interval=2,
    latest_checkpointed_iteration=None if not marker.exists() else marker.read_text().strip(),
    checkpoint_entries=[p.name for p in checkpoint.iterdir()],
    host_MemAvailable_GiB=psutil.virtual_memory().available / 2**30,
    parser=dict(path=str(owner_path), SHA256=hashlib.sha256(owner_path.read_bytes()).hexdigest()),
    formal_training_mutations=0, model_calls=0, optimizer_updates=0)
(AUDIT / 'latest-formal-snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')

if '--imports' in sys.argv:
    import ray

    inspector = AUDIT / 'source/original_worker_import_observer.py'
    assert hashlib.sha256(inspector.read_bytes()).hexdigest() == '77e4e0bdea4a3874a495e601b3feb251ad61fa8166a5d4417f126fa91c3d020c'
    tree = ast.parse(inspector.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inspect_worker')
    BINDINGS = job['source_bindings']
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(inspector), 'exec'), globals())
    port = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    host = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    ray.init(address=host + ':' + port, log_to_driver=False)
    try:
        named = [x for x in ray.util.list_named_actors(all_namespaces=True)
                 if 'WorkerDict_' in x['name'] and 'register_center' not in x['name']]
        assert len(named) == 2
        requests = [ray.get_actor(x['name'], namespace=x['namespace'])
                    .execute_with_func_generator.remote(func=inspect_worker) for x in named]
        try:
            results = ray.get(requests, timeout=25)
            assert all(m.get('matches_prepared_source') is not False
                       for result in results for m in result['modules'].values())
            record = dict(status='completed_read_only_worker_binding_check', collected_unix=time.time(),
                          pid=process.pid, birth=process.create_time(), results=results,
                          observer_SHA256=hashlib.sha256(inspector.read_bytes()).hexdigest())
        except ray.exceptions.GetTimeoutError:
            record = dict(status='pending_busy_native_workers_not_claimed_verified', collected_unix=time.time(),
                          pid=process.pid, birth=process.create_time(),
                          note='Only read-only metadata RPC requested; no cancellation or training mutation.')
        (AUDIT / 'actual-worker-imports.json').write_text(json.dumps(record, indent=2) + '\n')
        snapshot['worker_import_status'] = record['status']
    finally:
        ray.shutdown()

print(json.dumps({k: v for k, v in snapshot.items() if k not in ('metrics', 'phases', 'errors')}, indent=2))
print(json.dumps(dict(latest_metrics=metrics[-1:] if metrics else [], latest_phases=phases[-8:],
                      error_marker_count=len(errors)), indent=2))
