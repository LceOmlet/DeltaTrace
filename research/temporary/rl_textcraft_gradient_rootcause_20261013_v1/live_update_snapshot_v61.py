"""Reuse the existing one-shot input observer on the approved live DTPO run.

This adapter changes only job identity, receipt destination and install timing.
The original observer delegates to the unchanged native update and restores it.
No additional model, backward, optimizer, scheduler or rollout call is made.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN = ROOT/'runs/textcraft-dt-grpo-multiplier-no-whiten-20261014-v1'
DEST = OUT/'live-native-update-input-v61'
DEST.mkdir(exist_ok=True)
JOB = OUT/'DT-group-multiplier-v40-job.json'
CLIENT = DEST/'client.json'
RESULT = DEST/'installation.json'
OVERRIDE = RUN/'runtime-overrides/live-native-update-input-v61.json'
ORIGINAL = OUT/'install_update_snapshot.py'
ORIGINAL_SHA = '62757e6090aed73174c4c93d68f509d7dc91e27ff186c837b6ee2c20b25b99ab'
WORKERS = {1426979: 1791977192.37, 1429032: 1791977210.63}


def observe_status():
    report = dict(unix=time.time(),installation=None,client=None,captures=[])
    if CLIENT.exists():
        client = json.loads(CLIENT.read_bytes())
        alive = False
        try:
            p = psutil.Process(client['pid'])
            alive = p.create_time() == client['birth'] and p.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
        report['client'] = dict(client, alive=alive)
        report['stderr_tail'] = Path(client['stderr']).read_text(errors='replace')[-3000:]
    if RESULT.exists():
        report['installation'] = json.loads(RESULT.read_bytes())
    for p in sorted(DEST.glob('*/rank*-capture.json')):
        report['captures'].append(dict(path=str(p), SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                       value=json.loads(p.read_bytes())))
    return report


if '--client' in sys.argv:
    import ray
    job = json.loads(JOB.read_bytes())
    p = psutil.Process(job['pid'])
    assert p.create_time() == job['birth'] and Path(job['run']) == RUN
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA
    # Load only the original observer definition, never the old-run installer.
    tree = ast.parse(ORIGINAL.read_bytes(), filename=str(ORIGINAL))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'METHOD' for t in node.targets):
            break
        nodes.append(node)
    namespace = dict(__name__='existing_passive_observer_v61', __file__=str(ORIGINAL))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ORIGINAL), 'exec'), namespace)
    # The generic observer already supports immediate capture for a non-'dt'
    # label; this is a receipt label only, not a change to a training method.
    namespace['OUT'] = DEST
    namespace['METHOD'] = 'approved_DTPO_actual_update'
    original_install = namespace['install']

    def install_checked(worker):
        import os
        import psutil
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        assert psutil.Process().create_time() == WORKERS[os.getpid()]
        owners = [w for w in worker.worker_dict.values() if getattr(w, '_is_actor', False)]
        assert len(owners) == 1
        actor = owners[0].actor
        assert actor.update_policy.__func__ is DataParallelPPOActor.update_policy
        result = original_install(worker)
        result['native_update_before_install'] = True
        return result

    gcs = next(c for c in p.children(recursive=True) if c.name() == 'gcs_server')
    port = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    host = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    ray.init(address=host+':'+port, log_to_driver=False)
    try:
        names = [a for a in ray.util.list_named_actors(all_namespaces=True)
                 if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
        assert len(names) == 2
        refs = [ray.get_actor(a['name'], namespace=a['namespace']).execute_with_func_generator.remote(
            func=install_checked) for a in names]
        workers = ray.get(refs, timeout=7200)
        assert {w['pid']: w['birth'] for w in workers} == WORKERS
        report = dict(unix=time.time(), status='existing_one_shot_observer_installed',
            driver_pid=p.pid, driver_birth=p.create_time(), workers=workers,
            observer=dict(path=str(ORIGINAL), SHA256=ORIGINAL_SHA),
            adapter=dict(path=__file__, SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            additional_model_calls=0, additional_optimizer_calls=0,
            training_configuration_changes=0, credit_method_changes=0,
            purpose='Read actual native PPO input to verify the approved zero-group route and no final whitening.',
            rootcause_repair_verified=False)
        assert not RESULT.exists() and not OVERRIDE.exists()
        RESULT.write_text(json.dumps(report, indent=2)+'\n')
        OVERRIDE.write_text(json.dumps(report, indent=2)+'\n')
    finally:
        ray.shutdown()
elif '--read' not in sys.argv and not CLIENT.exists():
    job = json.loads(JOB.read_bytes())
    assert psutil.Process(job['pid']).create_time() == job['birth']
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA
    stdout = DEST/'client.stdout.log'
    stderr = DEST/'client.stderr.log'
    env = dict(os.environ)
    env.update(json.loads((RUN/'launch.json').read_bytes())['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    with stdout.open('wb') as out, stderr.open('wb') as err:
        proc = subprocess.Popen([env['VENV_PYTHON'], '-u', __file__, '--client'],
            stdout=out, stderr=err, env=env, start_new_session=True)
    client = dict(pid=proc.pid, birth=psutil.Process(proc.pid).create_time(),
                  stdout=str(stdout), stderr=str(stderr), driver_pid=job['pid'], driver_birth=job['birth'])
    CLIENT.write_text(json.dumps(client, indent=2)+'\n')

print(json.dumps(dict(receipt=str(RESULT), report=observe_status())))
