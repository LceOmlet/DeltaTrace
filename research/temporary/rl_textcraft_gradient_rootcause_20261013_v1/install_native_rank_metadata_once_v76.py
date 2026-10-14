"""Retain both ranks' existing metadata for one naturally scheduled v3 update.

Calls original native methods and returns their original objects. The callback
only inspects Python metadata; no tensor arithmetic, collective, model call,
backward, optimizer or scheduler call is added. This is not a training logger.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN = ROOT / 'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
DEST = OUT / 'live-native-rank-metadata-once-v76'
JOB = OUT / 'DTPO-variance-v3-job.json'
WORKERS = {3151306: 1791993626.33, 3153050: 1791993643.98}
ACTOR_SHA = '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
CORE_SHA = 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'


def install(worker):
    import hashlib
    import inspect
    import json
    import os
    from pathlib import Path
    import time
    from types import MethodType
    import psutil
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo import core_algos

    assert psutil.Process().create_time() == WORKERS[os.getpid()]
    owners = [w for w in worker.worker_dict.values() if getattr(w, '_is_actor', False)]
    assert len(owners) == 1
    w = owners[0]
    actor = w.actor
    assert type(actor) is DataParallelPPOActor and w.world_size == 2
    assert actor.update_policy.__func__ is DataParallelPPOActor.update_policy
    assert actor._optimizer_step.__func__ is DataParallelPPOActor._optimizer_step
    assert hashlib.sha256(Path(inspect.getsourcefile(type(actor))).read_bytes()).hexdigest() == ACTOR_SHA
    assert hashlib.sha256(Path(inspect.getsourcefile(core_algos)).read_bytes()).hexdigest() == CORE_SHA
    assert actor.config.ppo_mini_batch_size == 32
    assert actor.config.ppo_micro_batch_size_per_gpu == 4
    assert actor.config.ppo_epochs == 1 and actor.config.loss_agg_mode == 'token-mean'
    assert not hasattr(actor, '_native_rank_metadata_once_v76_installed')
    original_update = actor.update_policy
    original_step = actor._optimizer_step
    update_existed = 'update_policy' in vars(actor)
    step_existed = '_optimizer_step' in vars(actor)
    previous_update = vars(actor).get('update_policy')
    previous_step = vars(actor).get('_optimizer_step')
    actor._native_rank_metadata_once_v76_installed = True

    def observed_update(self, data):
        record = dict(unix=time.time(), pid=os.getpid(), birth=psutil.Process().create_time(),
            rank=w.rank, completed_iteration_before_update=w.actor_lr_scheduler.last_epoch,
            rows=len(data), native_norm_objects=[], native_metrics=None,
            actor_SHA256=ACTOR_SHA, core_SHA256=CORE_SHA,
            additional_model_calls=0, additional_backward_calls=0,
            additional_optimizer_scheduler_calls=0, additional_tensor_collectives=0,
            additional_tensor_arithmetic=0, credit_method_changes=0,
            native_reduction='Original token-mean B4 accumulation and native two-rank reduction.',
            rootcause_repair_verified=False)

        def observed_step(instance):
            norm = original_step()
            try:
                placements = getattr(norm, 'placements', ())
                record['native_norm_objects'].append(dict(
                    norm_class=type(norm).__module__ + '.' + type(norm).__qualname__,
                    dtype=str(norm.dtype), placements=[str(p) for p in placements],
                    placement_classes=[type(p).__module__ + '.' + type(p).__qualname__ for p in placements],
                    placement_state=[dict(vars(p)) for p in placements],
                    original_return_object_unchanged=True))
            except Exception as exc:
                record.setdefault('metadata_errors', []).append(repr(exc))
            return norm

        try:
            self._optimizer_step = MethodType(observed_step, self)
            result = original_update(data=data)
            record['native_metrics'] = result
            record['status'] = 'one_naturally_scheduled_native_update_returned'
            return result
        except BaseException as exc:
            record.update(status='native_update_raised', exception=repr(exc))
            raise
        finally:
            if step_existed:
                self._optimizer_step = previous_step
            else:
                del self._optimizer_step
            if update_existed:
                self.update_policy = previous_update
            else:
                del self.update_policy
            record['original_update_restored'] = self.update_policy == original_update
            record['original_optimizer_step_restored'] = self._optimizer_step == original_step
            try:
                path = DEST / f'rank{w.rank}-native-update.json'
                assert not path.exists()
                path.write_text(json.dumps(record, indent=2, default=str) + '\n')
            except Exception as exc:
                print('v76 passive metadata write failed: ' + repr(exc), flush=True)

    actor.update_policy = MethodType(observed_update, actor)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=w.rank,
        scheduler_at_install=w.actor_lr_scheduler.last_epoch, installed=True,
        actor_SHA256=ACTOR_SHA, core_SHA256=CORE_SHA)


def status():
    report = dict(unix=time.time(), installation=None, captures=[], client=None)
    path = DEST / 'client.json'
    if path.exists():
        c = json.loads(path.read_bytes())
        try:
            p = psutil.Process(c['pid'])
            alive = p.create_time() == c['birth'] and p.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        report['client'] = dict(c, alive=alive)
        report['client_stderr_tail'] = Path(c['stderr']).read_text(errors='replace')[-2500:]
    p = DEST / 'installation.json'
    if p.exists():
        report['installation'] = json.loads(p.read_bytes())
    for p in sorted(DEST.glob('rank*-native-update.json')):
        report['captures'].append(dict(path=str(p), SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                       value=json.loads(p.read_bytes())))
    return report


DEST.mkdir(exist_ok=True)
if '--client' in sys.argv:
    import ray
    job = json.loads(JOB.read_bytes())
    driver = psutil.Process(job['pid'])
    assert driver.create_time() == job['birth'] and Path(job['run']) == RUN
    gcs = next(c for c in driver.children(recursive=True) if c.name() == 'gcs_server')
    host = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    port = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    ray.init(address=host + ':' + port, log_to_driver=False)
    try:
        names = [a for a in ray.util.list_named_actors(all_namespaces=True)
                 if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
        assert len(names) == 2
        workers = ray.get([ray.get_actor(a['name'], namespace=a['namespace']).execute_with_func_generator.remote(
            func=install) for a in names], timeout=7200)
        assert {w['pid']: w['birth'] for w in workers} == WORKERS
        report = dict(unix=time.time(), status='one_shot_native_metadata_observation_installed',
            driver_pid=job['pid'], driver_birth=job['birth'], workers=workers,
            source=dict(path=__file__, SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            code_commit=os.environ.get('DT_ROOTCAUSE_METADATA_COMMIT'),
            model_backward_optimizer_scheduler_calls_added=0,
            tensor_arithmetic_collectives_added=0, training_math_changes=0,
            purpose='Retain both original rank metadata dictionaries and native norm return descriptors for one actual update.',
            rootcause_repair_verified=False)
        for p in (DEST / 'installation.json', RUN / 'runtime-overrides/native-rank-metadata-once-v76.json'):
            assert not p.exists()
            p.write_text(json.dumps(report, indent=2) + '\n')
    finally:
        ray.shutdown()
elif '--read' not in sys.argv and not (DEST / 'client.json').exists():
    job = json.loads(JOB.read_bytes())
    assert psutil.Process(job['pid']).create_time() == job['birth']
    env = dict(os.environ)
    env.update(json.loads((RUN / 'launch.json').read_bytes())['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    if '--code-commit' in sys.argv:
        env['DT_ROOTCAUSE_METADATA_COMMIT'] = sys.argv[sys.argv.index('--code-commit') + 1]
    stdout, stderr = DEST / 'client.stdout.log', DEST / 'client.stderr.log'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        p = subprocess.Popen([env['VENV_PYTHON'], '-u', __file__, '--client'],
            env=env, stdout=out, stderr=err, start_new_session=True)
    (DEST / 'client.json').write_text(json.dumps(dict(pid=p.pid, birth=psutil.Process(p.pid).create_time(),
        stdout=str(stdout), stderr=str(stderr)), indent=2) + '\n')
print(json.dumps(dict(receipt=str(DEST / 'installation.json'), report=status())))
