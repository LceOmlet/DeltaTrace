"""Count original microbatches and optimizer boundaries in one natural update.

This observer delegates every native call once and returns its original value.
It performs no extra forward, backward, synchronization or optimizer update.
Host elapsed times include device waits and are not GPU kernel durations.
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
AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'
RUN = ROOT / 'runs/textcraft-dtpo-v6-8gpu-200-20261017-v1'
VERSION = 'one-natural-native-actor-update-v1'


def install(worker):
    import hashlib
    import json
    import os
    from pathlib import Path
    import time
    from types import MethodType
    import psutil

    owner, = [w for w in worker.worker_dict.values() if getattr(w, '_is_actor', False)]
    expected = {r['pid']: r['birth'] for r in json.loads((RUN /
        'runtime-overrides/actual-worker-imports-after-first-DT-v1.json').read_bytes())['results']}
    assert psutil.Process().create_time() == expected[os.getpid()]
    actor = owner.actor
    delegate = actor.update_policy
    source = Path(delegate.__func__.__code__.co_filename)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
    assert 'update_policy' not in vars(actor)

    def observe(instance, data):
        record = dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=owner.rank,
            started=time.time(), native_actor_source=str(source), native_actor_SHA256=source_sha,
            local_minibatch=actor.config.ppo_mini_batch_size,
            microbatch_per_rank=actor.config.ppo_micro_batch_size_per_gpu,
            use_dynamic_bsz=actor.config.use_dynamic_bsz, ppo_epochs=actor.config.ppo_epochs,
            local_input_rows=len(data), scheduler_before=owner.actor_lr_scheduler.last_epoch,
            forwards=[], optimizer_boundaries=[],
            additional_forward_backward_tensor_optimizer_sync_calls=0,
            timing_scope='Host callable elapsed time, including original device waits')
        original_forward = actor._forward_micro_batch
        original_step = actor._optimizer_step
        saved = {name: vars(actor).get(name) for name in ('_forward_micro_batch', '_optimizer_step')}
        present = {name: name in vars(actor) for name in saved}

        def forward(this, *args, **kwargs):
            batch = args[0] if args else kwargs['micro_batch']
            entry = dict(started=time.time(), input_shape=list(batch['input_ids'].shape),
                         response_shape=list(batch['responses'].shape))
            tick = time.perf_counter()
            try:
                return original_forward(*args, **kwargs)
            finally:
                entry.update(finished=time.time(), host_seconds=time.perf_counter()-tick)
                record['forwards'].append(entry)

        def step(this, *args, **kwargs):
            entry = dict(started=time.time(), forwards_so_far=len(record['forwards']))
            tick = time.perf_counter()
            try:
                return original_step(*args, **kwargs)
            finally:
                entry.update(finished=time.time(), host_seconds=time.perf_counter()-tick)
                record['optimizer_boundaries'].append(entry)

        actor._forward_micro_batch = MethodType(forward, actor)
        actor._optimizer_step = MethodType(step, actor)
        tick = time.perf_counter()
        try:
            result = delegate(data)
            record['status'] = 'original_natural_update_returned'
            return result
        except BaseException as exc:
            record.update(status='original_natural_update_raised', original_exception=type(exc).__name__)
            raise
        finally:
            record.update(finished=time.time(), host_seconds=time.perf_counter()-tick)
            for name in saved:
                if present[name]:
                    setattr(actor, name, saved[name])
                else:
                    delattr(actor, name)
            del actor.update_policy
            record['original_callables_restored'] = actor.update_policy == delegate and all(
                (name in vars(actor)) == present[name] for name in saved)
            try:
                with (AUDIT / f'{VERSION}-rank{owner.rank}.json').open('x') as stream:
                    json.dump(record, stream, indent=2)
            except Exception as exc:
                print('Optional native update observation unavailable: '+type(exc).__name__, flush=True)

    actor.update_policy = MethodType(observe, actor)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=owner.rank,
        installed=True, scheduler=owner.actor_lr_scheduler.last_epoch,
        local_minibatch=actor.config.ppo_mini_batch_size,
        microbatch_per_rank=actor.config.ppo_micro_batch_size_per_gpu,
        native_actor_SHA256=source_sha, additional_model_optimizer_calls=0)


if __name__ == '__main__':
    source = json.loads((RUN / 'source.json').read_bytes())
    if '--client' not in sys.argv:
        assert not (AUDIT / (VERSION+'-job.json')).exists()
        env = dict(os.environ, **source['effective_environment'])
        env['CUDA_VISIBLE_DEVICES'] = ''
        with (AUDIT / (VERSION+'.stdout')).open('xb') as out, (AUDIT / (VERSION+'.stderr')).open('xb') as err:
            child = subprocess.Popen([env['VENV_PYTHON'], __file__, '--client'], env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        record = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time(), submitted=time.time(),
            source=__file__, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            status='optional_one_natural_update_metadata_installation_pending')
        (AUDIT / (VERSION+'-job.json')).write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record))
    else:
        import ray
        driver = psutil.Process(source['pid'])
        assert driver.create_time() == source['birth']
        gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
        host = next(v.split('=',1)[1] for v in gcs.cmdline() if v.startswith('--node-ip-address='))
        port = next(v.split('=',1)[1] for v in gcs.cmdline() if v.startswith('--gcs_server_port='))
        ray.init(address=host+':'+port, log_to_driver=False)
        try:
            names = [a for a in ray.util.list_named_actors(all_namespaces=True)
                if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
            assert len(names) == 8
            rows = ray.get([ray.get_actor(a['name'], namespace=a['namespace']).execute_with_func_generator.remote(func=install)
                for a in names])
            (AUDIT / (VERSION+'-installation.json')).write_text(json.dumps(dict(
                collected=time.time(), status='installed_one_natural_update_observation', workers=rows,
                additional_model_optimizer_calls=0), indent=2)+'\n')
        finally:
            ray.shutdown()
