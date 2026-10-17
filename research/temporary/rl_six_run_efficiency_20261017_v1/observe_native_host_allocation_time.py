"""Read native pinned-allocator counters around one natural DT owner call.

This delegates the exact existing callable, returns its original object and
restores the previous observation adapter immediately. No extra tensor,
model, DT, optimizer, synchronization or checkpoint operation is performed.
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


def install(worker):
    from pathlib import Path
    from types import MethodType
    import json
    import os
    import time
    import torch
    import psutil

    actor, = [w for w in worker.worker_dict.values() if getattr(w, '_is_actor', False)]
    expected = {r['pid']: r['birth'] for r in json.loads((RUN / 'runtime-overrides/actual-worker-imports-after-first-DT-v1.json').read_bytes())['results']}
    assert psutil.Process().create_time() == expected[os.getpid()]
    runner = actor._deltatrace_producer.runner
    delegate = runner.attribute
    assert delegate.__func__.__name__ == 'observed_attribution', 'Keep the already installed existing-timing observer'
    assert not hasattr(runner, '_native_host_allocation_time_installed')
    runner._native_host_allocation_time_installed = True

    def counters():
        try:
            values = torch.cuda.memory.host_memory_stats()
            return {key: value for key, value in values.items() if
                    key.startswith(('host_alloc_time.', 'host_free_time.')) or
                    key in ('num_host_alloc', 'num_host_free')}
        except Exception as exc:
            return dict(observation_unavailable=type(exc).__name__)

    def observe(instance, *args, **kwargs):
        record = dict(pid=os.getpid(), rank=actor.rank, started=time.time(),
            before=counters(), source='Original torch.cuda.memory.host_memory_stats',
            counter_time_unit='microseconds, as documented by the installed owner',
            additional_model_DT_tensor_optimizer_collective_or_sync_calls=0)
        try:
            result = delegate(*args, **kwargs)
            record['owner_seconds'] = result[1].get('complete_attribution_seconds_with_diagnostics')
            record['owner_calls'] = result[1].get('calls')
            record['status'] = 'one_original_attribute_returned'
            return result
        except BaseException as exc:
            record.update(status='original_attribute_raised', original_exception=type(exc).__name__)
            raise
        finally:
            runner.attribute = delegate
            record.update(after=counters(), finished=time.time(), previous_callable_restored=runner.attribute == delegate)
            try:
                with (AUDIT / f'one-DT-native-host-counters-rank{actor.rank}.json').open('x') as stream:
                    json.dump(record, stream, indent=2)
            except Exception as exc:
                print('Optional native host counter observation unavailable: ' + type(exc).__name__, flush=True)

    runner.attribute = MethodType(observe, runner)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=actor.rank,
                scheduler=actor.actor_lr_scheduler.last_epoch, installed=True,
                additional_model_DT_tensor_optimizer_calls=0)


if __name__ == '__main__':
    source = json.loads((RUN / 'source.json').read_bytes())
    if '--client' not in sys.argv:
        assert not (AUDIT / 'host-counter-observer-job.json').exists()
        env = dict(os.environ, **source['effective_environment'])
        env['CUDA_VISIBLE_DEVICES'] = ''
        with (AUDIT / 'host-counter-observer.stdout').open('xb') as out, (AUDIT / 'host-counter-observer.stderr').open('xb') as err:
            child = subprocess.Popen([env['VENV_PYTHON'], __file__, '--client'], env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        record = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time(),
            submitted=time.time(), source_path=__file__, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            status='native_metadata_installation_pending', model_calls=0, optimizer_updates=0)
        (AUDIT / 'host-counter-observer-job.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record))
    else:
        import ray
        driver = psutil.Process(source['pid'])
        assert driver.create_time() == source['birth']
        gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
        host = next(v.split('=', 1)[1] for v in gcs.cmdline() if v.startswith('--node-ip-address='))
        port = next(v.split('=', 1)[1] for v in gcs.cmdline() if v.startswith('--gcs_server_port='))
        ray.init(address=host + ':' + port, log_to_driver=False)
        try:
            names = [a for a in ray.util.list_named_actors(all_namespaces=True)
                     if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
            assert len(names) == 8
            rows = ray.get([ray.get_actor(a['name'], namespace=a['namespace']).execute_with_func_generator.remote(func=install)
                            for a in names])
            record = dict(status='installed_one_natural_attribute_native_host_counter_observation',
                collected=time.time(), workers=rows, model_calls=0, DT_calls=0, optimizer_updates=0)
            (AUDIT / 'host-counter-observer-installation.json').write_text(json.dumps(record, indent=2) + '\n')
        finally:
            ray.shutdown()
