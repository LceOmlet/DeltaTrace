"""Observe64 already scheduled input preparations in two native vLLM engines.

The official TorchDispatchMode logging interface delegates each operation once.
No extra copies, tensors, synchronization, generation or optimizer calls occur.
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
VERSION = 'native-natural-MRoPE-copy-observation-20261017-v1'


def install(worker):
    import hashlib
    import inspect
    import json
    import os
    from pathlib import Path
    import time
    from types import MethodType
    import psutil
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    owner, = [w for w in worker.worker_dict.values() if getattr(w, '_is_actor', False)]
    if owner.rank not in (0, 6):
        return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=owner.rank, status='not_selected')
    runner = owner.rollout_sharding_manager.model_runner
    path = Path(inspect.getfile(type(runner)))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'e8a73e1ccbfff37b880cd47ba9c115db572aef3c9d4d9eee50b18b729117f9bb'
    assert '_prepare_inputs' not in vars(runner)
    delegate = runner._prepare_inputs
    positions = runner.mrope_positions
    target_addresses = (positions.gpu.data_ptr(), positions.cpu.data_ptr())
    record = dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=owner.rank,
        installed=time.time(), scheduler_at_install=owner.actor_lr_scheduler.last_epoch,
        native_source=str(path), native_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        additional_model_tensor_copy_optimizer_sync_calls=0, calls=[],
        timing_scope='Original host API elapsed time, including original waiting. Not isolated GPU kernel duration.')

    class ObserveCopy(TorchDispatchMode):
        def __init__(self, entry):
            super().__init__()
            self.entry = entry

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            target = False
            if func == torch.ops.aten.copy_.default and len(args) >= 2:
                try:
                    target = (args[0].data_ptr(), args[1].data_ptr()) == target_addresses
                except (AttributeError, RuntimeError):
                    pass
            if not target:
                return func(*args, **kwargs)
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self.entry['position_copies'].append(dict(tokens=args[0].shape[-1],
                    host_seconds=time.perf_counter()-started,
                    non_blocking=args[2] if len(args)>2 else kwargs.get('non_blocking', False)))

    def observe(instance, *args, **kwargs):
        entry = dict(started=time.time(), position_copies=[])
        started = time.perf_counter()
        raised = False
        try:
            with ObserveCopy(entry):
                return delegate(*args, **kwargs)
        except BaseException as exc:
            raised = True
            entry['original_exception'] = type(exc).__name__
            raise
        finally:
            entry.update(finished=time.time(), host_seconds=time.perf_counter()-started)
            record['calls'].append(entry)
            if raised or len(record['calls']) >= 64:
                del runner._prepare_inputs
                record.update(finished=time.time(), original_callable_restored=runner._prepare_inputs == delegate,
                    status='original_input_preparation_raised' if raised else 'bounded_natural_input_preparation_observation_completed')
                try:
                    with (AUDIT / f'{VERSION}-rank{owner.rank}.json').open('x') as stream:
                        json.dump(record, stream, indent=2)
                except Exception as exc:
                    print('Optional native input-copy observation unavailable: '+type(exc).__name__, flush=True)

    runner._prepare_inputs = MethodType(observe, runner)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=owner.rank,
                installed=True, scheduler_epoch=owner.actor_lr_scheduler.last_epoch)


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
            status='passive_native_copy_observation_pending_original_worker_boundary')
        (AUDIT / (VERSION+'-job.json')).write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record))
    else:
        import ray
        driver = psutil.Process(source['pid'])
        assert driver.create_time() == source['birth']
        gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
        host = next(v.split('=',1)[1] for v in gcs.cmdline() if v.startswith('--node-ip-address='))
        port = next(v.split('=',1)[1] for v in gcs.cmdline() if v.startswith('--gcs_server_port='))
        expected = {r['pid']:r['birth'] for r in json.loads((RUN /
            'runtime-overrides/actual-worker-imports-after-first-DT-v1.json').read_bytes())['results']}
        ray.init(address=host+':'+port, log_to_driver=False)
        try:
            names = [a for a in ray.util.list_named_actors(all_namespaces=True)
                if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
            assert len(names) == 8
            rows = ray.get([ray.get_actor(a['name'], namespace=a['namespace']).execute_with_func_generator.remote(func=install)
                for a in names])
            assert all(expected[r['pid']] == r['birth'] for r in rows)
            (AUDIT / (VERSION+'-installation.json')).write_text(json.dumps(dict(
                collected=time.time(), status='installed_bounded_passive_input_copy_observation', workers=rows,
                additional_model_tensor_copy_optimizer_sync_calls=0), indent=2)+'\n')
        finally:
            ray.shutdown()
