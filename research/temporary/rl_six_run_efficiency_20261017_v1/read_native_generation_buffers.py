"""Read the existing vLLM position-buffer metadata at an original RPC boundary.

No hooks, tensors, copies, model calls or configuration changes are installed.
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
VERSION = 'native-generation-position-buffers-20261017-v1'


def read(worker):
    import hashlib
    import inspect
    import os
    from pathlib import Path
    import psutil
    import torch

    owner, = [w for w in worker.worker_dict.values() if getattr(w, '_is_actor', False)]
    runner = owner.rollout_sharding_manager.model_runner
    path = Path(inspect.getfile(type(runner)))
    buffer = getattr(runner, 'mrope_positions', None)
    details = None
    if buffer is not None:
        details = dict(cpu_shape=list(buffer.cpu.shape), gpu_shape=list(buffer.gpu.shape),
            cpu_stride=list(buffer.cpu.stride()), gpu_stride=list(buffer.gpu.stride()),
            cpu_dtype=str(buffer.cpu.dtype), gpu_dtype=str(buffer.gpu.dtype),
            cpu_contiguous=buffer.cpu.is_contiguous(), gpu_contiguous=buffer.gpu.is_contiguous(),
            cpu_pinned=buffer.cpu.is_pinned())
    scheduler = getattr(getattr(runner, 'vllm_config', None), 'scheduler_config', None)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=owner.rank,
        native_runner_class=type(runner).__module__+'.'+type(runner).__qualname__,
        native_runner_source=str(path), native_runner_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        uses_mrope=getattr(runner, 'uses_mrope', None), pin_memory=getattr(runner, 'pin_memory', None),
        max_num_tokens=getattr(runner, 'max_num_tokens', None), mrope_positions=details,
        async_scheduling=getattr(scheduler, 'async_scheduling', None),
        native_allocator_settings=torch.cuda.memory._snapshot()['allocator_settings'],
        scheduler_epoch=owner.actor_lr_scheduler.last_epoch,
        additional_model_tensor_copy_optimizer_sync_or_config_calls=0)


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
            status='readonly_native_metadata_pending_original_worker_boundary')
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
            rows = ray.get([ray.get_actor(a['name'], namespace=a['namespace']).execute_with_func_generator.remote(func=read)
                for a in names])
            assert all(expected[r['pid']] == r['birth'] for r in rows)
            record = dict(collected=time.time(), status='read_existing_native_generation_buffer_metadata',
                driver_pid=driver.pid, driver_birth=driver.create_time(), workers=rows,
                source=__file__, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                additional_model_tensor_copy_optimizer_sync_or_config_calls=0)
            (AUDIT / (VERSION+'.json')).write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(record), flush=True)
        finally:
            ray.shutdown()
