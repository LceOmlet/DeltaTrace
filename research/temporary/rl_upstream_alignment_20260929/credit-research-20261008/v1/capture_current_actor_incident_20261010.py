"""Attach passive incident recording through the existing VERL worker RPC.

Only native update_policy/optimizer methods are called. The observer keeps a
rolling CPU snapshot of original inputs, trainable local tensors, optimizer
state, and RNG. A native nonfinite skip retains that snapshot and its native
step call counts. It does not change losses, masks, precision, gradients,
parameters, optimizer decisions, configuration, or training limits.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import psutil
import ray

ACTOR_SHA = '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
EXPECTED = {987808: 1791553850.00, 989860: 1791553867.51}
OUT = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-actor-incidents-20261010-v1'


def install(worker):
    import copy
    import math
    import random

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    pid = os.getpid()
    birth = psutil.Process().create_time()
    assert EXPECTED.get(pid) == birth, (pid, birth)
    reports = []
    for name, owner in worker.worker_dict.items():
        if not getattr(owner, '_is_actor', False):
            continue
        actor = owner.actor
        source = Path(inspect.getsourcefile(type(actor)))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == ACTOR_SHA
        if hasattr(actor, '_native_incident_observer_20261010'):
            reports.append(actor._native_incident_observer_20261010['installation'])
            continue
        out = Path(OUT) / ('rank'+str(owner.rank)+'-pid'+str(pid))
        out.mkdir(parents=True, exist_ok=True)
        native_update = actor.update_policy
        native_step = actor._optimizer_step
        optimizer = actor.actor_optimizer
        state = dict(update_index=0, active=False,
                     native_optimizer_step_calls=0,
                     incident_count=0)

        def cpu(value):
            # Local FSDP2 tensors are diagnostic artifacts, not a replacement
            # for the framework's checkpoint or distributed restore owner.
            if isinstance(value, torch.Tensor):
                if hasattr(value, 'to_local'):
                    value = value.to_local()
                return value.detach().to('cpu', copy=True)
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu(v) for v in value)
            return copy.deepcopy(value)

        def write(name, value):
            path = out / name
            tmp = path.with_suffix(path.suffix+'.tmp')
            tmp.write_text(json.dumps(value, indent=2, allow_nan=True)+'\n')
            tmp.replace(path)

        def append(value):
            with (out/'native-events.jsonl').open('a') as stream:
                stream.write(json.dumps(value, allow_nan=True)+'\n')

        def diagnostic_error(phase, error):
            append(dict(event='observer_error', phase=phase, unix=time.time(),
                        error=repr(error), pid=pid))

        def snapshot(data):
            started = time.perf_counter()
            parameters = {key: cpu(value) for key, value in
                          actor.actor_module.named_parameters() if value.requires_grad}
            metadata = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                        placements=str(getattr(value, 'placements', None)),
                        mesh=str(getattr(value, 'device_mesh', None)))
                        for key, value in actor.actor_module.named_parameters() if value.requires_grad}
            artifact = dict(unix=time.time(), pid=pid, birth=birth, rank=owner.rank,
                update_index=state['update_index'], source_path=str(source), source_sha256=ACTOR_SHA,
                input_batch=data.batch.to('cpu').clone(),
                input_non_tensor_batch=copy.deepcopy(data.non_tensor_batch),
                input_meta_info=copy.deepcopy(data.meta_info),
                trainable_local_parameters=parameters, parameter_metadata=metadata,
                optimizer_local_state=cpu(optimizer.state_dict()),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                effective_config=OmegaConf.to_container(owner.config, resolve=True),
                scope=__doc__)
            pending = out/'pending-update.pt'
            tmp = out/'pending-update.pt.tmp'
            torch.save(artifact, tmp)
            tmp.replace(pending)
            # No copies of model activations or full frozen weights are kept.
            del artifact, parameters
            record = dict(event='native_update_input_saved', unix=time.time(),
                pid=pid, birth=birth, update_index=state['update_index'],
                path=str(pending), bytes=pending.stat().st_size,
                seconds=time.perf_counter()-started,
                pss_bytes=psutil.Process().memory_full_info().pss,
                CUDA_allocated=torch.cuda.memory_allocated(), CUDA_reserved=torch.cuda.memory_reserved())
            state['snapshot'] = record
            append(record)

        def observed_optimizer_step(_optimizer, _args, _kwargs):
            if state['active']:
                state['native_optimizer_step_calls'] += 1

        def observed_step(*args, **kwargs):
            before_step = state['native_optimizer_step_calls']
            started = time.perf_counter()
            result = native_step(*args, **kwargs)
            try:
                norm = float(result.detach().item())
                record = dict(event='native_optimizer_return', unix=time.time(),
                    pid=pid, birth=birth, update_index=state['update_index'],
                    optimizer_index=state['optimizer_index'], grad_norm=norm,
                    native_optimizer_step_calls=state['native_optimizer_step_calls']-before_step,
                    seconds=time.perf_counter()-started)
                if not math.isfinite(norm):
                    state['incident_count'] += 1
                    pending = out/'pending-update.pt'
                    incident = out/('incident-'+str(time.time_ns())+'-update.pt')
                    # A hard link retains the original snapshot without a
                    # second copy; the next rolling snapshot replaces its inode.
                    if state.get('snapshot') is not None:
                        assert state['snapshot']['update_index'] == state['update_index']
                        os.link(pending, incident)
                        record.update(incident_path=str(incident), bytes=incident.stat().st_size,
                            snapshot=state['snapshot'])
                    else:
                        record['input_snapshot_missing'] = True
                    record['incident_count'] = state['incident_count']
                    write('latest-incident.json', record)
                append(record)
            except Exception as error:
                diagnostic_error('native_step_return', error)
            state['optimizer_index'] += 1
            return result

        def observed_update(*args, **kwargs):
            data = kwargs.get('data', args[0] if args else None)
            state['update_index'] += 1
            state['optimizer_index'] = 0
            state['active'] = True
            state['snapshot'] = None
            try:
                try:
                    snapshot(data)
                except Exception as error:
                    diagnostic_error('input_snapshot', error)
                return native_update(*args, **kwargs)
            finally:
                state['active'] = False
                append(dict(event='native_update_return', unix=time.time(),
                    pid=pid, update_index=state['update_index'],
                    optimizer_returns=state['optimizer_index'],
                    incident_count=state['incident_count']))

        installation = dict(unix=time.time(), pid=pid, birth=birth, rank=owner.rank,
            name=name, actor_source=str(source), actor_sha256=ACTOR_SHA,
            observer_path=__file__, observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            out=str(out), scope=__doc__, loss_changes=0, parameter_changes=0,
            new_model_calls=0, configuration_changes=0, training_limits_added=0,
            methods=['actor.update_policy', 'actor._optimizer_step'],
            native_optimizer_hook='register_step_post_hook; original step method unchanged')
        write('installation.json', installation)
        state['installation'] = installation
        # Instance-bound wrappers delegate to the captured original bound methods.
        actor.update_policy = observed_update
        actor._optimizer_step = observed_step
        state['optimizer_post_hook'] = optimizer.register_step_post_hook(observed_optimizer_step)
        actor._native_incident_observer_20261010 = state
        reports.append(installation)
    assert len(reports) == 1, reports
    return reports


if __name__ == '__main__':
    import sys
    output = Path(sys.argv[1])
    ray.init(address=sys.argv[2], log_to_driver=False, ignore_reinit_error=False)
    named = [x for x in ray.util.list_named_actors(all_namespaces=True)
             if x['name'].startswith(sys.argv[3]) and 'register_center' not in x['name']]
    assert len(named) == 2, named
    refs = [ray.get_actor(x['name'], namespace=x['namespace']).execute_with_func_generator.remote(install)
            for x in named]
    record = dict(started_unix=time.time(), pid=os.getpid(), birth=psutil.Process().create_time(),
                  named_actors=named, scope=__doc__, complete=False)
    output.write_text(json.dumps(record, indent=2)+'\n')
    try:
        record.update(results=ray.get(refs), complete=True, completed_unix=time.time())
        output.write_text(json.dumps(record, indent=2)+'\n')
    finally:
        ray.shutdown()
