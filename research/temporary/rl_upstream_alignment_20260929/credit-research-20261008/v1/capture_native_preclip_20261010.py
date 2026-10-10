"""Preserve gradients only when the native norm is nonfinite, before clipping.

Calls PyTorch's original _get_total_norm once, returns its exact object, and
does not compute a replacement norm, clipping coefficient, gradient or loss.
The existing passive actor observer supplies the native update/optimizer index.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time


def observe_native_norm(native, active, callback, on_error):
    @functools.wraps(native)
    def observed(*args, **kwargs):
        result = native(*args, **kwargs)
        if active():
            try:
                callback(result, args[0] if args else kwargs['tensors'])
            except Exception as error:
                on_error(error)
        return result
    return observed


def install(worker):
    import math
    import psutil
    import torch
    import torch.nn.utils.clip_grad as owner

    expected = {987808:1791553850.00, 989860:1791553867.51}
    pid = os.getpid()
    assert expected.get(pid) == psutil.Process().create_time()
    actors = [v.actor for v in worker.worker_dict.values() if getattr(v, '_is_actor', False)]
    assert len(actors) == 1
    actor = actors[0]
    state = actor._native_incident_observer_20261010
    actor_source = Path(inspect.getsourcefile(type(actor)))
    assert hashlib.sha256(actor_source.read_bytes()).hexdigest() == state['installation']['actor_sha256']
    assert not state['active'], 'Install only at the native worker RPC boundary'
    out = Path(state['installation']['out'])
    installed = out/'preclip-installation.json'
    if installed.exists():
        assert getattr(owner._get_total_norm, '_native_preclip_observer_20261010', False)
        return json.loads(installed.read_text())
    native = owner._get_total_norm
    native_source = Path(inspect.getsourcefile(native))

    def append(value):
        with (out/'preclip-events.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=True)+'\n')

    def capture(norm, tensors):
        local_norm = norm.to_local() if hasattr(norm, 'to_local') else norm
        value = float(local_norm.detach().item())
        record = dict(event='native_norm_before_clipping',unix=time.time(),pid=pid,
                      update_index=state['update_index'],optimizer_index=state['optimizer_index'],
                      local_total_norm=value,norm_type=type(norm).__name__,
                      norm_placements=str(getattr(norm,'placements',None)))
        if not math.isfinite(value):
            started = time.perf_counter()
            assert isinstance(tensors,(list,tuple)), type(tensors)
            names = {id(p.grad):name for name,p in actor.actor_module.named_parameters() if p.grad is not None}
            grads, statistics = {}, []
            for index, tensor in enumerate(tensors):
                local = tensor.to_local() if hasattr(tensor,'to_local') else tensor
                cpu = local.detach().to('cpu',copy=True)
                grads[index] = cpu
                bad = (~torch.isfinite(cpu)).nonzero()
                if bad.numel():
                    statistics.append(dict(index=index,name=names.get(id(tensor)),
                        shape=list(tensor.shape),local_shape=list(cpu.shape),dtype=str(cpu.dtype),
                        nan_count=int(torch.isnan(cpu).sum()),inf_count=int(torch.isinf(cpu).sum()),
                        first_bad_local_positions=bad[:8].tolist()))
            path = out/('preclip-'+str(time.time_ns())+'-gradients.pt')
            torch.save(dict(record=record,gradients_by_native_index=grads,
                            parameter_names_by_native_index={i:names.get(id(t)) for i,t in enumerate(tensors)},
                            nonfinite_parameters=statistics,scope=__doc__),path)
            record.update(path=str(path),bytes=path.stat().st_size,nonfinite_parameters=statistics,
                          seconds=time.perf_counter()-started)
        append(record)

    def error(error):
        append(dict(event='preclip_observer_error',unix=time.time(),pid=pid,error=repr(error)))

    wrapped = observe_native_norm(native,lambda:state['active'],capture,error)
    wrapped._native_preclip_observer_20261010 = True
    owner._get_total_norm = wrapped
    receipt = dict(unix=time.time(),pid=pid,birth=psutil.Process().create_time(),
        observer_path=__file__,observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        owner_path=str(native_source),owner_sha256=hashlib.sha256(native_source.read_bytes()).hexdigest(),
        original_function=native.__qualname__,scope=__doc__,new_model_calls=0,
        norm_or_gradient_replacement=False,clipping_or_optimizer_changes=False)
    installed.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
