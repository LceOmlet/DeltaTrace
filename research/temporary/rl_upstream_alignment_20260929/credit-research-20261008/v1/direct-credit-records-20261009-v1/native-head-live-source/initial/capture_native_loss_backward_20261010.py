"""Observe the existing loss-to-log-prob gradient without replacing the loss.

The native functions are called once and their exact objects are returned.
Tensor.register_hook returns the same gradient. Only a nonfinite gradient
copies the small loss inputs to CPU; no hidden states or vocabulary tensor is
retained. Weak references prevent the diagnostic from retaining GPU graphs.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import weakref


def observe_call(native, active, callback, on_error):
    signature = inspect.signature(native)

    @functools.wraps(native)
    def observed(*args, **kwargs):
        result = native(*args, **kwargs)
        if active():
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                callback(result, bound.arguments)
            except Exception as error:
                on_error(error)
        return result

    return observed


def observe_gradient(tensor, callback, on_error):
    def observed(gradient):
        try:
            callback(gradient)
        except Exception as error:
            on_error(error)
        return gradient
    return tensor.register_hook(observed)


def install(worker):
    import psutil
    import torch
    from verl.trainer.ppo import core_algos

    pid = os.getpid()
    birth = psutil.Process().create_time()
    assert {987808:1791553850.0,989860:1791553867.51}.get(pid) == birth
    actors = [v.actor for v in worker.worker_dict.values() if getattr(v, '_is_actor', False)]
    assert len(actors) == 1
    actor = actors[0]
    state = actor._native_incident_observer_20261010
    assert not state['active'], 'Install at the existing worker RPC boundary'
    if 'loss_backward_installation' in state:
        return state['loss_backward_installation']
    owner = inspect.getmodule(type(actor))
    owner_path = Path(inspect.getsourcefile(owner))
    assert hashlib.sha256(owner_path.read_bytes()).hexdigest() == state['installation']['actor_sha256']
    core_path = Path(inspect.getsourcefile(core_algos))
    assert hashlib.sha256(core_path.read_bytes()).hexdigest() == 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
    assert owner.compute_policy_loss is core_algos.compute_policy_loss
    assert owner.kl_penalty is core_algos.kl_penalty
    assert actor.config.use_kl_loss and actor.config.kl_loss_type == 'low_var_kl'
    out = Path(state['installation']['out'])
    counters = {}

    def append(value):
        with (out/'loss-backward-events.jsonl').open('a') as stream:
            stream.write(json.dumps(value,allow_nan=True)+'\n')

    def error(error):
        append(dict(event='loss_backward_observer_error',unix=time.time(),pid=pid,error=repr(error)))

    def policy_result(result, arguments):
        tensor = arguments['log_prob']
        if not tensor.requires_grad:
            return
        key = (state['update_index'],state['optimizer_index'])
        index = counters.get(key,0)
        counters.clear()
        counters[key] = index+1
        context = dict(update_index=key[0],optimizer_index=key[1],microbatch_index=index,
            tensors={name:weakref.ref(arguments[name]) for name in
                     ['log_prob','old_log_prob','advantages','response_mask']})
        state['current_loss_backward_context'] = context

        def capture(gradient):
            started = time.perf_counter()
            finite = bool(torch.isfinite(gradient).all().item())
            record = dict(event='native_log_prob_gradient',unix=time.time(),pid=pid,birth=birth,
                update_index=context['update_index'],optimizer_index=context['optimizer_index'],
                microbatch_index=context['microbatch_index'],shape=list(gradient.shape),
                dtype=str(gradient.dtype),finite=finite)
            if not finite:
                values = {}
                missing = []
                for name, reference in context['tensors'].items():
                    value = reference()
                    if value is None:
                        missing.append(name)
                    else:
                        values[name] = value.detach().to('cpu',copy=True)
                values['log_prob_gradient'] = gradient.detach().to('cpu',copy=True)
                record.update(missing_tensors=missing,kl_penalty=context.get('kl_penalty'))
                path = out/('loss-backward-'+str(time.time_ns())+'.pt')
                torch.save(dict(record=record,tensors=values,scope=__doc__),path)
                record.update(path=str(path),bytes=path.stat().st_size)
            record['seconds'] = time.perf_counter()-started
            append(record)
            if state.get('current_loss_backward_context') is context:
                state.pop('current_loss_backward_context')

        # The hook retains only scalar metadata and weak references to tensors.
        observe_gradient(tensor,capture,error)

    def kl_result(result, arguments):
        context = state.get('current_loss_backward_context')
        if context and context['tensors']['log_prob']() is arguments['logprob']:
            context['tensors']['ref_logprob'] = weakref.ref(arguments['ref_logprob'])
            context['kl_penalty'] = arguments['kl_penalty']

    owner.compute_policy_loss = observe_call(owner.compute_policy_loss,lambda:state['active'],policy_result,error)
    owner.kl_penalty = observe_call(owner.kl_penalty,lambda:state['active'],kl_result,error)
    installation = dict(unix=time.time(),pid=pid,birth=birth,
        observer_path=__file__,observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        actor_owner_path=str(owner_path),actor_owner_sha256=hashlib.sha256(owner_path.read_bytes()).hexdigest(),
        loss_owner_path=str(core_path),loss_owner_sha256=hashlib.sha256(core_path.read_bytes()).hexdigest(),
        native_functions=['compute_policy_loss','kl_penalty'],hook='torch.Tensor.register_hook',
        use_kl_loss=bool(actor.config.use_kl_loss),kl_loss_type=str(actor.config.kl_loss_type),
        scope=__doc__,model_calls=0,loss_or_gradient_replacement=False,
        dtype_configuration_or_optimizer_changes=False)
    (out/'loss-backward-installation.json').write_text(json.dumps(installation,indent=2)+'\n')
    state['loss_backward_installation'] = installation
    return installation
