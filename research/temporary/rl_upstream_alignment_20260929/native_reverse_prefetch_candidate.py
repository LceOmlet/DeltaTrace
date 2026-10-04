"""Diagnostic-only composition of the installed FSDP2 forward-prefetch API.

No forward, collective, finite rule, parameter dtype, or offload implementation
is supplied here. Enter around a complete original DT attribute call; the
temporary setting is active only inside each original reverse layer replay.
"""
import dataclasses
import inspect
import time
import types

import torch


def _group(module):
    return module._get_fsdp_state()._fsdp_param_group


def _tensor_records(value, path, records, seen):
    """Read existing owner tensors; never copy or keep their storage alive."""
    if isinstance(value, torch.Tensor):
        local = getattr(value, '_local_tensor', value)
        storage = local.untyped_storage()
        key = (str(local.device), storage.data_ptr(), storage.nbytes())
        alias = key in seen
        seen.add(key)
        records.append(dict(path=path, shape=list(value.shape), local_shape=list(local.shape),
                            dtype=str(local.dtype), device=str(local.device),
                            tensor_bytes=local.numel()*local.element_size(),
                            storage_data_ptr=storage.data_ptr(),
                            storage_bytes=storage.nbytes(), storage_alias=alias))
    elif isinstance(value, dict):
        for key, item in value.items():
            _tensor_records(item, f'{path}.{key}', records, seen)
    elif isinstance(value, (list, tuple)):
        names = getattr(value, '_fields', None)
        for index, item in enumerate(value):
            _tensor_records(item, f'{path}.{names[index] if names else index}', records, seen)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _tensor_records(getattr(value, field.name), f'{path}.{field.name}', records, seen)


def _buffers(module):
    group = _group(module)
    records, seen = [], set()
    _tensor_records(group._all_gather_result, 'pending_all_gather_result', records, seen)
    for index, param in enumerate(group.fsdp_params):
        _tensor_records(param.all_gather_outputs, f'param_{index}.all_gather_outputs', records, seen)
    _tensor_records(getattr(group.comm_ctx, 'all_gather_state', None),
                    'shared_comm_all_gather_state', records, seen)
    return dict(pending=group._all_gather_result is not None, tensors=records,
                unique_storage_bytes=sum(r['storage_bytes'] for r in records if not r['storage_alias']),
                scope='Existing owner all-gather tensors only; not allocator or physical peak memory')


def _state_names(states):
    return [getattr(state._fsdp_param_group, '_module_fqn', None) for state in states]


class NativeReversePrefetch:
    """Temporarily prefetch one next reverse layer using its original owner.

    Usage: ``with NativeReversePrefetch(runner.model) as candidate: ...``.
    ``candidate.report()`` is JSON-serializable and retains no tensor references.
    The context itself does not run a model or issue any collective.
    """
    def __init__(self, model):
        self.model = model
        self.layers = list(model.model.language_model.layers)
        self._indices = {id(layer): index for index, layer in enumerate(self.layers)}
        self._started_targets = {}
        self._active = False
        self._report = dict(candidate='official_FSDP2_single_next_reverse_prefetch',
                           instrumented=True, status='not_entered', calls=[], cleanup=[],
                           owner_interface='FSDPModule.set_modules_to_forward_prefetch',
                           unchanged=['original replay callback', 'finite rules', 'reshard policy',
                                      'mixed precision', 'CPUOffloadPolicy', 'unshard_async_op'],
                           scope='Diagnostic only; no numerical tolerance or speed acceptance claim')

    def report(self):
        return self._report

    def __enter__(self):
        if self._active:
            raise RuntimeError('NativeReversePrefetch is already active')
        tick = time.perf_counter()
        distributed = torch.distributed
        self._report['rank'] = distributed.get_rank() if distributed.is_initialized() else None
        topology = []
        for index, layer in enumerate(self.layers):
            if not callable(getattr(layer, 'set_modules_to_forward_prefetch', None)):
                raise TypeError(f'Actual layer {index} has no original FSDP forward-prefetch interface')
            group = _group(layer)
            if group is None:
                raise TypeError(f'Actual layer {index} has no FSDP parameter group')
            state = layer._get_fsdp_state()
            setter = layer.set_modules_to_forward_prefetch
            topology.append(dict(layer=index, module_type=f'{type(layer).__module__}.{type(layer).__name__}',
                                 group_fqn=group._module_fqn,
                                 owner_setter_source=inspect.getsourcefile(setter),
                                 original_forward_prefetch=_state_names(state._states_to_forward_prefetch),
                                 unshard_async_op=group.unshard_async_op,
                                 offload_policy_type=type(group.offload_policy).__name__,
                                 parameters=[dict(name=getattr(param, '_param_fqn', None),
                                                  original_shape=list(param._orig_size),
                                                  padded_sharded_shape=list(param.padded_sharded_param_size),
                                                  original_dtype=str(getattr(param, 'orig_dtype', None)),
                                                  param_dtype=str(getattr(param, 'param_dtype', None)))
                                             for param in group.fsdp_params],
                                 buffers_at_enter=_buffers(layer)))
        self._report['topology'] = topology
        self._original_replay = self.model.replay_finite_layer
        self._had_instance_replay = 'replay_finite_layer' in vars(self.model)
        self._instance_replay = vars(self.model).get('replay_finite_layer')
        original = self._original_replay

        def replay(_model, layer, callback):
            index = self._indices[id(layer)]
            target = self.layers[index-1] if index else None
            state = layer._get_fsdp_state()
            saved_list = state._states_to_forward_prefetch
            before = _buffers(target) if target is not None else None
            row = dict(layer=index, next_reverse_layer=index-1 if index else None,
                       rank=self._report['rank'], prior_forward_prefetch=_state_names(saved_list),
                       current_before_replay=_buffers(layer), target_before=before,
                       configured=False, restored=False)
            self._report['calls'].append(row)
            def after_owner_pre_forward(_layer, _args):
                current_pending = _group(layer)._all_gather_result is not None
                row['after_owner_pre_forward'] = dict(
                    current_pending=current_pending,
                    current_pending_handle_consumed=(row['current_before_replay']['pending'] and not current_pending),
                    next_reverse_layer_pending=(None if target is None else
                                                _group(target)._all_gather_result is not None),
                    scope='Original FSDP pre-hook has run; pending metadata only, not a device synchronization')
            observation_handle = None
            try:
                layer.set_modules_to_forward_prefetch([target] if target is not None else [])
                row['configured'] = True
                row['actual_forward_prefetch'] = _state_names(state._states_to_forward_prefetch)
                # Original FSDP pre-hook uses prepend=True; this passive hook
                # observes its consumed current result and started next result.
                observation_handle = layer.register_forward_pre_hook(after_owner_pre_forward)
                return original(layer, callback)
            finally:
                if observation_handle is not None:
                    observation_handle.remove()
                # Public setter has no getter. Restore the exact prior owner
                # list object, including settings supplied by another caller.
                state._states_to_forward_prefetch = saved_list
                row['restored'] = state._states_to_forward_prefetch is saved_list
                row['current_after_replay'] = _buffers(layer)
                if target is not None:
                    after = _buffers(target)
                    row['target_after_replay'] = after
                    if not before['pending'] and after['pending']:
                        self._started_targets[index-1] = target

        self.model.replay_finite_layer = types.MethodType(replay, self.model)
        self._active = True
        self._report['status'] = 'active'
        self._report['enter_metadata_seconds'] = time.perf_counter()-tick
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._had_instance_replay:
            self.model.replay_finite_layer = self._instance_replay
        else:
            delattr(self.model, 'replay_finite_layer')
        self._report['replay_method_restored'] = self.model.replay_finite_layer == self._original_replay
        self._active = False
        pending = [(index, layer) for index, layer in self._started_targets.items()
                   if _group(layer)._all_gather_result is not None]
        self._report['pending_started_targets_at_exit'] = [index for index, _ in pending]
        cleanup_errors = []
        for index, layer in pending:
            row = dict(layer=index, before=_buffers(layer),
                       owner_calls=['FSDPModule.unshard()', 'FSDPModule.reshard()'])
            self._report['cleanup'].append(row)
            try:
                # Original public unshard waits/copies out an already pending
                # result. Reshard alone would not consume that result.
                layer.unshard()
                layer.reshard()
                row['after'] = _buffers(layer)
                row['completed'] = not row['after']['pending']
                if not row['completed']:
                    raise RuntimeError(f'Original owner left layer {index} pending after cleanup')
            except BaseException as error:
                row['completed'] = False
                row['error'] = f'{type(error).__name__}: {error}'
                cleanup_errors.append(error)
        self._report['pending_all_layers_after_exit'] = [
            index for index, layer in enumerate(self.layers)
            if _group(layer)._all_gather_result is not None]
        self._report['exception'] = None if exc is None else f'{type(exc).__name__}: {exc}'
        self._report['status'] = 'exception' if exc is not None else (
            'unexpected_pending' if pending or self._report['pending_all_layers_after_exit'] else 'completed')
        if exc is None:
            if cleanup_errors:
                raise cleanup_errors[0]
            if pending or self._report['pending_all_layers_after_exit']:
                raise RuntimeError('Complete reverse replay ended with pending owner all-gathers; see report')
        return False
