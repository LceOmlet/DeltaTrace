"""Prepared-only passive metadata for one existing native DT replay path.

No forward is added, chunked or replaced. The owner's replay_finite_layer is
called once with its original arguments. Tensor objects never enter records.
CUDA allocator counts below are virtual allocator observations, not mx-smi
physical VRAM. This module does not use the runner's observer switch.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from types import MethodType


def _tensor_metadata(value):
    result = dict(shape=[int(v) for v in value.shape], dtype=str(value.dtype),
                  device=str(value.device), stride=[int(v) for v in value.stride()],
                  storage_offset=int(value.storage_offset()),
                  logical_bytes=int(value.numel() * value.element_size()))
    try:
        storage = value.untyped_storage()
        result.update(storage_bytes=int(storage.nbytes()),
                      storage_data_ptr=int(storage.data_ptr()))
    except Exception as exc:
        result['storage_metadata_error'] = f'{type(exc).__name__}: {exc}'
    return result


def _metadata(value, torch):
    if isinstance(value, torch.Tensor):
        return _tensor_metadata(value)
    if isinstance(value, (tuple, list)):
        return [_metadata(item, torch) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata(item, torch) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {'python_type': f'{type(value).__module__}.{type(value).__qualname__}'}


def _allocator(torch):
    result = dict(scope='torch virtual allocator; not mx-smi physical VRAM',
                  allocated_bytes=None, reserved_bytes=None)
    try:
        if torch.cuda.is_initialized():
            result.update(allocated_bytes=int(torch.cuda.memory_allocated()),
                          reserved_bytes=int(torch.cuda.memory_reserved()))
    except Exception as exc:
        result['metadata_error'] = f'{type(exc).__name__}: {exc}'
    return result


def _layer_identity(model, layer):
    result = dict(object_id=id(layer),
                  python_type=f'{type(layer).__module__}.{type(layer).__qualname__}',
                  block_type=getattr(layer, 'block_type', None), layer_index=None)
    language_model = getattr(getattr(model, 'model', None), 'language_model', None)
    for index, candidate in enumerate(getattr(language_model, 'layers', ())):
        if candidate is layer:
            result['layer_index'] = index
            break
    return result


def _targets(layer):
    mlp = layer.mlp
    yield 'mlp', mlp
    for name, module in (('gate', mlp.gate_proj), ('up', mlp.up_proj),
                         ('silu', mlp.act_fn), ('down', mlp.down_proj)):
        yield name, module
        base_layer = getattr(module, 'base_layer', None)
        if base_layer is not None:
            yield f'{name}.base_layer', base_layer
        for family in ('lora_A', 'lora_B'):
            adapters = getattr(module, family, None)
            if adapters is not None:
                for adapter, child in adapters.items():
                    yield f'{name}.{family}.{adapter}', child


@contextmanager
def passive_native_mlp(model, jsonl_path, *, source_identity=None, torch_module=None):
    """Observe original replays during this temporary context only.

    ``source_identity`` is copied through JSON, so it cannot retain caller
    tensors. The yielded state holds only counts and observation I/O errors.
    File/hook observation errors are reported without replacing an original
    return value or exception. This is not a numerical or capacity validator.
    """
    if torch_module is None:
        import torch as torch_module
    identity = json.loads(json.dumps(source_identity or {}))
    original = model.replay_finite_layer
    missing = object()
    original_instance_binding = vars(model).get('replay_finite_layer', missing)
    state = dict(replay_calls=0, records=0, observation_errors=[])
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', buffering=1) as stream:
        def emit(phase, **payload):
            record = dict(phase=phase, unix=time.time(), monotonic=time.perf_counter(),
                          source_identity=identity, **payload,
                          allocator=_allocator(torch_module))
            try:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                state['records'] += 1
            except Exception as exc:
                message = f'{type(exc).__name__}: {exc}'
                state['observation_errors'].append(message)
                try:
                    print(f'[passive native MLP observation error] {message}', file=sys.stderr)
                except Exception:
                    pass

        def observed(_self, layer, replay, *args, **kwargs):
            state['replay_calls'] += 1
            call = state['replay_calls']
            layer_info = _layer_identity(model, layer)
            handles = []
            common = dict(replay_call=call, layer=layer_info)
            emit('native_mlp_replay_enter', **common)

            def attach(name, module):
                module_info = dict(object_id=id(module),
                    python_type=f'{type(module).__module__}.{type(module).__qualname__}')

                def before(_module, inputs, keywords):
                    try:
                        emit('native_mlp_module_pre', **common, module=name,
                             module_identity=module_info,
                             inputs=_metadata(inputs, torch_module),
                             keywords=_metadata(keywords, torch_module))
                    except Exception as exc:
                        emit('native_mlp_metadata_error', **common, module=name,
                             error=f'{type(exc).__name__}: {exc}')

                def after(_module, inputs, output):
                    try:
                        emit('native_mlp_module_post', **common, module=name,
                             module_identity=module_info,
                             output=_metadata(output, torch_module))
                    except Exception as exc:
                        emit('native_mlp_metadata_error', **common, module=name,
                             error=f'{type(exc).__name__}: {exc}')

                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after))

            try:
                try:
                    for name, module in _targets(layer):
                        attach(name, module)
                except Exception as exc:
                    emit('native_mlp_hook_setup_error', **common,
                         error=f'{type(exc).__name__}: {exc}')
                result = original(layer, replay, *args, **kwargs)
                try:
                    output = _metadata(result, torch_module)
                except Exception as exc:
                    output = {'metadata_error': f'{type(exc).__name__}: {exc}'}
                emit('native_mlp_replay_return', **common, output=output)
                return result
            except BaseException as exc:
                emit('native_mlp_replay_exception', **common,
                     exception_type=f'{type(exc).__module__}.{type(exc).__qualname__}',
                     exception_message=str(exc))
                raise
            finally:
                for handle in reversed(handles):
                    try:
                        handle.remove()
                    except Exception as exc:
                        emit('native_mlp_hook_remove_error', **common,
                             error=f'{type(exc).__name__}: {exc}')
                handles.clear()
                emit('native_mlp_replay_hooks_removed', **common)

        model.replay_finite_layer = MethodType(observed, model)
        try:
            emit('native_mlp_observer_installed',
                 scope='Temporary passive observation; no numerical/capacity claim')
            yield state
        finally:
            if original_instance_binding is missing:
                delattr(model, 'replay_finite_layer')
            else:
                model.replay_finite_layer = original_instance_binding
            emit('native_mlp_observer_binding_restored')
