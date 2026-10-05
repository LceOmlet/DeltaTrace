"""Bounded layer3/case0 operand capture around the unchanged native owners.

Save the last native replay's paired rows0/1 and the original two finite norm
calls' case0 operands. No norm, residual addition, finite formula, precision,
prefix, or model call is replaced. Copies and serialization are diagnostics;
this context is not a performance measurement or an accuracy threshold.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch


def _source(function):
    function = inspect.unwrap(function)
    path = Path(inspect.getsourcefile(function)).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                function=function.__name__, first_line=function.__code__.co_firstlineno)


def _arg(args, kwargs, name, index):
    return kwargs[name] if name in kwargs else args[index]


@contextmanager
def observe_norm_operands(runner, artifact_path):
    """Yield a receipt; write .pt/.json only after restoring the original hooks.

    artifact_path is supplied by the caller, e.g. rank1-norm-operands.pt. Wrap
    the original trace_token_attribution invocation without its observer or
    diagnostics flag. The .pt payload preserves each captured tensor's dtype.
    """
    artifact_path = Path(artifact_path)
    module = sys.modules[type(runner).__module__]
    original_decoder = module.decoder_finite_pullback
    layer = runner.model.model.language_model.layers[3]
    receipt = dict(scope=__doc__, layer_index=3, case_index=0, paired_rows=[0, 1],
        owner_decoder=_source(original_decoder), decoder_calls=0,
        native_captures={}, norm_calls=[], diagnostic_errors=[],
        artifact_path=str(artifact_path),
        retained_payload='Native two-row clones and finite case0 clones only; no full B8 tensor storage.')
    native = {}
    calls = []
    handles = []

    def clone(value, rows):
        # to_local is the same public local-weight read used by the actual HF
        # RMSNorm owner; it does not gather a sharded parameter or change it.
        local = value.to_local() if hasattr(value, 'to_local') else value
        selected = local if rows is None else local[:rows]
        result = selected.detach().clone()
        meta = dict(original_type=type(value).__name__, original_shape=list(value.shape),
            original_dtype=str(value.dtype), original_device=str(value.device),
            original_stride=list(local.stride()), original_storage_offset=local.storage_offset(),
            local_shape=list(local.shape), retained_shape=list(result.shape),
            retained_dtype=str(result.dtype), retained_stride=list(result.stride()),
            retained_bytes=result.numel()*result.element_size())
        if hasattr(value, 'placements'):
            meta['original_placements'] = [str(x) for x in value.placements]
        return result, meta

    def capture(name, *, input_name=None):
        def hook(_module, args, output):
            try:
                value = output if isinstance(output, torch.Tensor) else output[0]
                saved, meta = clone(value, 2)
                native[name] = saved
                prior = receipt['native_captures'].get(name, {})
                receipt['native_captures'][name] = dict(meta, calls=prior.get('calls', 0)+1)
                if input_name is not None:
                    saved, meta = clone(args[0], 2)
                    native[input_name] = saved
                    prior = receipt['native_captures'].get(input_name, {})
                    receipt['native_captures'][input_name] = dict(meta, calls=prior.get('calls', 0)+1)
            except Exception as exc:
                receipt['diagnostic_errors'].append(dict(where='native_hook_'+name, error=repr(exc)))
        return hook

    def observed_decoder(*args, **kwargs):
        if _arg(args, kwargs, 'layer', 0) is not layer:
            return original_decoder(*args, **kwargs)
        boundaries = _arg(args, kwargs, 'boundaries', 4)
        original_norm = boundaries.norm_residual
        receipt['decoder_calls'] += 1
        norm_count = 0

        def norm_residual(*inner_args, **inner_kwargs):
            nonlocal norm_count
            # Invoke the same compiled owner with the same objects and args.
            result = original_norm(*inner_args, **inner_kwargs)
            index = norm_count
            norm_count += 1
            try:
                name = ('post_attention_norm', 'input_norm')[index]
                data = dict(name=name, original_call_index=index,
                    eps=_arg(inner_args, inner_kwargs, 'eps', 5))
                metadata = dict(name=name, original_call_index=index, eps=data['eps'], tensors={})
                for field, arg_index, rows in (('x0', 0, 1), ('x1', 1, 1),
                        ('raw_weight', 2, None), ('upstream', 3, 1), ('residual', 4, 1)):
                    saved, meta = clone(_arg(inner_args, inner_kwargs, field, arg_index), rows)
                    data[field] = saved
                    metadata['tensors'][field] = meta
                data['result'], metadata['tensors']['result'] = clone(result, 1)
                calls.append(data)
                receipt['norm_calls'].append(metadata)
            except Exception as exc:
                receipt['diagnostic_errors'].append(dict(where='finite_norm_capture',
                    call_index=index, error=repr(exc)))
            return result

        try:
            boundaries.norm_residual = norm_residual
            return original_decoder(*args, **kwargs)
        finally:
            boundaries.norm_residual = original_norm

    try:
        for name, target, input_name in (
                ('input_norm_output', layer.input_layernorm, 'input_norm_input'),
                ('post_norm_output', layer.post_attention_layernorm, 'post_norm_input'),
                ('attention_output', layer.self_attn, None),
                ('mlp_output', layer.mlp, None),
                ('decoder_output', layer, None)):
            handles.append(target.register_forward_hook(capture(name, input_name=input_name)))
        module.decoder_finite_pullback = observed_decoder
        yield receipt
    finally:
        module.decoder_finite_pullback = original_decoder
        for handle in handles:
            handle.remove()
        try:
            # These are the actual HF addition's branches and stored outputs,
            # not additions reconstructed by the diagnostic.
            additions = {}
            for name, fields in (
                    ('attention_residual_add', ('input_norm_input', 'attention_output', 'post_norm_input')),
                    ('mlp_residual_add', ('post_norm_input', 'mlp_output', 'decoder_output'))):
                if all(field in native for field in fields):
                    additions[name] = dict(left=native[fields[0]], right=native[fields[1]],
                        output=native[fields[2]], original_fields=list(fields))
                else:
                    receipt['diagnostic_errors'].append(dict(where=name,
                        error='Missing actual native output', missing=[f for f in fields if f not in native]))
            memo = {}
            def cpu(value):
                if isinstance(value, torch.Tensor):
                    key = id(value)
                    if key not in memo:
                        memo[key] = value.cpu()
                    return memo[key]
                if isinstance(value, dict):
                    return {k: cpu(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [cpu(v) for v in value]
                return value
            payload = cpu(dict(native=native, finite_norm_calls=calls,
                residual_additions=additions, receipt=receipt))
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, artifact_path)
            receipt.update(artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                artifact_bytes=artifact_path.stat().st_size,
                receipt_path=str(artifact_path.with_suffix('.json')))
            artifact_path.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
        except Exception as exc:
            receipt['diagnostic_errors'].append(dict(where='serialize_operands', error=repr(exc)))
        finally:
            native.clear()
            calls.clear()
