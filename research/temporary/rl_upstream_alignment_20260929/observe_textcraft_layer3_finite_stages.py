"""Read-only layer3/case0 observation around the original finite operators.

Native forward hooks retain only the last replay's paired rows0/1. The original
decoder and boundary callables receive their original arguments and return
their exact original objects. Supplemental contractions call the owner's own
_token_effect; they are debugging work, not its normal98-call ledger or a
performance measurement. No observer/diagnostics flag or precision is changed.
"""
from contextlib import contextmanager
import hashlib
import inspect
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
def observe_layer3_finite_stages(runner):
    """Yield only small scalar/metadata records for one original attribute call.

    Wrap the existing trace_token_attribution call in this context. Do not pass
    attribute observer or diagnostics=True. Read the report after context exit.
    All module hooks and original boundary callables are restored in finally.
    """
    module = sys.modules[type(runner).__module__]
    original_decoder = module.decoder_finite_pullback
    token_effect = module._token_effect
    layer = runner.model.model.language_model.layers[3]
    report = dict(scope=__doc__, layer_index=3, case_index=0, paired_rows=[0, 1],
        owner_decoder=_source(original_decoder), owner_token_effect=_source(token_effect),
        block_type=layer.block_type, decoder_calls=0, native_captures={},
        stages=[], diagnostic_errors=[],
        retained_payload='Only two actual native endpoint rows per module; tensors never enter the report.')
    snapshots = {}
    sums = []
    handles = []

    def capture(name):
        def hook(_module, _args, output):
            try:
                value = output if isinstance(output, torch.Tensor) else output[0]
                # clone is deliberate: a slice alone would retain the full
                # B8 storage. Prefix/root captures are overwritten by replay.
                snapshots[name] = value.detach()[:2].clone()
                prior = report['native_captures'].get(name, {})
                report['native_captures'][name] = dict(calls=prior.get('calls', 0)+1,
                    original_output_shape=list(value.shape), dtype=str(value.dtype),
                    retained_shape=list(snapshots[name].shape),
                    retained_bytes=snapshots[name].numel()*snapshots[name].element_size())
            except Exception as exc:
                report['diagnostic_errors'].append(dict(where='native_hook_'+name, error=repr(exc)))
        return hook

    def effect(name, coefficients, endpoints):
        try:
            # Exactly the owner's contraction on failed case0's coefficient
            # and its actual paired0/1 endpoints; no alternate finite rule.
            value = token_effect(coefficients[:1], endpoints[:2])
            scalar = value.detach().sum()
            report['stages'].append(dict(name=name,
                coefficient_dtype=str(coefficients.dtype), endpoint_dtype=str(endpoints.dtype),
                owner_contraction_dtype=str(value.dtype),
                original_coefficient_shape=list(coefficients.shape),
                contracted_coefficient_shape=list(coefficients[:1].shape),
                contracted_endpoint_shape=list(endpoints[:2].shape)))
            sums.append(scalar)
        except Exception as exc:
            report['diagnostic_errors'].append(dict(where='effect_'+name, error=repr(exc)))

    def native(name):
        value = snapshots.get(name)
        if value is None:
            report['diagnostic_errors'].append(dict(where='native_snapshot_'+name,
                error='The original module output has not been observed.'))
        return value

    def native_effect(name, coefficients, field):
        endpoints = native(field)
        if endpoints is not None:
            effect(name, coefficients, endpoints)

    def observed_decoder(*args, **kwargs):
        current = _arg(args, kwargs, 'layer', 0)
        if current is not layer:
            return original_decoder(*args, **kwargs)
        values = _arg(args, kwargs, 'values', 1)
        upstream = _arg(args, kwargs, 'upstream', 2)
        boundaries = _arg(args, kwargs, 'boundaries', 4)
        report['decoder_calls'] += 1
        originals = {name: getattr(boundaries, name)
                     for name in ('mlp', 'norm_residual', 'attention_input')}
        norm_calls = 0
        native_effect('decoder.native_output', upstream, 'decoder_output')
        native_effect('decoder.native_mlp_branch', upstream, 'mlp_output')
        effect('decoder.native_residual_branch', upstream, values['post_norm_input'])

        def mlp(*inner_args, **inner_kwargs):
            result = originals['mlp'](*inner_args, **inner_kwargs)
            actual_upstream = _arg(inner_args, inner_kwargs, 'upstream', 6)
            native_effect('mlp.output_signal', actual_upstream, 'mlp_output')
            native_effect('mlp.input_signal', result, 'post_norm_output')
            return result

        def norm_residual(*inner_args, **inner_kwargs):
            nonlocal norm_calls
            result = originals['norm_residual'](*inner_args, **inner_kwargs)
            # These are the original decoder's two observed norm calls in its
            # original order, not a replacement composition or norm formula.
            index = norm_calls
            norm_calls += 1
            actual_upstream = _arg(inner_args, inner_kwargs, 'upstream', 3)
            residual = _arg(inner_args, inner_kwargs, 'residual', 4)
            if index == 0:
                native_effect('post_norm.output_signal', actual_upstream, 'post_norm_output')
                effect('post_norm.residual_signal', residual, values['post_norm_input'])
                effect('post_norm.input_total', result, values['post_norm_input'])
                native_effect('attention.output_signal', result, 'attention_output')
            elif index == 1:
                native_effect('input_norm.output_signal', actual_upstream, 'input_norm_output')
                effect('input_norm.residual_signal', residual, values['input_norm_input'])
                effect('input_norm.input_total', result, values['input_norm_input'])
            else:
                report['diagnostic_errors'].append(dict(where='norm_residual',
                    error='Additional original norm call observed', call_index=index))
            return result

        def attention_input(*inner_args, **inner_kwargs):
            result = originals['attention_input'](*inner_args, **inner_kwargs)
            native_effect('attention.input_signal', result, 'input_norm_output')
            return result

        try:
            boundaries.mlp = mlp
            boundaries.norm_residual = norm_residual
            boundaries.attention_input = attention_input
            result = original_decoder(*args, **kwargs)
            effect('decoder.input_total', result[0], values['input_norm_input'])
            return result
        finally:
            for name, function in originals.items():
                setattr(boundaries, name, function)

    try:
        for name, target in (('mlp_output', layer.mlp),
                             ('input_norm_output', layer.input_layernorm),
                             ('post_norm_output', layer.post_attention_layernorm),
                             ('attention_output', layer.self_attn),
                             ('decoder_output', layer)):
            handles.append(target.register_forward_hook(capture(name)))
        module.decoder_finite_pullback = observed_decoder
        yield report
    finally:
        module.decoder_finite_pullback = original_decoder
        for handle in handles:
            handle.remove()
        try:
            if sums:
                for row, value in zip(report['stages'], torch.stack(sums).cpu().tolist()):
                    row['failed_sample_effect'] = value
        except Exception as exc:
            report['diagnostic_errors'].append(dict(where='resolve_small_sums', error=repr(exc)))
        finally:
            sums.clear()
            snapshots.clear()
