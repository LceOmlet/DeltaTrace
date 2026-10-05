"""Passive GDN mixer projections on the existing matched single-EOS roots.

Compose with the existing primitive contexts. Save only actual full-EOS
norm-gate coefficients for original selected slots, then project them against
single-EOS native norm and mixer inputs/outputs. Three scalar differences
telescope to the existing primitive mixer difference. This observation adds
no model call, finite rule, tolerance, correction, or recurrent-state capture.
"""
from contextlib import contextmanager, ExitStack
import inspect
import sys

import torch

from observe_textcraft_conditional_boundaries import (
    _attribute_metadata, _slots, _source, _tensor_metadata,
)
from observe_textcraft_conditional_primitives import _effect, _layers
from observe_textcraft_norm_operands import _arg


DEFAULT_LAYERS = (6, 8)


def _actual_capture_start(caller, owner_code):
    """Read only the original GDN call's parameter, retaining no frame."""
    current = caller
    try:
        while current is not None and current.f_code is not owner_code:
            current = current.f_back
        if current is None:
            raise ValueError('The original GDN pullback frame is absent.')
        return int(current.f_locals['capture_start'])
    finally:
        del current, caller


@contextmanager
def collect_actual_gdn(runner, bank, slots, *, layers=DEFAULT_LAYERS):
    """Nest inside collect_actual_primitives on the one original full trace.

    The original decoder and compiled gdn_norm_gate are called first; their
    exact result objects are returned. The caller clears gdn_bank after the
    two original single-root probes, together with the primitive bank.
    """
    slots = _slots(slots)
    if slots != bank['slots']:
        raise ValueError('Use the original coefficient-bank slot order.')
    indices, owner_layers = _layers(runner, layers)
    if any(owner_layers[i].block_type != 'linear_attention' for i in indices):
        raise ValueError('The requested original layers must be GDN layers.')
    if 'gdn_bank' in bank:
        raise ValueError('Do not overwrite an existing GDN observation bank.')
    module = sys.modules[type(runner).__module__]
    original_decoder = module.decoder_finite_pullback
    original_gdn = module.gdn_finite_pullback
    gdn_code = inspect.unwrap(original_gdn).__code__
    owner_copy = module._copy
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    gdn_bank = bank['gdn_bank'] = {}
    receipt = bank['metadata']['conditional_gdn'] = dict(
        scope=__doc__, layer_indices=list(indices), original_slots=list(slots),
        original_decoder=_source(original_decoder), original_gdn_pullback=_source(original_gdn),
        owner_copy=_source(owner_copy),
        actual_compile_gdn_scalar_rules=runner.compile_gdn_scalar_rules,
        layers={}, diagnostics=[],
        coefficient_scope='Actual joint full-EOS m/mo/mz before the original native_mo dtype conversion.',
        lifetime='CPU selected-slot bank for this one original B4 group only.')
    layer_indices = {id(owner_layers[i]): i for i in indices}

    def observed_decoder(*args, **kwargs):
        layer = _arg(args, kwargs, 'layer', 0)
        index = layer_indices.get(id(layer))
        if index is None:
            return original_decoder(*args, **kwargs)
        boundaries = _arg(args, kwargs, 'boundaries', 4)
        original_gate = boundaries.gdn_norm_gate
        entry = dict(slots={slot: {} for slot in slots})
        metadata = dict(decoder_index=index, block_type=layer.block_type,
                        norm_gate_calls=0, actual_norm_gate=_source(original_gate),
                        coefficients={}, diagnostics=[])

        def gate(*inner_args, **inner_kwargs):
            result = original_gate(*inner_args, **inner_kwargs)
            metadata['norm_gate_calls'] += 1
            try:
                coordinates = _attribute_metadata(sys._getframe(1), attribute_code)
                if coordinates != bank['metadata'].get('coordinates'):
                    raise ValueError('The original full-EOS GDN layout changed.')
                m = _arg(inner_args, inner_kwargs, 'm', 2)
                rule = _arg(inner_args, inner_kwargs, 'norm_gate_rule', 5)
                mo, mz = result
                mixer = layer.linear_attn
                capture_start = _actual_capture_start(sys._getframe(1), gdn_code)
                coefficient_length = coordinates['suffix_length']-capture_start
                shape = (4, coefficient_length, mixer.num_v_heads, mixer.head_v_dim)
                if any(tuple(value.shape) != shape for value in (m, mo, mz)):
                    raise ValueError('Actual norm-gate coefficients differ from the original compact B4 suffix.')
                if any(entry['slots'][slot] for slot in slots):
                    raise ValueError('A second norm-gate call would overwrite original coefficients.')
                entry['coordinates'] = coordinates
                entry['capture_start'] = capture_start
                metadata['actual_norm_gate_rule'] = rule
                metadata['capture_start'] = capture_start
                metadata['native_suffix_length'] = coordinates['suffix_length']
                metadata['coefficient_suffix_length'] = coefficient_length
                metadata['actual_native_o'] = _tensor_metadata(_arg(inner_args, inner_kwargs, 'o', 0))
                metadata['actual_native_z'] = _tensor_metadata(_arg(inner_args, inner_kwargs, 'z', 1))
                for name, value in (('m', m), ('mo', mo), ('mz', mz)):
                    metadata['coefficients'][name] = dict(original=_tensor_metadata(value), slots={})
                    for slot in slots:
                        saved = owner_copy(value[slot:slot+1].flatten(2), 'cpu', pinned_host=False)
                        entry['slots'][slot][name] = saved
                        metadata['coefficients'][name]['slots'][str(slot)] = _tensor_metadata(saved)
            except Exception as exc:
                metadata['diagnostics'].append(dict(where='norm_gate_coefficients', error=repr(exc)))
            return result

        try:
            boundaries.gdn_norm_gate = gate
            result = original_decoder(*args, **kwargs)
            if index in gdn_bank:
                receipt['diagnostics'].append(dict(where='duplicate_decoder', layer=index))
            else:
                gdn_bank[index] = entry
                receipt['layers'][str(index)] = metadata
            return result
        finally:
            boundaries.gdn_norm_gate = original_gate

    module.decoder_finite_pullback = observed_decoder
    try:
        yield receipt
    finally:
        module.decoder_finite_pullback = original_decoder
        receipt['captured_layer_indices'] = sorted(gdn_bank)
        receipt['retained_cpu_tensor_bytes'] = sum(
            value.numel()*value.element_size() for entry in gdn_bank.values()
            for fields in entry['slots'].values() for value in fields.values())


@contextmanager
def contract_saved_native_gdn(runner, bank, slots):
    """Nest inside contract_saved_native_primitives, outside the root observer.

    Prefix B4 is ignored. Only the original paired8 suffix root installs the
    hooks; the original score callback is called first, and its returned
    object is unchanged. At that score boundary, captures are closed and the
    original CPU _token_effect produces scalars; all activations are released.
    """
    slots = _slots(slots)
    if slots != bank['slots']:
        raise ValueError('Use the original coefficient-bank slot order.')
    gdn_bank = bank['gdn_bank']
    primitive_bank = bank['primitive_bank']
    module = sys.modules[type(runner).__module__]
    original_score = module.selected_target_log_probs
    owner_copy, owner_effect = module._copy, module._token_effect
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    owner_layers = runner.model.model.language_model.layers
    report = dict(scope=__doc__, layer_indices=sorted(gdn_bank), original_slots=list(slots),
        wrapped_existing_score=_source(original_score), owner_token_effect=_source(owner_effect),
        actual_compile_gdn_scalar_rules=runner.compile_gdn_scalar_rules,
        paired_root_capture_entries=0, ignored_prefix_entries=0, score_calls=0,
        capture_metadata={}, layers=[], diagnostics=[],
        three_terms=['out_proj', 'norm_gate', 'remaining_input_FLA'],
        interpretation='Descriptive arithmetic identity. remaining_input_FLA includes original convolution, projections, FLA, and storage; it is not an isolated FLA kernel error.')
    native = {}
    mixer_effects = {}
    stack = ExitStack()
    active = False
    closed = False

    def cleanup():
        nonlocal active, closed
        try:
            stack.close()
        except Exception as exc:
            report['diagnostics'].append(dict(where='hook_cleanup', error=repr(exc)))
        finally:
            active = False
            closed = True

    def begin_root(_module, args, kwargs):
        nonlocal active
        ids = kwargs.get('input_ids', args[0] if args else None)
        if not isinstance(ids, torch.Tensor):
            report['diagnostics'].append(dict(where='native_call', error='No input tensor.'))
            return
        if ids.shape[0] == 4:
            report['ignored_prefix_entries'] += 1
            return
        try:
            coordinates = _attribute_metadata(sys._getframe(1), attribute_code)
            if closed or active or tuple(ids.shape) != (8, coordinates['suffix_length']):
                raise ValueError('Only the original paired8 suffix root may start this scope.')
            if coordinates != bank['metadata'].get('coordinates'):
                raise ValueError('The original full-EOS prefix/suffix coordinates changed.')
            report['coordinates'] = coordinates
            report['paired_root_capture_entries'] += 1
            active = True
            for index in sorted(gdn_bank):
                layer = owner_layers[index]
                mixer = layer.linear_attn
                suffix, width = coordinates['suffix_length'], layer.hidden_size
                value_width = mixer.num_v_heads*mixer.head_v_dim
                native[index] = {slot: {} for slot in slots}
                mixer_effects[index] = {slot: {} for slot in slots}
                metadata = report['capture_metadata'][str(index)] = dict(
                    mixer_forward=_source(type(mixer).forward), norm_forward=_source(type(mixer.norm).forward),
                    head_count=mixer.num_v_heads, head_width=mixer.head_v_dim,
                    capture_start=gdn_bank[index]['capture_start'], native_suffix_length=suffix, fields={})

                def save(name, value, *, _index=index, _suffix=suffix, _width=width,
                         _value_width=value_width, _metadata=metadata):
                    expected_width = _width if name in ('input', 'output') else _value_width
                    if not active or tuple(value.shape) != (8, _suffix, expected_width):
                        raise ValueError('Actual GDN field is not the original paired8 suffix.')
                    if name in _metadata['fields']:
                        raise ValueError('A second native field capture would overwrite the original root.')
                    _metadata['fields'][name] = dict(actual=_tensor_metadata(value), slots={})
                    observed = value if name in ('input', 'output') else value[:,gdn_bank[_index]['capture_start']:]
                    for slot in slots:
                        saved = owner_copy(observed[2*slot:2*slot+2], 'cpu', pinned_host=False)
                        _metadata['fields'][name]['slots'][str(slot)] = _tensor_metadata(saved)
                        if name in ('input', 'output'):
                            # The primitive bank already owns these two
                            # coefficients. Contract transient paired rows
                            # here, rather than retaining another activation.
                            coefficient = primitive_bank[_index]['slots'][slot]['m_mixer_'+name]
                            mixer_effects[_index][slot][name] = _effect(owner_effect, coefficient, saved)
                        else:
                            native[_index][slot][name] = saved

                def mixer_input(_target, inner_args, inner_kwargs, *, _save=save, _index=index):
                    try:
                        _save('input', _arg(inner_args, inner_kwargs, 'hidden_states', 0))
                    except Exception as exc:
                        report['diagnostics'].append(dict(where='native_input', layer=_index, error=repr(exc)))

                def mixer_output(_target, _args, output, *, _save=save, _index=index):
                    try:
                        _save('output', output if isinstance(output, torch.Tensor) else output[0])
                    except Exception as exc:
                        report['diagnostics'].append(dict(where='native_output', layer=_index, error=repr(exc)))

                def norm_output(_target, inner_args, inner_kwargs, output, *, _save=save,
                                _index=index, _suffix=suffix, _heads=mixer.num_v_heads,
                                _dimension=mixer.head_v_dim, _metadata=metadata):
                    try:
                        # The original HF call passes (o,z) positionally and
                        # flattens pairedB, time, head in exactly this order.
                        fields = dict(o=_arg(inner_args, inner_kwargs, 'x', 0),
                                      z=_arg(inner_args, inner_kwargs, 'gate', 1), gated=output)
                        _metadata['norm_flat_fields'] = {name:_tensor_metadata(value) for name,value in fields.items()}
                        for name, value in fields.items():
                            if tuple(value.shape) != (8*_suffix*_heads, _dimension):
                                raise ValueError('Native norm field differs from the original flat head geometry.')
                            _save(name, value.reshape(8, _suffix, _heads*_dimension))
                    except Exception as exc:
                        report['diagnostics'].append(dict(where='native_norm', layer=_index, error=repr(exc)))

                stack.callback(mixer.register_forward_pre_hook(mixer_input, with_kwargs=True).remove)
                stack.callback(mixer.register_forward_hook(mixer_output).remove)
                stack.callback(mixer.norm.register_forward_hook(norm_output, with_kwargs=True).remove)
        except Exception as exc:
            report['diagnostics'].append(dict(where='begin_paired_root', error=repr(exc)))
            cleanup()

    def project():
        for index, entry in sorted(gdn_bank.items()):
            try:
                for slot in slots:
                    q = entry['slots'][slot]
                    primitive = primitive_bank[index]['slots'][slot]
                    actual = native[index][slot]
                    effects = dict(
                        mixer_input=mixer_effects[index][slot]['input'],
                        mixer_output=mixer_effects[index][slot]['output'],
                        gated_output=_effect(owner_effect, q['m'], actual['gated']),
                        o_input=_effect(owner_effect, q['mo'], actual['o']),
                        z_input=_effect(owner_effect, q['mz'], actual['z']))
                    terms = dict(out_proj=effects['gated_output']-effects['mixer_output'],
                        norm_gate=effects['o_input']+effects['z_input']-effects['gated_output'],
                        remaining_input_FLA=effects['mixer_input']-effects['o_input']-effects['z_input'])
                    mixer_gap = effects['mixer_input']-effects['mixer_output']
                    report['layers'].append(dict(decoder_index=index, original_slot=slot,
                        paired_rows=[2*slot,2*slot+1], prefix_cut=entry['coordinates']['prefix_cut'],
                        suffix_length=entry['coordinates']['suffix_length'], effects=effects,
                        capture_start=entry['capture_start'], coefficient_suffix_length=q['m'].shape[1],
                        three_terms=terms, three_term_sum=sum(terms.values()),
                        primitive_mixer_difference=mixer_gap,
                        arithmetic_closure_residual=mixer_gap-sum(terms.values()),
                        coefficients={name:_tensor_metadata(value) for name,value in q.items()},
                        primitive_coefficients={name:_tensor_metadata(primitive[name]) for name in ('m_mixer_input','m_mixer_output')},
                        native_fields={name:_tensor_metadata(value) for name,value in actual.items()}))
            except Exception as exc:
                report['diagnostics'].append(dict(where='GDN_projection', layer=index, error=repr(exc)))

    def score_and_project(*args, **kwargs):
        value = original_score(*args, **kwargs)
        report['score_calls'] += 1
        try:
            cleanup()
            with torch.no_grad():
                project()
        except Exception as exc:
            report['diagnostics'].append(dict(where='score_projection', error=repr(exc)))
        finally:
            native.clear()
            mixer_effects.clear()
        return value

    handle = runner.model._conditional.register_forward_pre_hook(begin_root, with_kwargs=True)
    module.selected_target_log_probs = score_and_project
    try:
        yield report
    finally:
        module.selected_target_log_probs = original_score
        handle.remove()
        cleanup()
        native.clear()
        mixer_effects.clear()
        report['recorded_layer_slot_scalars'] = len(report['layers'])
        report['expected_layer_slot_scalars'] = len(gdn_bank)*len(slots)
