"""Passive subblock projections of the unchanged conditional-boundary owner.

Compose with the existing full-EOS coefficient collector and single-EOS root
observer. Only actual selected-slot coefficients are kept briefly on CPU.
The single-root observation uses original decoder capture hooks and an output
hook for the original mixer; it adds no forward, finite rule or reference.
Six descriptive contractions telescope to the decoder input/output difference.
CPU norm projections distinguish storage/backend terms from the remaining
joint-direction residual. They do not define tolerances or correct credit.
"""
from contextlib import contextmanager, ExitStack
import importlib
import inspect
import sys

import torch

from observe_textcraft_conditional_boundaries import (
    _attribute_metadata, _slots, _source, _tensor_metadata,
)
from observe_textcraft_norm_operands import _arg


DEFAULT_LAYERS = (6, 8, 11)
NATIVE_FIELDS = frozenset(('input_norm_output', 'post_norm_input',
                           'post_norm_output', 'mlp_output'))


def _layers(runner, layers):
    indices = tuple(int(i) for i in layers)
    owner_layers = runner.model.model.language_model.layers
    if len(indices) != len(set(indices)) or any(i < 0 or i >= len(owner_layers) for i in indices):
        raise ValueError('Use distinct original decoder indices.')
    return indices, owner_layers


def _local_weight(value):
    # The native Qwen norm and the existing norm observer use this same public
    # local read. No sharded-parameter gather is introduced by the diagnostic.
    return value.to_local() if hasattr(value, 'to_local') else value


@contextmanager
def collect_actual_primitives(runner, bank, slots, *, layers=DEFAULT_LAYERS):
    """Nest inside collect_actual_coefficients during one original full trace.

    Populate bank['primitive_bank']; discard it after both single-root probes.
    The original decoder, boundary and mixer callbacks are called first and
    their exact original returned objects are returned without replacement.
    No public observer or diagnostics flag is supplied.
    """
    slots = _slots(slots)
    if slots != bank['slots']:
        raise ValueError('Primitive slots must retain the original coefficient-bank order.')
    indices, owner_layers = _layers(runner, layers)
    module = sys.modules[type(runner).__module__]
    original_decoder = module.decoder_finite_pullback
    owner_copy = module._copy
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    if 'primitive_bank' in bank:
        raise ValueError('Do not overwrite an existing primitive bank.')
    primitive_bank = bank['primitive_bank'] = {}
    receipt = bank['metadata']['conditional_primitives'] = dict(
        scope=__doc__, layer_indices=list(indices), original_slots=list(slots),
        original_decoder=_source(original_decoder), owner_copy=_source(owner_copy),
        layers={}, diagnostics=[],
        coefficient_scope='Actual full-span joint finite coefficients; not recomputed single-delete coefficients.',
        lifetime='CPU bank owned by this one original B4 group; caller clears it after both single roots.')
    layer_indices = {id(owner_layers[i]): i for i in indices}

    def observed_decoder(*args, **kwargs):
        layer = _arg(args, kwargs, 'layer', 0)
        index = layer_indices.get(id(layer))
        if index is None:
            return original_decoder(*args, **kwargs)
        boundaries = _arg(args, kwargs, 'boundaries', 4)
        original_mlp = boundaries.mlp
        original_norm = boundaries.norm_residual
        original_mixer = _arg(args, kwargs, 'mixer_pullback', 3)
        upstream = _arg(args, kwargs, 'upstream', 2)
        entry = dict(slots={slot: {} for slot in slots}, norms={})
        metadata = dict(decoder_index=index, block_type=layer.block_type,
                        coefficients={}, norm_calls=[], diagnostics=[], decoder_calls=1)
        norm_count = 0

        def save_coefficient(name, value):
            try:
                coordinates = _attribute_metadata(sys._getframe(1), attribute_code)
                if coordinates != bank['metadata'].get('coordinates'):
                    raise ValueError('Primitive callback changed the original full-EOS layout.')
                if value.shape != (4, coordinates['suffix_length'], layer.hidden_size):
                    raise ValueError('Actual primitive coefficient is not the original B4 suffix.')
                entry['coordinates'] = coordinates
                metadata['coefficients'][name] = dict(original=_tensor_metadata(value), slots={})
                for slot in slots:
                    saved = owner_copy(value[slot:slot+1], 'cpu', pinned_host=False)
                    entry['slots'][slot][name] = saved
                    metadata['coefficients'][name]['slots'][str(slot)] = _tensor_metadata(saved)
            except Exception as exc:
                metadata['diagnostics'].append(dict(where='coefficient_'+name, error=repr(exc)))

        def mlp(*inner_args, **inner_kwargs):
            result = original_mlp(*inner_args, **inner_kwargs)
            save_coefficient('m_mlp_norm_output', result)
            return result

        def norm_residual(*inner_args, **inner_kwargs):
            nonlocal norm_count
            result = original_norm(*inner_args, **inner_kwargs)
            position = norm_count
            norm_count += 1
            if position < 2:
                name = ('post_attention_norm', 'input_norm')[position]
                save_coefficient(('m_mixer_output', 'm_input')[position], result)
                try:
                    raw = _arg(inner_args, inner_kwargs, 'raw_weight', 2)
                    weight = owner_copy(_local_weight(raw), 'cpu', pinned_host=False)
                    norm_module = layer.post_attention_layernorm if position == 0 else layer.input_layernorm
                    eps = _arg(inner_args, inner_kwargs, 'eps', 5)
                    entry['norms'][name] = dict(raw_weight=weight, eps=eps,
                                               native_norm_class=type(norm_module))
                    metadata['norm_calls'].append(dict(name=name, original_call_index=position,
                        eps=eps, raw_weight=_tensor_metadata(weight), original_raw_weight_type=type(raw).__name__,
                        owner_forward=_source(type(norm_module).forward),
                        owner_norm_primitive=_source(type(norm_module)._norm),
                        actual_boundary=_source(original_norm)))
                except Exception as exc:
                    metadata['diagnostics'].append(dict(where='norm_'+name, error=repr(exc)))
            else:
                metadata['diagnostics'].append(dict(where='norm_call_order', original_call_index=position))
            return result

        def mixer(*inner_args, **inner_kwargs):
            result = original_mixer(*inner_args, **inner_kwargs)
            save_coefficient('m_mixer_input', result[0])
            return result

        # Only substitute observation callbacks at the original owner seam.
        # Argument objects, diagnostics=False and consume_captures are kept.
        call_args = list(args)
        call_kwargs = dict(kwargs)
        if 'mixer_pullback' in call_kwargs:
            call_kwargs['mixer_pullback'] = mixer
        else:
            call_args[3] = mixer
        try:
            boundaries.mlp = mlp
            boundaries.norm_residual = norm_residual
            result = original_decoder(*call_args, **call_kwargs)
            save_coefficient('m_output', upstream)
            # m_input is the actual second norm return, also returned by the
            # original decoder; no alternative input coefficient is computed.
            if index in primitive_bank:
                metadata['diagnostics'].append(dict(where='duplicate_decoder', error='Second capture for this layer.'))
            else:
                primitive_bank[index] = entry
                receipt['layers'][str(index)] = metadata
            return result
        finally:
            boundaries.mlp = original_mlp
            boundaries.norm_residual = original_norm

    module.decoder_finite_pullback = observed_decoder
    try:
        yield receipt
    finally:
        module.decoder_finite_pullback = original_decoder
        receipt['captured_layer_indices'] = sorted(primitive_bank)
        receipt['all_requested_layers_present'] = receipt['captured_layer_indices'] == sorted(indices)
        receipt['retained_cpu_tensor_bytes'] = sum(
            tensor.numel()*tensor.element_size()
            for layer in primitive_bank.values()
            for values in layer['slots'].values() for tensor in values.values()) + sum(
            spec['raw_weight'].numel()*spec['raw_weight'].element_size()
            for layer in primitive_bank.values() for spec in layer['norms'].values())


def _effect(owner, coefficients, endpoints):
    return float(owner(coefficients, endpoints).detach().sum())


def _difference(actual, reference):
    delta = actual.double()-reference.double()
    return dict(exactly_equal=torch.equal(actual, reference),
                maximum_absolute_difference=float(delta.abs().max()),
                rms_difference=float(delta.square().mean().sqrt()),
                mean_difference=float(delta.mean()))


def _norm_projection(owner_effect, norm_spec, branch_coefficient, total_coefficient,
                     residual_coefficient, actual_input, actual_output):
    """Use original HF/finite owners on saved single-pair CPU operands only."""
    signed = importlib.import_module('signed_secant_rules')
    weight = norm_spec['raw_weight']
    norm = norm_spec['native_norm_class'](actual_input.shape[-1], eps=norm_spec['eps']).to(
        device='cpu', dtype=weight.dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
        native_cpu = norm(actual_input)
        # Same HF primitive and weight order as the existing norm analyzer.
        # These continuous outputs deliberately omit only output type_as(x).
        continuous32 = norm._norm(actual_input.float())*(1.0+norm.weight.float())
        continuous64 = norm._norm(actual_input.double())*(1.0+norm.weight.double())
        single32 = signed.rmsnorm_secant_pullback(actual_input[0::2].float(),
            actual_input[1::2].float(), 1.0+weight.float(), branch_coefficient.float(), norm_spec['eps'])
        single64 = signed.rmsnorm_secant_pullback(actual_input[0::2].double(),
            actual_input[1::2].double(), 1.0+weight.double(), branch_coefficient.double(), norm_spec['eps'])
        e = dict(joint_combined_input=_effect(owner_effect, total_coefficient, actual_input),
                 joint_residual_input=_effect(owner_effect, residual_coefficient, actual_input),
                 actual_GPU_native_output=_effect(owner_effect, branch_coefficient, actual_output),
                 same_HF_CPU_native_output=_effect(owner_effect, branch_coefficient, native_cpu),
                 same_HF_CPU_continuous_FP32_output=_effect(owner_effect, branch_coefficient, continuous32),
                 same_HF_CPU_continuous_FP64_output=_effect(owner_effect, branch_coefficient, continuous64),
                 original_single_pair_finite_FP32_input=_effect(owner_effect, single32, actual_input),
                 original_single_pair_finite_FP64_input=_effect(owner_effect, single64, actual_input))
    parts = dict(
        joint_conditional_minus_single_pair_FP64=(e['joint_combined_input']-e['joint_residual_input']-e['original_single_pair_finite_FP64_input']),
        single_pair_FP64_identity_residual=(e['original_single_pair_finite_FP64_input']-e['same_HF_CPU_continuous_FP64_output']),
        continuous_FP64_minus_FP32=(e['same_HF_CPU_continuous_FP64_output']-e['same_HF_CPU_continuous_FP32_output']),
        CPU_continuous_FP32_minus_native_storage=(e['same_HF_CPU_continuous_FP32_output']-e['same_HF_CPU_native_output']),
        CPU_native_minus_actual_GPU_native=(e['same_HF_CPU_native_output']-e['actual_GPU_native_output']))
    gap = e['joint_combined_input']-e['joint_residual_input']-e['actual_GPU_native_output']
    return dict(effects=e, decomposition=parts, joint_native_gap=gap,
        sum_of_decomposition=sum(parts.values()), arithmetic_closure_residual=gap-sum(parts.values()),
        single_pair_FP32_identity_residual=(e['original_single_pair_finite_FP32_input']-e['same_HF_CPU_continuous_FP32_output']),
        actual_GPU_native_vs_same_HF_CPU=_difference(actual_output, native_cpu),
        sources=dict(HF_forward=_source(type(norm).forward), HF_norm=_source(type(norm)._norm),
                     single_pair_finite=_source(signed.rmsnorm_secant_pullback)),
        dtypes=dict(input=str(actual_input.dtype), actual_GPU_output=str(actual_output.dtype),
                    CPU_native_output=str(native_cpu.dtype), continuous32=str(continuous32.dtype),
                    continuous64=str(continuous64.dtype), single32=str(single32.dtype), single64=str(single64.dtype)),
        interpretation='The joint-direction term also includes arithmetic in the saved joint coefficients; it is not proof that all remaining error is joint decomposition. CPU/GPU output differences are reported separately.')


def _addition_projection(owner_effect, coefficient, left, right, actual_output):
    native_cpu = left+right
    continuous32 = left.float()+right.float()
    e = dict(left_branch=_effect(owner_effect, coefficient, left),
             right_branch=_effect(owner_effect, coefficient, right),
             actual_GPU_output=_effect(owner_effect, coefficient, actual_output),
             same_dtype_CPU_output=_effect(owner_effect, coefficient, native_cpu),
             FP32_CPU_output=_effect(owner_effect, coefficient, continuous32))
    parts = dict(branch_sum_minus_FP32_CPU=e['left_branch']+e['right_branch']-e['FP32_CPU_output'],
                 FP32_CPU_minus_native_storage=e['FP32_CPU_output']-e['same_dtype_CPU_output'],
                 CPU_native_minus_actual_GPU=e['same_dtype_CPU_output']-e['actual_GPU_output'])
    gap = e['left_branch']+e['right_branch']-e['actual_GPU_output']
    return dict(effects=e, decomposition=parts, branch_sum_minus_actual_output=gap,
                arithmetic_closure_residual=gap-sum(parts.values()),
                actual_output_vs_same_dtype_CPU_add=_difference(actual_output, native_cpu),
                dtypes=dict(left=str(left.dtype), right=str(right.dtype), actual_output=str(actual_output.dtype),
                            CPU_native_output=str(native_cpu.dtype), CPU_FP32_output=str(continuous32.dtype)))


@contextmanager
def contract_saved_native_primitives(runner, bank, slots):
    """Nest inside the existing contract context, outside observe_original_root.

    Only its original paired8 root starts passive captures. Prefix B4 is
    ignored. Captures close at the original score and cannot see replay.
    The existing score wrapper is called first and returned unchanged; only
    scalar/metadata primitive results survive this context.
    """
    slots = _slots(slots)
    if slots != bank['slots']:
        raise ValueError('Use the original group slot order.')
    primitive_bank = bank['primitive_bank']
    module = sys.modules[type(runner).__module__]
    owner_copy = module._copy
    owner_effect = module._token_effect
    original_score = module.selected_target_log_probs
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    owner_layers = runner.model.model.language_model.layers
    decoder_capture = module.NativeDecoderCapture if runner.capture_backend is None else runner.capture_backend.NativeDecoderCapture
    report = dict(scope=__doc__, layer_indices=sorted(primitive_bank), original_slots=list(slots),
        wrapped_existing_score=_source(original_score), owner_token_effect=_source(owner_effect),
        original_decoder_capture=dict(class_name=decoder_capture.__name__,
                                      class_module=decoder_capture.__module__,
                                      constructor=_source(decoder_capture.__init__)), paired_root_capture_entries=0,
        ignored_prefix_entries=0, score_calls=0, layers=[], capture_metadata={}, diagnostics=[],
        primitive_scope='Actual single root outputs with fixed full-EOS cut; full-span joint coefficients.',
        numerical_scope='Descriptive CPU projections only; no tolerance, correction or official kernel-failure claim.',
        six_terms=['mlp', 'post_norm', 'mlp_residual_add', 'mixer', 'input_norm', 'attention_residual_add'])
    captures = {}
    mixer_outputs = {}
    stack = ExitStack()
    active = False
    closed = False

    def cleanup():
        nonlocal active, closed
        try:
            stack.close()
        except Exception as exc:
            report['diagnostics'].append(dict(where='capture_cleanup', error=repr(exc)))
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
            if closed or active or ids.shape != (8, coordinates['suffix_length']):
                raise ValueError('Only one original paired8 suffix root may start this capture scope.')
            if coordinates != bank['metadata'].get('coordinates'):
                raise ValueError('Single root does not retain the original full-EOS cut and suffix.')
            report['coordinates'] = coordinates
            report['paired_root_capture_entries'] += 1
            active = True
            for index in sorted(primitive_bank):
                layer = owner_layers[index]
                cap = decoder_capture(layer, destination='cpu', copy_tensors=True, retained_names=NATIVE_FIELDS)
                original_retain = cap.retain
                def guarded_retain(name, value, *, _original=original_retain, _index=index):
                    try:
                        return _original(name, value)
                    except Exception as exc:
                        report['diagnostics'].append(dict(where='native_decoder_capture', layer=_index,
                                                          field=name, error=repr(exc)))
                cap.retain = guarded_retain
                captures[index] = cap
                try:
                    cap.__enter__()
                except Exception:
                    cap.__exit__(*sys.exc_info())
                    raise
                stack.callback(cap.__exit__, None, None, None)
                target = layer.linear_attn if layer.block_type == 'linear_attention' else layer.self_attn
                def mixer_output(_target, _args, output, *, _index=index,
                                 _width=layer.hidden_size, _suffix=coordinates['suffix_length']):
                    try:
                        value = output if isinstance(output, torch.Tensor) else output[0]
                        if not active or value.shape != (8, _suffix, _width):
                            raise ValueError('Mixer output is not the original paired8 root suffix.')
                        if _index in mixer_outputs:
                            raise ValueError('Mixer output would overwrite the original root capture.')
                        mixer_outputs[_index] = owner_copy(value, 'cpu', pinned_host=False)
                    except Exception as exc:
                        report['diagnostics'].append(dict(where='native_mixer_output', layer=_index, error=repr(exc)))
                stack.callback(target.register_forward_hook(mixer_output).remove)
        except Exception as exc:
            report['diagnostics'].append(dict(where='begin_paired_root', error=repr(exc)))
            cleanup()

    def project_from_frame(caller):
        current = caller
        root = None
        try:
            while current is not None and current.f_code is not attribute_code:
                current = current.f_back
            if current is None:
                raise ValueError('Original attribute frame absent at the score callback.')
            coordinates = _attribute_metadata(current, attribute_code)
            if coordinates != bank['metadata'].get('coordinates'):
                raise ValueError('Primitive projection layout differs from the full-EOS bank.')
            root = current.f_locals['root']
            for index, entry in sorted(primitive_bank.items()):
                try:
                    cap = captures[index]
                    native = cap.values
                    x = root[str(index)]
                    y = root['final_norm_input' if index == 31 else str(index+1)]
                    a = mixer_outputs[index]
                    n1, s, n2, f = (native[name] for name in ('input_norm_output', 'post_norm_input', 'post_norm_output', 'mlp_output'))
                    fields = dict(x=x, n1=n1, a=a, s=s, n2=n2, f=f, y=y)
                    expected = (8, coordinates['suffix_length'], owner_layers[index].hidden_size)
                    if any(value.device.type != 'cpu' or value.shape != expected for value in fields.values()):
                        raise ValueError('Original native primitive fields must remain paired8 CPU suffix tensors.')
                    report['capture_metadata'][str(index)] = dict(
                        original_decoder_calls=cap.calls.copy(), native_fields={name:_tensor_metadata(value) for name,value in fields.items()},
                        mixer_module='linear_attn' if owner_layers[index].block_type=='linear_attention' else 'self_attn',
                        mixer_field_semantics='Final native mixer module output, after its output projection; not FLA internal o or raw FA output.')
                    for slot in slots:
                        q = entry['slots'][slot]
                        values = {name:value[2*slot:2*slot+2] for name,value in fields.items()}
                        mo, u, v, w, mi = (q[name] for name in ('m_output','m_mlp_norm_output','m_mixer_output','m_mixer_input','m_input'))
                        def e(m, field):
                            return _effect(owner_effect, m, values[field])
                        terms = dict(
                            mlp=e(u,'n2')-e(mo,'f'),
                            post_norm=e(v,'s')-e(mo,'s')-e(u,'n2'),
                            mlp_residual_add=e(mo,'s')+e(mo,'f')-e(mo,'y'),
                            mixer=e(w,'n1')-e(v,'a'),
                            input_norm=e(mi,'x')-e(v,'x')-e(w,'n1'),
                            attention_residual_add=e(v,'x')+e(v,'a')-e(v,'s'))
                        cin, cout = e(mi,'x'), e(mo,'y')
                        row = dict(decoder_index=index, block_type=owner_layers[index].block_type,
                            original_slot=slot, paired_rows=[2*slot,2*slot+1], prefix_cut=coordinates['prefix_cut'],
                            suffix_length=coordinates['suffix_length'], Cin=cin, Cout=cout, Cin_minus_Cout=cin-cout,
                            six_terms=terms, six_term_sum=sum(terms.values()),
                            arithmetic_closure_residual=(cin-cout)-sum(terms.values()),
                            coefficients={name:_tensor_metadata(value) for name,value in q.items()},
                            native_fields={name:_tensor_metadata(value) for name,value in values.items()},
                            interpretation='Six-term arithmetic identity; mixer and MLP remain mixed conditional/numerical residuals.')
                        row['norms'] = dict(
                            post_attention_norm=_norm_projection(owner_effect, entry['norms']['post_attention_norm'], u, v, mo, values['s'], values['n2']),
                            input_norm=_norm_projection(owner_effect, entry['norms']['input_norm'], w, mi, v, values['x'], values['n1']))
                        row['residual_additions'] = dict(
                            mlp=_addition_projection(owner_effect, mo, values['s'], values['f'], values['y']),
                            attention=_addition_projection(owner_effect, v, values['x'], values['a'], values['s']))
                        report['layers'].append(row)
                except Exception as exc:
                    report['diagnostics'].append(dict(where='primitive_projection', layer=index, error=repr(exc)))
        finally:
            del current, caller, root

    def score_and_project(*args, **kwargs):
        value = original_score(*args, **kwargs)
        report['score_calls'] += 1
        try:
            cleanup()
            with torch.no_grad():
                project_from_frame(sys._getframe(1))
        except Exception as exc:
            report['diagnostics'].append(dict(where='score_projection', error=repr(exc)))
        finally:
            for cap in captures.values():
                cap.values.clear()
            captures.clear()
            mixer_outputs.clear()
        return value

    handle = runner.model._conditional.register_forward_pre_hook(begin_root, with_kwargs=True)
    module.selected_target_log_probs = score_and_project
    try:
        yield report
    finally:
        module.selected_target_log_probs = original_score
        handle.remove()
        cleanup()
        for cap in captures.values():
            cap.values.clear()
        captures.clear()
        mixer_outputs.clear()
        report['recorded_layer_slot_scalars'] = len(report['layers'])
        report['expected_layer_slot_scalars'] = len(primitive_bank)*len(slots)
