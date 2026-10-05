"""Passive original-FLA conditional projections on matched single-EOS roots.

The existing outer symmetric memory callback owns all finite mathematics and
endpoint averaging. Its five input coefficients and actual incoming do are
retained only for selected slots of one B4 group, on CPU. Original FLA Python
RETURN events supply normalized q/k/v/beta and public raw_g/output; those
native tensors are immediately contracted and discarded. This diagnostic
adds no model call, finite rule, reference, tolerance, correction or h/A tape.
"""
from contextlib import contextmanager, ExitStack
import importlib
import inspect
import sys

import torch

from observe_textcraft_conditional_boundaries import (
    _attribute_metadata, _slots, _source, _tensor_metadata,
)
from observe_textcraft_conditional_primitives import _effect, _layers


DEFAULT_LAYERS = (6, 8)
INPUT_FIELDS = ('q', 'k', 'v', 'beta', 'g')
NATIVE_NAMES = {'q': 'q', 'k': 'k', 'v': 'v', 'beta': 'beta', 'g': 'raw_g'}


def _actual_group(caller, owner_code):
    """Read only actual scalar layout locals of the original GDN callback."""
    current = caller
    try:
        while current is not None and current.f_code is not owner_code:
            current = current.f_back
        if current is None:
            raise ValueError('The original GDN pullback frame is absent.')
        values = current.f_locals
        return dict(capture_start=int(values['capture_start']),
                    cut=int(values['cut']),
                    head_start=int(values['start']) if values['fla_head_batch_size'] is not None else 0,
                    configured_head_batch_size=values['fla_head_batch_size'],
                    compact_length=int(values['length']),
                    rebased_fla_coefficient_start=int(values['fla_coefficient_start']))
    finally:
        del current, caller


@contextmanager
def collect_actual_fla(runner, bank, slots, *, layers=DEFAULT_LAYERS):
    """Nest inside the existing GDN collector on the original full trace.

    The caller clears fla_bank after both original single-root observations.
    No input operands or recurrent states survive a callback. The original
    outer callback is called first and its exact returned object is returned.
    """
    slots = _slots(slots)
    if slots != bank['slots']:
        raise ValueError('Use the original coefficient-bank slot order.')
    indices, owner_layers = _layers(runner, layers)
    if any(owner_layers[i].block_type != 'linear_attention' for i in indices):
        raise ValueError('Requested layers must be original GDN layers.')
    if 'fla_bank' in bank:
        raise ValueError('Do not overwrite an existing FLA observation bank.')
    module = sys.modules[type(runner).__module__]
    owner_copy, owner_effect = module._copy, module._token_effect
    gdn_code = inspect.unwrap(module.gdn_finite_pullback).__code__
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    originals = {i: runner.finite_fla_by_layer[i] for i in indices}
    entries = bank['fla_bank'] = {i: dict(groups=[]) for i in indices}
    receipt = bank['metadata']['conditional_fla'] = dict(
        scope=__doc__, layer_indices=list(indices), original_slots=list(slots),
        original_gdn_pullback=_source(module.gdn_finite_pullback),
        original_finite_fla=_source(runner.finite_fla),
        owner_copy=_source(owner_copy), owner_token_effect=_source(owner_effect),
        actual_native_fla_fp16='Read from actual operands; no dtype is changed here.',
        actual_head_batch_size=runner.gdn_head_batch_size,
        actual_compile_gdn_scalar_rules=runner.compile_gdn_scalar_rules,
        layers={}, diagnostics=[],
        coefficient_scope='Final original outer symmetric callback result; no second endpoint averaging.',
        lifetime='Selected-slot CPU coefficients and actual incoming do for one original B4 group only.')

    def wrapper(index, original):
        metadata = receipt['layers'][str(index)] = dict(
            original_outer_callback=_source(original),
            actual_norm_gate_rule=runner.norm_gate_rules[index], calls=0, groups=[])

        def observed(endpoints, upstream, scale):
            result = original(endpoints, upstream, scale)
            metadata['calls'] += 1
            try:
                coordinates = _attribute_metadata(sys._getframe(1), attribute_code)
                if coordinates != bank['metadata'].get('coordinates'):
                    raise ValueError('Original full-EOS prefix/suffix geometry changed.')
                layout = _actual_group(sys._getframe(1), gdn_code)
                q = result['q']
                shape = tuple(q.shape)
                if len(shape) != 4 or shape[0] != 4 or tuple(upstream.shape) != shape:
                    raise ValueError('Original callback is not B4/head-group FLA geometry.')
                time_start = layout['capture_start'] + layout['cut']
                if time_start+shape[1] != coordinates['suffix_length']:
                    raise ValueError('Actual compact FLA coefficients do not match native suffix coordinates.')
                entry = dict(layout=layout, coordinates=coordinates,
                             head_count=shape[2], head_width=shape[3],
                             native_time_start=time_start, slots={slot: {} for slot in slots})
                info = dict(layout=layout, coordinates=coordinates,
                            head_count=shape[2], head_width=shape[3], native_time_start=time_start,
                            actual_scale=float(scale), incoming_do=_tensor_metadata(upstream),
                            incoming_state=_tensor_metadata(endpoints['h']),
                            fields={}, actual_joint_input_projections={}, incoming_state_pair_absmax={})
                for name in INPUT_FIELDS:
                    coefficient = result[name]
                    endpoint = endpoints[NATIVE_NAMES[name]]
                    if tuple(coefficient.shape[:3]) != shape[:3] or tuple(endpoint.shape[:3]) != (8,)+shape[1:3]:
                        raise ValueError('Actual input coefficient/endpoint head-group shape differs.')
                    info['fields'][name] = dict(coefficient=_tensor_metadata(coefficient),
                                               endpoint=_tensor_metadata(endpoint), selected={})
                    for slot in slots:
                        saved = owner_copy(coefficient[slot:slot+1].flatten(2), 'cpu', pinned_host=False)
                        entry['slots'][slot][name] = saved
                        paired = owner_copy(endpoint[2*slot:2*slot+2].flatten(2), 'cpu', pinned_host=False)
                        values = info['actual_joint_input_projections'].setdefault(str(slot), {})
                        values[name] = _effect(owner_effect, saved, paired)
                        info['fields'][name]['selected'][str(slot)] = _tensor_metadata(saved)
                        del paired
                for slot in slots:
                    saved = owner_copy(upstream[slot:slot+1].flatten(2), 'cpu', pinned_host=False)
                    entry['slots'][slot]['do'] = saved
                    state = endpoints['h']
                    # Only a descriptive scalar from the actual incoming first
                    # chunk; no state capture, reconstruction or new criterion.
                    info['incoming_state_pair_absmax'][str(slot)] = float(
                        (state[2*slot+1, 0].float()-state[2*slot, 0].float()).abs().max().item())
                entries[index]['groups'].append(entry)
                metadata['groups'].append(info)
            except Exception as exc:
                receipt['diagnostics'].append(dict(where='outer_FLA_callback', layer=index, error=repr(exc)))
            return result
        return observed

    try:
        for index, original in originals.items():
            runner.finite_fla_by_layer[index] = wrapper(index, original)
        yield receipt
    finally:
        for index, original in originals.items():
            runner.finite_fla_by_layer[index] = original
        receipt['retained_cpu_tensor_bytes'] = sum(
            value.numel()*value.element_size() for entry in entries.values()
            for group in entry['groups'] for fields in group['slots'].values() for value in fields.values())


@contextmanager
def contract_saved_native_fla(runner, bank, slots):
    """Nest inside the existing GDN contractor, outside the root observer.

    Only the original paired8 root opens original LocalCaptureEvents. Stage
    RETURN supplies normalized q/k/v/beta; PUBLIC RETURN supplies unchanged
    raw g and value[0], the native FP16 public output before its BF16 boundary.
    Native fields are immediately contracted per original head group. Score
    is called once, the same object returned, and observation cleanup is final.
    """
    slots = _slots(slots)
    if slots != bank['slots']:
        raise ValueError('Use the original coefficient-bank slot order.')
    entries = bank['fla_bank']
    module = sys.modules[type(runner).__module__]
    original_score = module.selected_target_log_probs
    owner_copy, owner_effect = module._copy, module._token_effect
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    owner_layers = runner.model.model.language_model.layers
    chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
    event_module = importlib.import_module('accelerated.native_capture_events')
    public = inspect.unwrap(chunk.chunk_gated_delta_rule)
    stage = inspect.unwrap(chunk.chunk_gated_delta_rule_fwd)
    report = dict(scope=__doc__, original_slots=list(slots), layer_indices=sorted(entries),
        original_public_FLA=_source(public), original_stage=_source(stage),
        original_local_events=_source(event_module.LocalCaptureEvents.__init__),
        owner_token_effect=_source(owner_effect), wrapped_existing_score=_source(original_score),
        capture_metadata={}, layers=[], diagnostics=[], score_calls=0,
        paired_root_capture_entries=0, ignored_prefix_entries=0,
        interpretation='F=sum original five input-coefficient contractions; Y=actual incoming native do times public FP16 output difference. F-Y includes joint finite conditional error and native storage, not an official kernel error claim.')
    stack = ExitStack()
    active = False
    active_layer = None
    closed = False
    values = {index: {slot: dict(fields={}, groups=[]) for slot in slots} for index in entries}

    def cleanup():
        nonlocal active, active_layer, closed
        try:
            stack.close()
        except Exception as exc:
            report['diagnostics'].append(dict(where='hook_cleanup', error=repr(exc)))
        finally:
            active = False
            active_layer = None
            closed = True

    def project_field(index, name, native):
        suffix = report['coordinates']['suffix_length']
        mixer = owner_layers[index].linear_attn
        vector = name in ('q', 'k', 'v', 'o')
        expected = (8, suffix, mixer.num_v_heads)+( (mixer.head_v_dim,) if vector else () )
        if tuple(native.shape) != expected:
            raise ValueError('Actual native FLA field is not paired8/full-head suffix geometry.')
        metadata = report['capture_metadata'][str(index)]['fields']
        if name in metadata:
            raise ValueError('A second native FLA field would overwrite this root observation.')
        metadata[name] = _tensor_metadata(native)
        for group_index, group in enumerate(entries[index]['groups']):
            start, head = group['native_time_start'], group['layout']['head_start']
            count = group['head_count']
            for slot in slots:
                source = native[2*slot:2*slot+2, start:, head:head+count]
                paired = owner_copy(source.flatten(2), 'cpu', pinned_host=False)
                coefficient = group['slots'][slot]['do' if name == 'o' else name]
                value = _effect(owner_effect, coefficient, paired)
                row = values[index][slot]
                row['fields'][name] = row['fields'].get(name, 0.)+value
                while len(row['groups']) <= group_index:
                    row['groups'].append({})
                row['groups'][group_index][name] = value
                del paired

    def event(frame, kind, returned):
        if kind != 'return' or not active or active_layer is None:
            return
        index = active_layer
        try:
            local = frame.f_locals
            metadata = report['capture_metadata'][str(index)]
            if frame.f_code is stage.__code__:
                metadata['stage_returns'] += 1
                metadata['actual_stage_scale'] = float(local['scale'])
                metadata['actual_stage_cu_seqlens'] = None if local['cu_seqlens'] is None else _tensor_metadata(local['cu_seqlens'])
                # local g has already been replaced by chunk cumulative g.
                # It is intentionally neither saved nor paired with coeff g.
                for name in ('q', 'k', 'v', 'beta'):
                    project_field(index, name, local[name])
            elif frame.f_code is public.__code__:
                metadata['public_returns'] += 1
                metadata['actual_public_scale_argument'] = local['scale']
                metadata['actual_qk_normalization'] = bool(local['use_qk_l2norm_in_kernel'])
                state = local['initial_state']
                metadata['actual_public_initial_state'] = None if state is None else _tensor_metadata(state)
                project_field(index, 'g', local['g'])
                project_field(index, 'o', returned[0])
        except Exception as exc:
            report['diagnostics'].append(dict(where='native_FLA_return', layer=index, error=repr(exc)))

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
                raise ValueError('Original full-EOS prefix/suffix coordinates changed.')
            report['coordinates'] = coordinates
            report['paired_root_capture_entries'] += 1
            active = True
            for index in sorted(entries):
                mixer = owner_layers[index].linear_attn
                report['capture_metadata'][str(index)] = dict(fields={}, stage_returns=0, public_returns=0)

                def before(_target, _args, _kwargs, *, _index=index):
                    nonlocal active_layer
                    active_layer = _index

                def after(_target, _args, _output):
                    nonlocal active_layer
                    active_layer = None

                stack.callback(mixer.register_forward_pre_hook(before, with_kwargs=True).remove)
                stack.callback(mixer.register_forward_hook(after).remove)
            stack.enter_context(event_module.LocalCaptureEvents(
                [stage.__code__, public.__code__], event, returns_only=True))
        except Exception as exc:
            report['diagnostics'].append(dict(where='begin_paired_root', error=repr(exc)))
            cleanup()

    def score_and_project(*args, **kwargs):
        value = original_score(*args, **kwargs)
        report['score_calls'] += 1
        try:
            cleanup()
            for index in sorted(entries):
                for slot in slots:
                    row = values[index][slot]
                    if set(row['fields']) != set(INPUT_FIELDS+('o',)):
                        raise ValueError('Original native FLA input/output observations are incomplete.')
                    F = sum(row['fields'][name] for name in INPUT_FIELDS)
                    Y = row['fields']['o']
                    report['layers'].append(dict(decoder_index=index, original_slot=slot,
                        paired_rows=[2*slot, 2*slot+1], fields=row['fields'], head_groups=row['groups'],
                        F_input_projection=F, Y_public_output_projection=Y,
                        conditional_FLA_difference=F-Y,
                        coordinates=report['coordinates'],
                        group_layouts=[dict(layout=group['layout'], native_time_start=group['native_time_start'],
                            head_count=group['head_count'], head_width=group['head_width']) for group in entries[index]['groups']]))
        except Exception as exc:
            report['diagnostics'].append(dict(where='score_projection', error=repr(exc)))
        finally:
            values.clear()
        return value

    handle = runner.model._conditional.register_forward_pre_hook(begin_root, with_kwargs=True)
    module.selected_target_log_probs = score_and_project
    try:
        yield report
    finally:
        module.selected_target_log_probs = original_score
        handle.remove()
        cleanup()
        values.clear()
        report['recorded_layer_slot_scalars'] = len(report['layers'])
        report['expected_layer_slot_scalars'] = len(entries)*len(slots)
