"""Passive FLA contractions on the same original full-response EOS pair.

Compose the existing full finite collector. Its original GDN capture event
supplies the public FP16 output and the subsequent actual BF16 norm input;
the same original finite callback supplies the five coefficients and native
do. Only selected paired CPU outputs survive until that layer's callback.
There is no model/FLA call, finite formula, reference, tolerance or correction.
"""
from contextlib import contextmanager
import importlib
import inspect
import sys
from types import SimpleNamespace

import torch

from observe_textcraft_conditional_boundaries import (
    _attribute_metadata, _slots, _source, _tensor_metadata,
)
from observe_textcraft_conditional_primitives import _effect, _layers
from observe_textcraft_conditional_fla import collect_actual_fla, _actual_group, INPUT_FIELDS


DEFAULT_LAYERS = (6, 8)


@contextmanager
def collect_actual_joint_fla(runner, bank, slots, *, layers=DEFAULT_LAYERS):
    """Replace only the existing full-trace FLA collector in the old recipe.

    This context internally composes collect_actual_fla; do not nest a second
    copy of that collector. The original capture backend/other capture classes
    and finite callback results remain owned by the unchanged runner. Outputs
    are discarded after the last real head group of each original layer.
    """
    slots = _slots(slots)
    if slots != bank['slots']:
        raise ValueError('Use the original coefficient-bank slot order.')
    indices, owner_layers = _layers(runner, layers)
    if any(owner_layers[i].block_type != 'linear_attention' for i in indices):
        raise ValueError('The requested original layers must be GDN layers.')
    module = sys.modules[type(runner).__module__]
    owner_copy, owner_effect = module._copy, module._token_effect
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    gdn_code = inspect.unwrap(module.gdn_finite_pullback).__code__
    chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
    public = inspect.unwrap(chunk.chunk_gated_delta_rule)
    original_backend = runner.capture_backend
    capture_owner = module if original_backend is None else original_backend
    original_capture = capture_owner.NativeGDNCapture
    layer_by_module = {id(owner_layers[i].linear_attn): i for i in indices}
    retained = {}
    report = dict(scope=__doc__, layer_indices=list(indices), original_slots=list(slots),
        original_public_FLA=_source(public), original_capture_event=_source(original_capture.event),
        original_capture_init=_source(original_capture.__init__),
        original_GDN_pullback=_source(module.gdn_finite_pullback),
        owner_copy=_source(owner_copy), owner_token_effect=_source(owner_effect),
        actual_compile_gdn_scalar_rules=runner.compile_gdn_scalar_rules,
        actual_head_batch_size=runner.gdn_head_batch_size,
        original_capture_backend_module=None if original_backend is None else getattr(original_backend, '__name__', None),
        capture_metadata={}, layers=[], diagnostics=[],
        actual_public_output_returns=0, actual_norm_input_captures=0,
        actual_original_finite_head_calls=0, peak_retained_cpu_output_bytes=0,
        added_native_model_calls=0, added_native_FLA_calls=0,
        output_boundary_scope='Y_public and Y_norm_boundary use the identical actual native do. Their difference isolates the public-output to norm-input storage boundary, not the earlier mo to do cast.',
        identity_scope='F_joint and Y_joint use the same original full-response EOS/factual endpoints. These contractions are descriptive, not an official finite-identity tolerance.')

    def retained_bytes():
        return sum(value.numel()*value.element_size() for entry in retained.values()
                   for field in entry['outputs'].values() for value in field.values())

    class JointCapture(original_capture):
        def event(self, frame, kind, returned):
            value = super().event(frame, kind, returned)
            index = layer_by_module.get(id(self.module))
            if index is None:
                return value
            try:
                label = self.codes.get(frame.f_code)
                is_public = kind == 'return' and label == 'FLA' and frame.f_code is public.__code__
                is_norm = kind == 'call' and label == 'norm' and frame.f_locals['self'] is self.module.norm
                if not (is_public or is_norm):
                    return value
                if not self.active:
                    raise ValueError('Original GDN capture is not active at its output boundary.')
                coordinates = _attribute_metadata(sys._getframe(1), attribute_code)
                if coordinates != bank['metadata'].get('coordinates'):
                    raise ValueError('Original full-EOS prefix/suffix coordinates changed.')
                expected = (8, coordinates['suffix_length'], self.module.num_v_heads, self.module.head_v_dim)
                if tuple(self.input_shape) != expected[:2]+(self.module.hidden_size,):
                    raise ValueError('Only the original paired8 suffix replay may supply outputs.')
                entry = retained.setdefault(index, dict(coordinates=coordinates,
                    capture_start=int(self.coefficient_start), outputs={}, groups=0))
                metadata = report['capture_metadata'].setdefault(str(index), dict(
                    coordinates=coordinates, capture_start=int(self.coefficient_start),
                    public_return_count=0, norm_input_count=0, fields={}, selected_cpu_outputs={}))
                if entry['coordinates'] != coordinates or entry['capture_start'] != self.coefficient_start:
                    raise ValueError('The original replay output geometry changed.')
                if is_public:
                    native = returned[0]
                    if tuple(native.shape) != expected:
                        raise ValueError('Original public output is not the paired8/full-head suffix.')
                    native = self.select_time(native)
                    name = 'public_output'
                    local = frame.f_locals
                    initial = local['initial_state']
                    metadata.update(actual_public_scale_argument=local['scale'],
                        actual_public_qk_normalization=bool(local['use_qk_l2norm_in_kernel']),
                        actual_public_cu_seqlens=None if local['cu_seqlens'] is None else _tensor_metadata(local['cu_seqlens']),
                        actual_public_initial_state=None if initial is None else _tensor_metadata(initial),
                        initial_state_source='Actual initial_state local in the original public FLA RETURN; not stage h or a reconstructed prefix.',
                        public_return_source=_source(public))
                    metadata['public_return_count'] += 1
                    report['actual_public_output_returns'] += 1
                else:
                    # super.event already retained exactly the tensor read by
                    # the original norm, with the actual compact capture cut.
                    native = self.endpoints['o']
                    name = 'norm_input'
                    metadata['norm_input_count'] += 1
                    report['actual_norm_input_captures'] += 1
                if name in entry['outputs']:
                    raise ValueError('A second original output event would overwrite this replay.')
                compact = (8, coordinates['suffix_length']-self.coefficient_start)+expected[2:]
                if tuple(native.shape) != compact:
                    raise ValueError('Actual output differs from the compact original capture geometry.')
                metadata['fields'][name] = _tensor_metadata(native)
                entry['outputs'][name] = {
                    slot: owner_copy(native[2*slot:2*slot+2].flatten(2), 'cpu', pinned_host=False)
                    for slot in slots}
                metadata['selected_cpu_outputs'][name] = {
                    str(slot): _tensor_metadata(saved) for slot, saved in entry['outputs'][name].items()}
                report['peak_retained_cpu_output_bytes'] = max(report['peak_retained_cpu_output_bytes'], retained_bytes())
            except Exception as exc:
                report['diagnostics'].append(dict(where='original_GDN_output_event', layer=index, error=repr(exc)))
            return value

    diagnostic_backend = SimpleNamespace(
        NativeGDNCapture=JointCapture,
        NativeDecoderCapture=capture_owner.NativeDecoderCapture,
        NativeDenseAttentionCapture=capture_owner.NativeDenseAttentionCapture)

    def wrapper(index, original, original_full_metadata):
        def observed(endpoints, upstream, scale):
            result = original(endpoints, upstream, scale)
            report['actual_original_finite_head_calls'] += 1
            try:
                coordinates = _attribute_metadata(sys._getframe(1), attribute_code)
                layout = _actual_group(sys._getframe(1), gdn_code)
                entry = retained[index]
                if entry['coordinates'] != coordinates or entry['capture_start'] != layout['capture_start']:
                    raise ValueError('Native output and original finite callback have different endpoints/cuts.')
                shape = tuple(upstream.shape)
                head, cut = layout['head_start'], layout['cut']
                mixer = owner_layers[index].linear_attn
                if len(shape) != 4 or shape[0] != 4 or shape[3] != mixer.head_v_dim:
                    raise ValueError('The actual original head-group shape differs.')
                if layout['configured_head_batch_size'] is None:
                    if head != 0 or shape[2] != mixer.num_v_heads:
                        raise ValueError('The original unpartitioned head-group shape differs.')
                elif shape[2] > layout['configured_head_batch_size'] or head+shape[2] > mixer.num_v_heads:
                    raise ValueError('The actual original head-group range differs.')
                if shape[1] != coordinates['suffix_length']-layout['capture_start']-cut:
                    raise ValueError('Original incoming do does not match its output suffix.')
                if set(entry['outputs']) != {'public_output', 'norm_input'}:
                    raise ValueError('Both original output boundaries must be observed before the finite callback.')
                info = original_full_metadata['layers'][str(index)]['groups'][-1]
                if info['layout'] != layout:
                    raise ValueError('Existing full F_joint metadata belongs to a different head group.')
                for slot in slots:
                    do = owner_copy(upstream[slot:slot+1].flatten(2), 'cpu', pinned_host=False)
                    def paired(name):
                        output = entry['outputs'][name][slot]
                        output = output.reshape(2, output.shape[1], owner_layers[index].linear_attn.num_v_heads, shape[3])
                        return output[:,cut:,head:head+shape[2]].flatten(2)
                    public_output, norm_input = paired('public_output'), paired('norm_input')
                    Y_public = _effect(owner_effect, do, public_output)
                    Y_norm = _effect(owner_effect, do, norm_input)
                    fields = info['actual_joint_input_projections'][str(slot)]
                    F_joint = sum(fields[name] for name in INPUT_FIELDS)
                    report['layers'].append(dict(decoder_index=index, original_slot=slot,
                        paired_rows=[2*slot, 2*slot+1], coordinates=coordinates, layout=layout,
                        native_time_start=layout['capture_start']+cut,
                        head_count=shape[2], head_width=shape[3], actual_scale=float(scale),
                        actual_native_do=_tensor_metadata(upstream), CPU_do=_tensor_metadata(do),
                        actual_FLA_incoming_h=info['incoming_state'],
                        incoming_first_chunk_h_pair_absmax=info['incoming_state_pair_absmax'][str(slot)],
                        original_joint_input_fields=dict(fields), F_joint=F_joint,
                        Y_joint_public_output=Y_public, Y_joint_norm_boundary=Y_norm,
                        F_joint_minus_Y_public=F_joint-Y_public,
                        public_minus_norm_boundary=Y_public-Y_norm,
                        F_joint_minus_Y_norm_boundary=F_joint-Y_norm))
                    del do, public_output, norm_input
                entry['groups'] += 1
                if head+shape[2] == owner_layers[index].linear_attn.num_v_heads:
                    report['capture_metadata'][str(index)]['consumed_original_head_groups'] = entry['groups']
                    entry['outputs'].clear()
                    del retained[index]
            except Exception as exc:
                report['diagnostics'].append(dict(where='same_endpoint_FLA_contraction', layer=index, error=repr(exc)))
            return result
        return observed

    with collect_actual_fla(runner, bank, slots, layers=indices) as original_receipt:
        original_receipt['joint_fla'] = report
        originals = {i: runner.finite_fla_by_layer[i] for i in indices}
        try:
            runner.capture_backend = diagnostic_backend
            for index, original in originals.items():
                runner.finite_fla_by_layer[index] = wrapper(index, original, original_receipt)
            yield original_receipt
        finally:
            for index, original in originals.items():
                runner.finite_fla_by_layer[index] = original
            runner.capture_backend = original_backend
            report['unconsumed_output_layer_indices'] = sorted(retained)
            report['retained_cpu_output_bytes_before_cleanup'] = retained_bytes()
            retained.clear()
            report['retained_cpu_output_bytes_after_cleanup'] = retained_bytes()
            report['recorded_layer_slot_scalars'] = len(report['layers'])
            report['expected_layer_slot_scalars'] = len(slots)*report['actual_original_finite_head_calls']
            report['expected_public_output_returns'] = len(indices)
            report['expected_norm_input_captures'] = len(indices)
