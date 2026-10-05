"""Isolated original-owner conditional-boundary observation.

One complete full-response-EOS trace temporarily saves its actual FP32 finite
coefficients for selected original slots, on CPU.  Later original single-EOS
root calls contract those coefficients against the runner's existing CPU root
checkpoints.  No native capture, finite formula, model forward or reference is
added or replaced.  The only persistent observation output is scalar metadata.
CPU contractions are diagnostics, not official GPU numerical acceptance.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import sys

import torch

from observe_textcraft_finite_effects import _call_site, observe_token_effects


def _source(function):
    function = inspect.unwrap(function)
    path = Path(inspect.getsourcefile(function)).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                function=function.__name__, first_line=function.__code__.co_firstlineno)


def _slots(slots):
    result = tuple(int(slot) for slot in slots)
    if len(result) != len(set(result)) or any(slot < 0 or slot >= 4 for slot in result):
        raise ValueError('Use distinct original slots from the native B4 group.')
    return result


def _tensor_metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                stride=list(value.stride()), bytes=value.numel()*value.element_size())


def _attribute_metadata(caller, attribute_code):
    """Read scalar call coordinates; no frame or tensor-valued local escapes."""
    current = caller
    try:
        while current is not None and current.f_code is not attribute_code:
            current = current.f_back
        if current is None:
            raise ValueError('The original attribute body is absent from this callback stack.')
        selection = current.f_locals['selection']
        return dict(prefix_cut=int(current.f_locals['prefix_start']),
                    suffix_length=int(selection.length), original_length=int(current.f_locals['original_length']),
                    logical_batch=int(selection.batch))
    finally:
        del current, caller


@contextmanager
def collect_actual_coefficients(runner, slots):
    """Yield one group's CPU coefficient bank plus its JSON-safe metadata.

    Usage:
        with collect_actual_coefficients(runner, slots) as coefficients:
            original_trace_token_attribution(... original full-EOS arguments ...)
        # coefficients['metadata'] is JSON-safe; coefficients['by_boundary']
        # contains CPU tensors used by contract_saved_native_roots below.

    The caller owns the bank's lifetime and discards it after both single-EOS
    root observations.  This context does not pass an attribute observer.
    """
    slots = _slots(slots)
    module = sys.modules[type(runner).__module__]
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    owner_copy = module._copy
    report = dict(scope=__doc__, original_slots=list(slots),
        owner_token_effect=_source(module._token_effect), owner_copy=_source(owner_copy),
        boundaries={}, diagnostics=[], total_retained_cpu_tensor_bytes=0,
        coefficient_storage='Per-original-slot CPU copies at actual native-owner callbacks; no CPU recomputation.',
        root_contraction_backend='Original _token_effect on CPU; not GPU tolerance validation.')
    bank = dict(slots=slots, by_boundary={}, metadata=report)

    # Reuse the previous passive owner ledger as well as its source-site
    # classifier.  Its wrapper calls the owner unchanged and returns the same
    # object; this outer wrapper only retains selected coefficient rows.
    with observe_token_effects(runner) as ledger:
        report['original_finite_effect_ledger'] = ledger
        original = module._token_effect

        def collect(m, x):
            value = original(m, x)
            site = _call_site(sys._getframe(1), attribute_code)
            if site['phase'] == 'seed_effect':
                boundary = '32'
            elif site['phase'] == 'layer_input_effect':
                boundary = str(site['current_layer_index'])
            else:
                return value
            try:
                coordinates = _attribute_metadata(sys._getframe(1), attribute_code)
                if coordinates['logical_batch'] != 4 or m.shape[0] != 4 or x.shape[0] != 8:
                    raise ValueError('The actual coefficient callback is not original B4/paired8.')
                if m.shape[1] != coordinates['suffix_length']:
                    raise ValueError('Actual coefficient length differs from the original suffix.')
                if boundary in bank['by_boundary']:
                    raise ValueError('A second full trace attempted to overwrite this group coefficient bank.')
                first_coordinates = report.get('coordinates')
                if first_coordinates is None:
                    report['coordinates'] = coordinates
                elif first_coordinates != coordinates:
                    raise ValueError('Original prefix/selection coordinates changed inside one full trace.')
                saved = {}
                for slot in slots:
                    # Existing owner transport, using its ordinary completed
                    # CPU copy.  No pinned asynchronous buffer lifecycle is
                    # introduced by this diagnostic.
                    saved[slot] = owner_copy(m[slot:slot+1], 'cpu', pinned_host=False)
                bank['by_boundary'][boundary] = saved
                metadata = {str(slot): _tensor_metadata(tensor) for slot, tensor in saved.items()}
                report['boundaries'][boundary] = dict(
                    boundary_kind='final_norm_input' if boundary == '32' else 'decoder_input',
                    decoder_index=None if boundary == '32' else int(boundary),
                    original_coefficient=_tensor_metadata(m), original_endpoints=_tensor_metadata(x),
                    source_site=site, original_finite_ledger_record_index=len(ledger['records'])-1,
                    selected_slot_copies=metadata,
                    retained_cpu_tensor_bytes=sum(tensor.numel()*tensor.element_size() for tensor in saved.values()))
                report['total_retained_cpu_tensor_bytes'] = sum(
                    row['retained_cpu_tensor_bytes'] for row in report['boundaries'].values())
            except Exception as exc:
                # Optional observation failure must not replace the owner's
                # original output or an exception raised by the original trace.
                report['diagnostics'].append(dict(boundary=boundary, error=repr(exc)))
            return value

        module._token_effect = collect
        try:
            yield bank
        finally:
            module._token_effect = original
            report['captured_boundary_indices'] = sorted(int(key) for key in bank['by_boundary'])
            report['all_33_actual_boundaries_present'] = report['captured_boundary_indices'] == list(range(33))
            report['retained_tensor_count'] = sum(len(values) for values in bank['by_boundary'].values())


@contextmanager
def contract_saved_native_roots(runner, coefficients, slots):
    """Yield scalars from existing single-EOS root captures and restore globals.

    Nest outside the existing root observer so its score-then-stop callback
    calls this wrapper before raising its original sentinel:

        with contract_saved_native_roots(runner, coefficients, slots) as scalars:
            original_observe_original_root(... fixed_cut=full_EOS_actual_cut)

    The original score function is called once and its exact returned object
    is returned unchanged.  Only the original attribute frame's already saved
    root inputs are read; no native hook/capture/forward is added.  The report
    retains neither frames, root dictionaries, activations nor coefficients.
    """
    slots = _slots(slots)
    if slots != coefficients['slots']:
        raise ValueError('Use the same original slot order as the full-EOS coefficient bank.')
    module = sys.modules[type(runner).__module__]
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    original_score = module.selected_target_log_probs
    token_effect = module._token_effect
    report = dict(scope=__doc__, original_slots=list(slots), owner_score=_source(original_score),
        owner_token_effect=_source(token_effect), score_calls=0, boundaries=[], diagnostics=[],
        contraction_backend='CPU: original owner _token_effect; saved FP32 coefficients and original BF16 root captures.',
        retained_output='Scalar sums and shape/source metadata only; no new activation capture.',
        boundary_scope='Complete original suffix, including propagation after the deleted source token.',
        comparison_scope='Full-span coefficients projected against an actual single-deletion activation difference; not a new finite rule or an automatic error criterion.')

    def contract_from_frame(caller):
        current = caller
        root = endpoints = coefficient = value = None
        try:
            while current is not None and current.f_code is not attribute_code:
                current = current.f_back
            if current is None:
                raise ValueError('The original attribute body is absent at the score callback.')
            coordinates = _attribute_metadata(current, attribute_code)
            if coordinates != coefficients['metadata'].get('coordinates'):
                raise ValueError('Single root does not retain the full-EOS prefix/suffix layout.')
            root = current.f_locals['root']
            for boundary in sorted(coefficients['by_boundary'], key=int):
                native_key = 'final_norm_input' if boundary == '32' else boundary
                endpoints = root[native_key]
                if endpoints.device.type != 'cpu' or endpoints.shape[0] != 8:
                    raise ValueError('Expected the original CPU paired8 root checkpoint.')
                for slot in slots:
                    coefficient = coefficients['by_boundary'][boundary][slot]
                    if coefficient.device.type != 'cpu' or coefficient.shape != endpoints[2*slot:2*slot+1].shape:
                        raise ValueError('Saved coefficient and actual native endpoint dimensions differ.')
                    # Invoke only the owner's contraction, in original row
                    # order, preserving its FP64 difference and multiplication.
                    # A slice is used transiently; no activation clone is kept.
                    value = token_effect(coefficient, endpoints[2*slot:2*slot+2])
                    report['boundaries'].append(dict(boundary_index=int(boundary),
                        boundary_kind='final_norm_input' if boundary == '32' else 'decoder_input',
                        original_slot=slot, native_root_key=native_key,
                        original_paired_rows=[2*slot, 2*slot+1],
                        coefficient=_tensor_metadata(coefficient),
                        native_root_checkpoint=_tensor_metadata(endpoints),
                        owner_token_effect_dtype=str(value.dtype),
                        full_suffix_contraction=float(value.detach().sum()),
                        prefix_cut=coordinates['prefix_cut'], suffix_length=coordinates['suffix_length']))
        finally:
            # In particular, do not retain the attribute frame/root via this
            # wrapper or its exception path after the outer sentinel returns.
            del current, caller, root, endpoints, coefficient, value

    def score_and_contract(*args, **kwargs):
        value = original_score(*args, **kwargs)
        report['score_calls'] += 1
        try:
            contract_from_frame(sys._getframe(1))
        except Exception as exc:
            report['diagnostics'].append(dict(score_call=report['score_calls'], error=repr(exc)))
        return value

    module.selected_target_log_probs = score_and_contract
    try:
        yield report
    finally:
        module.selected_target_log_probs = original_score
        report['recorded_boundary_slot_scalars'] = len(report['boundaries'])
        report['expected_boundary_slot_scalars'] = 33*len(slots)
