"""Passive original symmetric-FLA endpoint-order projections.

Observe the original averaged callback's RETURN locals after its two original
finite calls, retaining selected five-input coefficients on CPU. Native
contractions reuse the existing FLA observer's exact paired CPU operands and
original _token_effect; no extra native event, copy, model or FLA call is made.
Order differences/cancellation are descriptive, not a correction or tolerance.
"""
from contextlib import contextmanager
import importlib
import inspect
import sys

import observe_textcraft_conditional_fla as existing
from observe_textcraft_conditional_boundaries import (
    _attribute_metadata, _slots, _source, _tensor_metadata,
)
from observe_textcraft_conditional_primitives import _layers


INPUT_FIELDS = existing.INPUT_FIELDS
DEFAULT_LAYERS = existing.DEFAULT_LAYERS


@contextmanager
def collect_actual_fla_orders(runner, bank, slots, *, layers=DEFAULT_LAYERS):
    """Compose the original FLA collector; its averaged result is unchanged.

    The monitor exists only inside the original averaged callback invocation,
    after native replay capture has exited. The original averaging code, its
    two inner calls, all argument objects and its returned object are retained.
    """
    slots = _slots(slots)
    if slots != bank['slots']:
        raise ValueError('Use the original coefficient-bank slot order.')
    indices, _owner_layers = _layers(runner, layers)
    if 'fla_order_bank' in bank:
        raise ValueError('Do not overwrite an existing endpoint-order bank.')
    module = sys.modules[type(runner).__module__]
    owner_copy, owner_effect = module._copy, module._token_effect
    gdn_code = inspect.unwrap(module.gdn_finite_pullback).__code__
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    events = importlib.import_module('accelerated.native_capture_events').LocalCaptureEvents
    originals = {index:runner.finite_fla_by_layer[index] for index in indices}
    entries = bank['fla_order_bank'] = {index:dict(groups=[]) for index in indices}
    metadata = dict(scope=__doc__, original_slots=list(slots), layer_indices=list(indices),
        owner_copy=_source(owner_copy), owner_token_effect=_source(owner_effect),
        original_local_events=_source(events.__init__), layers={}, diagnostics=[],
        order0='Original reference/factual rows; native_input_adjoints reads factual rows1::2.',
        order1='Original pair permutation; native_input_adjoints reads original reference rows1::2.',
        upstream='The same original incoming do object is supplied to both orders; no seed is recomputed.',
        contraction_orientation='Both returned B4 coefficient dictionaries contract original factual-minus-reference native operands; no reverse-row/sign transformation.')

    def wrapper(index, original):
        original_code = inspect.unwrap(original).__code__
        if original_code.co_name != 'averaged':
            raise ValueError('Observe the original owner averaged callback before installing the prior collector.')
        info = metadata['layers'][str(index)] = dict(
            original_outer_callback=_source(original), monitored_returns=0, groups=[])

        def observed(endpoints, upstream, scale):
            def on_return(frame, kind, returned):
                if kind != 'return' or frame.f_code is not original_code:
                    return
                info['monitored_returns'] += 1
                try:
                    local = frame.f_locals
                    if local['endpoints'] is not endpoints or local['upstream'] is not upstream:
                        raise ValueError('Monitored original callback arguments changed identity.')
                    coordinates = _attribute_metadata(frame, attribute_code)
                    layout = existing._actual_group(frame, gdn_code)
                    entry = dict(layout=layout, coordinates=coordinates, orders=[],
                                 native_time_start=layout['capture_start']+layout['cut'])
                    group_info = dict(layout=layout, coordinates=coordinates,
                        native_time_start=entry['native_time_start'], actual_scale=float(scale),
                        original_incoming_do=_tensor_metadata(upstream),
                        original_return_fields={name:_tensor_metadata(returned[name]) for name in INPUT_FIELDS},
                        orders=[])
                    for order, name in enumerate(('forward', 'reverse')):
                        result = local[name]
                        saved_slots = {slot:{} for slot in slots}
                        order_info = dict(order=order, original_local_name=name, fields={})
                        for field in INPUT_FIELDS:
                            coefficient = result[field]
                            order_info['fields'][field] = dict(original=_tensor_metadata(coefficient), selected={})
                            for slot in slots:
                                saved = owner_copy(coefficient[slot:slot+1].flatten(2), 'cpu', pinned_host=False)
                                saved_slots[slot][field] = saved
                                order_info['fields'][field]['selected'][str(slot)] = _tensor_metadata(saved)
                        entry['orders'].append(saved_slots)
                        group_info['orders'].append(order_info)
                    entries[index]['groups'].append(entry)
                    info['groups'].append(group_info)
                except Exception as exc:
                    metadata['diagnostics'].append(dict(where='original_avg_RETURN', layer=index, error=repr(exc)))

            with events([original_code], on_return, returns_only=True):
                # Exactly one invocation of the existing outer owner, which
                # itself still owns its two finite calls and arithmetic mean.
                return original(endpoints, upstream, scale)
        return observed

    original_receipt = None
    try:
        for index, original in originals.items():
            runner.finite_fla_by_layer[index] = wrapper(index, original)
        with existing.collect_actual_fla(runner, bank, slots, layers=layers) as original_receipt:
            original_receipt['conditional_orders'] = metadata
            yield original_receipt
    finally:
        for index, original in originals.items():
            runner.finite_fla_by_layer[index] = original
        metadata['retained_cpu_tensor_bytes'] = sum(value.numel()*value.element_size()
            for layer in entries.values() for group in layer['groups']
            for order in group['orders'] for fields in order.values() for value in fields.values())
        metadata['recorded_original_avg_returns'] = sum(layer['monitored_returns'] for layer in metadata['layers'].values())
        metadata['recorded_group_counts'] = {str(index):len(entry['groups']) for index,entry in entries.items()}
        metadata['expected_original_avg_returns'] = (None if original_receipt is None else
            sum(layer['calls'] for layer in original_receipt['layers'].values()))


@contextmanager
def contract_saved_native_fla_orders(runner, bank, slots):
    """Reuse the original observer's paired CPU operands without another copy.

    Tensor identity maps each already stored average coefficient to the two
    original order coefficients. The prior _effect runs first and its scalar
    object is returned unchanged. The same owner function and same paired CPU
    operand are then used only for the two passive contractions. DO/Y is not
    intercepted. Existing event, geometry, score and cleanup code is reused.
    """
    slots = _slots(slots)
    if slots != bank['slots']:
        raise ValueError('Use the original coefficient-bank slot order.')
    entries = bank['fla_order_bank']
    averages = bank['fla_bank']
    # The contextmanager may be reconstructed by serialization. Its actual
    # body globals, rather than a presumed module dictionary, own _effect.
    effect_globals = inspect.unwrap(existing.contract_saved_native_fla).__globals__
    original_effect = effect_globals['_effect']
    lookup = {}
    values = {}
    report = dict(scope=__doc__, original_slots=list(slots), layers=[], diagnostics=[],
        original_effect=_source(original_effect), reused_average_contraction_calls=0,
        added_CPU_order_contractions=0,
        interpretation='F0/F1 are original full-EOS order coefficients projected on the same original single-EOS native operands. Cancellation alone does not show which estimate is accurate.')
    for index, layer in entries.items():
        original_groups = averages[index]['groups']
        if len(layer['groups']) != len(original_groups):
            raise ValueError('Original order and average head-group counts differ.')
        for head, (group, average) in enumerate(zip(layer['groups'], original_groups, strict=True)):
            if group['layout'] != average['layout'] or group['coordinates'] != average['coordinates']:
                raise ValueError('Original order and average group geometry differ.')
            for slot in slots:
                values.setdefault((index, slot), {0:{},1:{}})
                for name in INPUT_FIELDS:
                    coefficient = average['slots'][slot][name]
                    orders = [order[slot][name] for order in group['orders']]
                    if any(tuple(value.shape) != tuple(coefficient.shape) for value in orders):
                        raise ValueError('Original selected order and average coefficient shapes differ.')
                    if id(coefficient) in lookup:
                        raise ValueError('An original coefficient identity was assigned twice.')
                    lookup[id(coefficient)] = (index, slot, head, name, orders)

    def observed_effect(owner, coefficient, paired):
        result = original_effect(owner, coefficient, paired)
        match = lookup.get(id(coefficient))
        if match is None:
            return result
        report['reused_average_contraction_calls'] += 1
        index, slot, head, name, orders = match
        try:
            for order, actual in enumerate(orders):
                target = values[index, slot][order].setdefault(head, {})
                if name in target:
                    raise ValueError('The same original native field would overwrite an order projection.')
                target[name] = original_effect(owner, actual, paired)
                report['added_CPU_order_contractions'] += 1
        except Exception as exc:
            report['diagnostics'].append(dict(where='existing_native_CPU_contraction',
                layer=index, original_slot=slot, head_group=head, field=name, error=repr(exc)))
        return result

    completed = False
    try:
        effect_globals['_effect'] = observed_effect
        with existing.contract_saved_native_fla(runner, bank, slots) as original_receipt:
            original_receipt['conditional_orders'] = report
            yield original_receipt
        for row in original_receipt['layers']:
            index, slot = row['decoder_index'], row['original_slot']
            projections = values[index, slot]
            try:
                F = []
                per_order = []
                for order in (0, 1):
                    groups = projections[order]
                    if sorted(groups) != list(range(len(entries[index]['groups']))) or any(set(fields) != set(INPUT_FIELDS) for fields in groups.values()):
                        raise ValueError('Original five-field/head-group order projections are incomplete.')
                    fields = {name:sum(group[name] for group in groups.values()) for name in INPUT_FIELDS}
                    F.append(sum(fields.values()))
                    per_order.append(dict(order=order, fields=fields,
                                          head_groups=[groups[head] for head in sorted(groups)]))
                average, Y = row['F_input_projection'], row['Y_public_output_projection']
                mean = (F[0]+F[1])*0.5
                report['layers'].append(dict(decoder_index=index, original_slot=slot,
                    paired_rows=row['paired_rows'], coordinates=row['coordinates'],
                    group_layouts=row['group_layouts'], orders=per_order,
                    F_order0=F[0], F_order1=F[1], actual_F_average=average, Y_public_output_projection=Y,
                    F_order0_minus_Y=F[0]-Y, F_order1_minus_Y=F[1]-Y,
                    actual_average_minus_Y=average-Y,
                    order_projection_mean=mean, actual_average_closure_difference=average-mean,
                    opposing_signs=F[0]*F[1] < 0,
                    mean_absolute_order_projection=(abs(F[0])+abs(F[1]))*0.5,
                    absolute_projection_cancellation=(abs(F[0])+abs(F[1]))*0.5-abs(mean)))
            except Exception as exc:
                report['diagnostics'].append(dict(where='order_scalar_report', layer=index,
                                                  original_slot=slot, error=repr(exc)))
        completed = True
    finally:
        effect_globals['_effect'] = original_effect
        lookup.clear()
        values.clear()
        report['recorded_layer_slot_scalars'] = len(report['layers'])
        report['expected_layer_slot_scalars'] = len(entries)*len(slots)
        report['expected_reused_average_contraction_calls'] = sum(
            len(layer['groups'])*len(slots)*len(INPUT_FIELDS) for layer in entries.values())
        if completed:
            bank['fla_order_single_calls'] = bank.get('fla_order_single_calls', 0)+1
        if not completed or bank.get('fla_order_single_calls') == 2:
            entries.clear()
