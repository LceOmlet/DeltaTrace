"""CPU-only shape projection from the saved actual owner capture inventory.

This describes retained tensor payload, not physical peak or an OOM test.
It changes no capture, tensor layout, model call or memory policy.
"""
import hashlib
import json
from math import ceil, prod
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
INVENTORY = REPO / 'experiments/rl/results_native_root_capture_inventory_20261004.json'
GEOMETRY = REPO / 'experiments/rl/results_actual_prefix_request_geometry_20261004.json'


def project(row, suffix, context):
    fields = {item['field']: item for item in row['fields']}
    original_suffix = fields['decoder.gate_output']['shape'][1]
    original_context = (fields['mixer.key']['shape'][2]
                        if row['block_type'] == 'full_attention' else None)
    result = {}
    for group in row['storage_groups']:
        # Use the largest logical alias, once per original actual storage.
        field = max((fields[name] for name in group['fields']),
                    key=lambda item: item['logical_bytes'])
        before = field['shape']
        after = list(before)
        name = field['field']
        if name == 'endpoints.h':
            after[1] = ceil(suffix / 64)
        elif name == 'mixer.projected_qkv':
            after[2] = suffix + (before[2] - original_suffix)
        elif name == 'mixer.conv_output':
            after[2] = suffix
        else:
            for axis, size in enumerate(before):
                if size == original_suffix:
                    after[axis] = suffix
                elif original_context is not None and size == original_context:
                    after[axis] = context
        # Same native layout/dtype; preserve original storage/logical ratio.
        numerator = group['storage_bytes'] * prod(after)
        denominator = prod(before)
        assert numerator % denominator == 0, (name, before, after)
        kind = 'decoder' if name.startswith('decoder.') else row['block_type']
        result[kind] = result.get(kind, 0) + numerator // denominator
    return result


def source(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    raw = json.loads(INVENTORY.read_bytes())
    examples = ((0, 576, 7552), (1, 610, 7586),
                (0, 832, 32768), (0, 2630, 5382), (1, 3332, 6084))
    rows = []
    for rank, suffix, context in examples:
        inventory = raw['records'][f'root-capture-inventory-rank{rank}.json']['value']
        totals = {}
        for layer in inventory['rows']:
            for kind, size in project(layer, suffix, context).items():
                totals[kind] = totals.get(kind, 0) + size
        rows.append(dict(rank=rank, suffix=suffix, context=context,
                         bytes_by_capture=totals, retained_bytes=sum(totals.values()),
                         three_MLP_outputs_bytes=32 * 3 * 8 * suffix * 12288 * 2))
    value = dict(
        scope='CPU-only same-layout retained-payload projection; no model, GPU, allocation, physical peak, OOM claim or new threshold',
        formal_deployment=False, sources=[source(INVENTORY), source(GEOMETRY), source(Path(__file__))],
        assumptions=['Actual saved native dtype/layout and 32-layer capture fields retained unchanged',
                     'Aliases within each actual storage group counted once',
                     'Per-layer storage sums do not prove cross-layer/owner-cache alias accounting or liveness peak',
                     'Original inventory released captures each layer; projections are not measured simultaneous allocations',
                     'GDN h uses ceil(suffix/64); projected_qkv retains actual four-token conv context',
                     'Do not add retained payload to an independently measured baseline peak'],
        rows=rows)
    destination = REPO / 'experiments/rl/results_native_root_tape_retained_storage_20261005.json'
    with destination.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(destination), rows=rows), indent=2))
