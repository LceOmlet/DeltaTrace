"""Compare already saved forward-boundary scalars, without another model call.

These RMS summaries are independent of which adjoint is injected later. A
first differing summary is not proof of the first differing tensor/operator.
"""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
files = [HERE / 'native-full-policy-direction-exchange-v86-status-final.json',
         HERE / 'native-full-policy-saved-update-feedback-v90-status-final.json']
states = [json.loads(p.read_bytes())['report'] for p in files]

def select(state, case, rank):
    matches = [item for item in state['chain_reports']
               if item['value']['iteration'] == case and item['value']['rank'] == rank]
    assert len(matches) == 1
    return matches[0]

def summarize(item):
    found = defaultdict(list)
    for record in item['value']['records']:
        values = {k: v for k, v in record.items()
                  if k in ('output_shape', 'dtype', 'input_requires_grad')
                  or k.endswith('_forward_RMS_median_all') or k.endswith('_forward_RMS_range_all')}
        found[(record['micro'], record['boundary'])].append(values)
    return found

rows = []
sources = []
for rank in (0, 1):
    old = select(states[0], 'full80_native', rank)
    replay = select(states[1], 'old80_before', rank)
    sources.extend([dict(path=v['path'], SHA256=v['SHA256']) for v in (old, replay)])
    a, b = summarize(old), summarize(replay)
    assert set(a) == set(b)
    for micro in sorted({k[0] for k in a}):
        diffs = []
        def order(key):
            boundary = key[1]
            if '.layers.' in boundary:
                return int(boundary.rsplit('.', 1)[1])
            return 32 if boundary.endswith('.model.norm') else 33
        keys = sorted([k for k in a if k[0] == micro], key=order)
        for key in keys:
            if a[key] != b[key]:
                changes = []
                assert len(a[key]) == len(b[key])
                for i, (left, right) in enumerate(zip(a[key], b[key])):
                    for name in sorted(set(left) | set(right)):
                        if left.get(name) != right.get(name):
                            changes.append(dict(event=i, field=name,
                                original=left.get(name), replay=right.get(name)))
                diffs.append(dict(boundary=key[1], changes=changes))
        rows.append(dict(rank=rank, micro=micro, boundaries_compared=len(keys),
            differing_boundary_count=len(diffs),
            first_differing_saved_summary=None if not diffs else diffs[0],
            all_differing_boundaries=[d['boundary'] for d in diffs]))

report = dict(recorded_at=datetime.now().astimezone().isoformat(), scope=__doc__,
    sources=sources, inputs=[dict(path=str(p), SHA256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
    rows=rows, model_calls=0, backward_calls=0, optimizer_calls=0,
    limits=['An RMS summary may remain equal while tensor values differ. These scalars cannot establish full activation or in-memory state equality.',
        'A different saved summary localizes a diagnostic comparability gap, not the historical training root cause or an accepted official-kernel defect.'])
path = HERE / 'existing-forward-scalar-discrepancy-v98.json'
path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(path=str(path.resolve()), rows=rows)))
