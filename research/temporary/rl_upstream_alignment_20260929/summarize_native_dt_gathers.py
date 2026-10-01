"""Compare saved profile outputs by factual row identity; no model calls."""
import argparse
import subprocess

from stage_environment_entry import AUDIT, ENTRY, ROOT, SCP, SSH, remote

parser = argparse.ArgumentParser()
parser.add_argument('receipt_name')
args = parser.parse_args()
assert args.receipt_name.startswith('native-dt-gathers-') and '/' not in args.receipt_name
remote_dir = ROOT + '/receipts/owner-b8-dispatch-20260930/' + args.receipt_name
remote(fr'''source {ENTRY}/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json
from pathlib import Path
import torch
root = Path('{remote_dir}')
data = json.loads((root/'completed.json').read_text())
summary = dict(source=str(root/'completed.json'), modes={{}}, comparison={{}},
    scope='Saved factual rows repeated once. Two native B4 calls per rank per layout, not a formal iteration. Stream event sums overlap and must not be added to wall time. Differences below are observations, not a new tolerance.')
canonical = {{}}
for mode, workers in data['modes'].items():
    canonical[mode] = {{}}
    details = []
    for record in workers:
        tensors = torch.load(record['output_path'], map_location='cpu', weights_only=True)
        for local, (row_id, width) in enumerate(zip(record['row_indices'], record['response_lengths'])):
            assert row_id not in canonical[mode]
            canonical[mode][row_id] = {{key:value[local, :width].clone() for key,value in tensors.items()}}
        timings = {{}}
        for event in record['events']:
            timings[event['kind']] = timings.get(event['kind'], 0.) + event['stream_seconds']
        details.append(dict(rank=record['rank'], worker_seconds=record['worker_seconds'],
            lengths=record['lengths'], finite_outputs=record['finite_outputs'],
            parameter_all_gathers=len(record['gathers']),
            cpu_parameter_source_bytes=sum(row['cpu_source_bytes'] for row in record['gathers']),
            overlapping_stream_event_seconds=timings, restored=record['restored']))
    summary['modes'][mode] = dict(workers=details,
        max_worker_seconds=max(row['worker_seconds'] for row in workers))
left, right = canonical['contiguous'], canonical['owner_balanced']
assert set(left) == set(right) == set(range(16))
for key in left[0]:
    a = torch.cat([left[i][key] for i in sorted(left)]).double()
    b = torch.cat([right[i][key] for i in sorted(right)]).double()
    assert a.shape == b.shape
    delta = (a-b).abs()
    summary['comparison'][key] = dict(num_values=a.numel(), exact_equal=bool(torch.equal(a,b)),
        max_abs_difference=float(delta.max()), mean_abs_difference=float(delta.mean()),
        contiguous_max_abs=float(a.abs().max()), balanced_max_abs=float(b.abs().max()),
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
summary['identical_fixture_repeats'] = {{}}
for mode, workers in data['modes'].items():
    identities = {{}}
    for record in workers:
        saved = json.loads((root/f"inputs-{{mode}}-{{record['pid']}}.json").read_text())
        for row_id, sample in zip(record['row_indices'], saved['samples']):
            identity = (tuple(sample['selected_input_ids']), sample['source_start'],
                        sample['source_end'], sample['observed_return'])
            identities.setdefault(identity, []).append(row_id)
    pairs = []
    for ids in identities.values():
        assert len(ids) == 2
        a = canonical[mode][ids[0]]['dt_token_advantages'].double()
        b = canonical[mode][ids[1]]['dt_token_advantages'].double()
        pairs.append(dict(rows=ids, max_abs_difference=float((a-b).abs().max())))
    summary['identical_fixture_repeats'][mode] = pairs
summary['bounded_max_worker_speed_ratio'] = summary['modes']['contiguous']['max_worker_seconds'] / summary['modes']['owner_balanced']['max_worker_seconds']
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
''')
out = AUDIT / args.receipt_name
out.mkdir(exist_ok=True)
for name in ['completed.json', 'summary.json']:
    subprocess.run(SCP + [f'{SSH[-1]}:{remote_dir}/{name}', str(out/name)], check=True)
