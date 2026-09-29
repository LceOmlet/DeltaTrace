"""Read original experiment receipts; no training or acceptance thresholds."""
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
output = {}
for name in ('native-two-gpu-credit-native-lora', 'native-two-gpu-capacity32k',
             'native-task-length-cost-Webshop'):
    path = root / (name+'.json')
    if not path.exists():
        continue
    entry = json.loads(path.read_text())
    if 'runs' in entry:
        output[name] = {k: entry.get(k) for k in ('status', 'runs')}
    else:
        output[name] = entry

groups = {}
path = root / 'native-two-gpu-credit-native-lora.log'
for raw in path.read_text(errors='replace').splitlines():
    line = re.sub(r'\x1b\[[0-9;]*m', '', raw)
    if '[DT EOS minimum] ' not in line:
        continue
    prefix, payload = line.split('[DT EOS minimum] ', 1)
    for sample in json.loads(payload)['samples']:
        trace = sample['trace']
        identity = hashlib.sha256(json.dumps(sample['selected_input_ids']).encode()).hexdigest()
        groups.setdefault(identity, []).append(dict(worker=prefix.strip(), uid=sample['traj_uid'],
            source=sample['source_signed'], **{key: trace[key] for key in (
                'observed_return', 'root_effect', 'signed_sum', 'conservation_residual')}))
comparisons = []
for identity, rows in groups.items():
    baseline = rows[0]
    comparisons.append(dict(input_sha256=identity, uids=[r['uid'] for r in rows],
        workers=sorted(set(r['worker'] for r in rows)),
        max_credit_difference=max((max(abs(a-b) for a,b in zip(r['source'], baseline['source']))
                                   for r in rows), default=0),
        max_root_difference=max(abs(r['root_effect']-baseline['root_effect']) for r in rows),
        residuals=[r['conservation_residual'] for r in rows]))
output['identical_input_comparisons'] = comparisons
(root/'distributed-check-summary.json').write_text(json.dumps(output, indent=2)+'\n')
# Keep stdout small: full per-step metrics remain in the source receipts.
for name, entry in output.items():
    if isinstance(entry, dict) and 'tasks' in entry:
        entry['tasks'] = {task: {**{k:v for k,v in value.items() if k != 'metrics'},
            'grad_norm': value.get('metrics', {}).get('actor/grad_norm')} for task,value in entry['tasks'].items()}
print(json.dumps(output, indent=2))
