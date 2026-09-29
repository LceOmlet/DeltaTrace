import json
import re
from pathlib import Path
import sys

path = Path(sys.argv[1])
for raw in path.read_text(errors='replace').splitlines():
    line = re.sub(r'\x1b\[[0-9;]*m', '', raw)
    if '[DT EOS minimum] ' not in line:
        continue
    prefix, payload = line.split('[DT EOS minimum] ', 1)
    entry = json.loads(payload)
    samples = []
    for sample in entry['samples']:
        trace = sample['trace']
        samples.append(dict(uid=sample['traj_uid'], length=len(sample['selected_input_ids']),
            source=sample['source_signed'], **{key: trace[key] for key in (
                'observed_return', 'root_effect', 'signed_sum', 'conservation_residual',
                'factual_target_logp', 'reference_target_logp')}))
    print(json.dumps(dict(worker=prefix, samples=samples)))
