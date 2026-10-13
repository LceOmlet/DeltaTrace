"""Recover already logged native direct-target workload for the stopped DT run."""
import json
from pathlib import Path
import re
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
LOG = Path('/tmp/ray/session_2026-10-12_16-27-07_350166_3172194/logs/worker-9a450e7e7c2504bce6c1bffe137646be8b6f83147ffe8b00566b72cb-01000000-3175861.out')
pattern = re.compile(r'\[DT direct target workload\] trajectories=(\d+) nonzero_requests=(\d+) target_tokens=(\d+) source_tokens=(\d+)')
rows = []
for number, line in enumerate(LOG.read_text(errors='replace').splitlines(),1):
    match = pattern.search(line)
    if match:
        n, requests, target, source = map(int, match.groups())
        rows.append({'call_order':len(rows)+1,'trajectories':n,'nonzero_requests':requests,
            'full_history_target_tokens':target,'full_history_other_policy_tokens':source,
            'full_history_policy_tokens':target+source,
            'source':{'path':str(LOG),'line':number}})
report = {'unix':time.time(),'scope':__doc__,'model_GPU_calls':0,'rows':rows,
    'limits':'Call order will be aligned to original completed-iteration records before interpretation. Counts describe unique full-history trajectories before DT filtering and native retained-response mapping; they do not automatically equal the PPO retained policy-token counts.'}
path=OUT/'native-workload-temporal-v1.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
