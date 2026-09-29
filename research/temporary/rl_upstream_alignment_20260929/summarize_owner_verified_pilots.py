"""Summarize original completed metrics without inventing missing measurements."""
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent
snapshot=json.loads((root/'owner-verified-pilots-latest.json').read_text())
manifest=json.loads((root/'owner-verified-pilots-manifest.json').read_text())
result=dict(scope='Two iterations and two interactions per iteration; native interface/lifecycle check, not formal task performance or 32k capacity.',
    source_snapshot='owner-verified-pilots-latest.json',checked_unix=snapshot['checked_unix'],
    release='7431001',jobs=[])
for job,setup in zip(snapshot['jobs'],manifest):
    metrics={}
    for item in job['completed_metrics']:
        row=dict(re.findall(r'([\w/(). -]+):(-?(?:\d+\.\d+|\d+|nan|inf))',item['line']))
        # Original console formatter rounds to three decimals. Retain that scope.
        row={key.strip().removeprefix('- ').strip():float(value) for key,value in row.items()}
        if 'step' not in row:
            row={**row,'step':int(re.search(r'step:(\d+)',item['line'])[1])}
        metrics[int(row['step'])]=dict(source=item['source'],values=row)
    result['jobs'].append(dict(task=job['task'],method=job['method'],devices=job['devices'],
        exit_code=job['exit_code'],completed_checkpoint=job['checkpoint_marker'],
        raw_vllm=job['raw_vllm'],budget=setup['budget'],iterations=list(metrics.values()),
        recorded_errors=job['errors']))
result['total_raw_vllm_tokens']=sum(j['raw_vllm']['tokens'] for j in result['jobs'])
result['total_raw_vllm_nonfinite']=sum(j['raw_vllm']['nonfinite'] for j in result['jobs'])
result['limitations']=[
    'All tasks had zero reported task success in this deliberately two-interaction check; not a task-success nonregression result.',
    'No OOM or nonfinite DT error in these bounded runs does not resolve the historical DT failure.',
    'Original rollout/actor probability differences are retained and separately diagnosed, not assigned a new pass threshold.',
    'Printed gradient 0.000 is rounded, not proof of a zero gradient.',
    'Allocator/host-memory metrics from the trainer are not physical GPU peak or cgroup anonymous memory.',
    'Checkpoint writes contribute 71–198 seconds per iteration in these four concurrent jobs; not computation time.'
]
(root/'owner-verified-pilots-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ('total_raw_vllm_tokens','total_raw_vllm_nonfinite')}))
