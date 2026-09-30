"""Preserve already-produced first-update evidence; no training is launched."""
import json
from pathlib import Path
import re
import subprocess
from stage_environment_entry import ROOT, AUDIT, REPO, SCP, SSH

directory=AUDIT/'formal-progress-20261001'
directory.mkdir(exist_ok=True)
sources={
    'progress.json':ROOT+'/runs/official-trajectory-20260930-v8/progress-current.json',
    'textcraft-optimizer.json':ROOT+'/receipts/owner-b8-dispatch-20260930/textcraft-formal-optimizer.json',
}
for name,source in sources.items():
    subprocess.run(SCP+[f'{SSH[-1]}:{source}',str(directory/name)],check=True)
progress=json.loads((directory/'progress.json').read_text())
job=next(j for j in progress['jobs'] if j['task']=='TextCraft')
optimizer=json.loads((directory/'textcraft-optimizer.json').read_text())
assert job['pid']==optimizer['driver_pid']==212110
metrics=[];readouts=[]
for evidence in job['evidence']:
    for line in evidence['lines']:
        if line.startswith('step:1 '):
            metrics.append(dict(source=evidence['path'],raw=line,
                values={k:float(v) for k,v in re.findall(r'([\w/]+):(-?[\d.]+)',line)}))
        if line.startswith('[DeltaTrace readout] '):
            readouts.append(dict(source=evidence['path'],report=json.loads(line.split('] ',1)[1])))
assert len(metrics)==1 and len(readouts)==2
runtime=json.loads((REPO/'experiments/rl/current_runtime.json').read_text(encoding='utf8'))
result=dict(
    scope='TextCraft first formal iteration and original optimizer state. SQL and AppWorld have not yet completed their first formal update.',
    observed_unix=progress['unix'],task='TextCraft',driver_pid=job['pid'],
    versions=runtime['version_mapping'],
    deployment=next(j['version_mapping'] for j in runtime['jobs'] if j['task']=='TextCraft'),
    metrics=metrics,dt_readouts=readouts,optimizer_observation=optimizer,
    next_rollout_evidence=[e for e in job['evidence'] if any('Rounds 2/30' in l for l in e['lines'])],
    sources=sources,
    limitations=[
        'timing_s/gen and timing_s/step include the earlier deliberate process pause, so they are not steady-state throughput.',
        'Original console metrics are rounded; optimizer step and moment observations are separate original-state evidence.',
        'DT conservation_failures is the existing residual diagnostic, not a FA/FLA official tolerance assertion; no numerical correction was added.',
        'No current formal checkpoint is completed; existing older checkpoint tests retain their narrower recorded scope.',
        'Original console CPU and torch allocator totals are not per-job physical memory. Per-process PSS and mx-smi observations remain in progress.json.',
    ])
target=REPO/'experiments/rl/results_first_formal_update.json'
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(target)
print(json.dumps(dict(step=metrics[0]['values']['step'],
    nonzero_dt_advantages=sum(r['report']['nonzero_advantages'] for r in readouts),
    optimizer_steps=[r['optimizer_steps'] for rank in optimizer['workers'] for r in rank])))
