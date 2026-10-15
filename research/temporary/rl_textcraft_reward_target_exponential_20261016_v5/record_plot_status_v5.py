"""Record original plot artifacts and the applied read-only hourly configuration."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tomllib

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

output=Path(sys.argv[1])
snapshot=read(output/'source-snapshot.json')
receipt=read(output/'plot-receipt.json')
runtime_path=REPO/'experiments/rl/current_runtime.json'
runtime=read(runtime_path)
previous=runtime['textcraft_hourly_plot_latest']
if previous['collected_at']>=snapshot['collected_at']:
    old_output=Path(previous['previous_successful_output'])
    old=read(old_output/'plot-receipt.json')
    source=read(old_output/'source-snapshot.json')
    previous=dict(collected_at=source['collected_at'],output=str(old_output),
                  completed=old['completed_iterations'])
candidates=[previous]
for path in output.parent.glob('*/plot-receipt.json'):
    old=read(path);source=read(path.parent/'source-snapshot.json')
    if source['collected_at']<snapshot['collected_at']:
        candidates.append(dict(collected_at=source['collected_at'],output=str(path.parent),
                               completed=old['completed_iterations']))
previous=max(candidates,key=lambda p:p['collected_at'])
keys=('episode/reward/mean','actor/entropy_loss','actor/grad_norm','actor/kl_loss',
      'response_length/mean','episode/tool_call_count/mean')
latest={};phases={};sources={}
for job in snapshot['jobs']:
    method=job['method'];metrics=job['metrics']
    last=max(metrics,key=lambda p:p['step'],default=None)
    latest[method]={key:last['values'].get(key) if last else None for key in keys}
    sources[method]={k:last.get(k) for k in ('step','sources')} if last else None
    phases[method]=job.get('phase_label')
completed=receipt['completed_iterations']
summary=dict(collected_at=snapshot['collected_at'],
    collected_at_Beijing=dt.datetime.fromtimestamp(snapshot['collected_at'],
        dt.timezone(dt.timedelta(hours=8))).isoformat(),
    output=str(output),completed=completed,
    increments_since_previous_success={key:completed[key]-previous['completed'][key] for key in completed},
    previous_successful_output=previous['output'],latest=latest,latest_metric_sources=sources,
    phases=phases,DT_status='v5 first native rollout, no completed formal iteration',
    new_run='v5 starts at zero; v3/v4 histories are not joined',
    common_window=receipt['common_window'],training_changes=0,model_calls=0,goal_status='paused',
    scope='Original existing logs and plots only',summary=str(output/'hourly-summary.json'),
    receipt=str(output/'plot-receipt.json'),
    original_snapshot_sha256=sha(output/'source-snapshot.json'),
    CSV_sha256=sha(output/'metrics-with-sources.csv'),
    original_error_records={job['method']:job.get('errors',[]) for job in snapshot['jobs']})
write(output/'hourly-summary.json',summary)
write(HERE/'receipts/deployment-plot-summary.json',summary)
runtime['textcraft_hourly_plot_latest']=summary
runtime['latest_actual_training_plot_snapshot']=str(output)
write(runtime_path,runtime)

automation=tomllib.loads(Path('C:/Users/Administrator/.codex/automations/textcraft/automation.toml').read_text(encoding='utf-8'))
script=HERE/'plot_current_algorithms_v5.py'
assert automation['id']=='textcraft' and automation['status']=='ACTIVE'
assert str(script.relative_to(REPO)).replace('\\','/') in automation['prompt']
assert sha(script) in automation['prompt']
record=dict(id=automation['id'],status=automation['status'],kind=automation['kind'],
    name=automation['name'],target_thread_id=automation['target_thread_id'],
    updated_at=automation['updated_at'],script=str(script),script_sha256=sha(script),
    prompt_sha256=hashlib.sha256(automation['prompt'].encode()).hexdigest(),
    training_actions_authorized_in_hourly_task=False)
write(HERE/'receipts/hourly-automation-v5.json',record)
print(json.dumps(summary,ensure_ascii=False,indent=2))
