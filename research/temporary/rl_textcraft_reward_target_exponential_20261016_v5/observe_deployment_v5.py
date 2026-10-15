"""Record only actual v5 identity, initial RPC binding and original log phase."""
import hashlib
import json
from pathlib import Path
import psutil
import runpy
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'
read=lambda path:json.loads(Path(path).read_bytes())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
job=read(AUDIT/'DTPO-v5-job.json');run=Path(job['run'])
driver=psutil.Process(job['pid']);assert driver.create_time()==job['birth'] and driver.status()!=psutil.STATUS_ZOMBIE
override=run/'runtime-overrides/initial-state-and-existing-passive-records-v5.json'
initial=read(override);assert initial['complete'] and initial['initial_optimizer0_scheduler0_verified']
assert all(w['credit_mode']=='reward_target_exponential_v5' for w in initial['actual_v5_environment'])
worker_states=[]
for worker in initial['workers']:
    p=psutil.Process(worker['pid']);assert p.create_time()==worker['birth'] and p.status()!=psutil.STATUS_ZOMBIE
    worker_states.append(dict(pid=p.pid,birth=p.create_time(),rank=worker['rank'],phase=p.name()))
owner=runpy.run_path(str(ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'),run_name='v5_read_existing_logs')
snapshot=owner['collect'](str(ROOT),[str(run/'textcraft-dt')]);log=snapshot['jobs'][0]
latest=max(log['metrics'],key=lambda x:x['step'],default=None)
ppo=read(ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json')
pp=psutil.Process(ppo['pid']);assert pp.create_time()==ppo['birth'] and pp.status()!=psutil.STATUS_ZOMBIE
report=dict(observed_unix=time.time(),version=job['version'],driver=dict(pid=driver.pid,birth=driver.create_time()),
    workers=worker_states,completed_formal_iterations=latest['step'] if latest else 0,
    latest_completed_metrics=latest,original_log_phases=log.get('phases'),original_log_errors=log.get('errors'),
    completed_initial_runtime_override=dict(path=str(override),SHA256=sha(override)),
    PPO_identity_preserved=dict(pid=pp.pid,birth=pp.create_time()),
    scope='Actual initial owner/import/env verification and original logs; initialization is not a completed healthy update.',
    old_gradient_rootcause_verified=False)
status='running_initial_rollout_actual_workers_verified' if not latest else 'running_completed_native_iteration_observed'
job['status']=status
for path in (AUDIT/'DTPO-v5-job.json',run/'job.json'):path.write_text(json.dumps(job,indent=2)+'\n')
launch=read(run/'launch.json');launch['status']=status
(run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
source=read(run/'source.json');source['deployment'].update(status=status,initial_runtime_override=report['completed_initial_runtime_override'])
(run/'source.json').write_text(json.dumps(source,indent=2)+'\n')
for name in ('formal-training.json','active-training.json','active-source.json'):
    path=ROOT/name;manifest=read(path)
    selected=[j for j in manifest['jobs'] if j.get('task')=='TextCraft' and j.get('method')=='dt']
    assert len(selected)==1 and selected[0]['pid']==job['pid']
    selected[0].update(status=status,initial_worker_binding_receipt=str(override),completed_runtime_overrides=[str(override)])
    path.write_text(json.dumps(manifest,indent=2)+'\n')
(AUDIT/'deployment-actual-observation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
