"""Stop manifested obsolete attempts, preserve checkpoints, launch native bounded checks."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
release=root/'releases/64e5876'
manifest=json.loads((root/'formal-training.json').read_text())
expected={'Webshop':(4185212,26677),'AppWorld':(3563554,3615395)}
rec=root/'receipts/rollout-major-cost/linear-terminal-migration.json'
assert not rec.exists()
stopped=[]
for task,(pid,worker) in expected.items():
    job=next(j for j in manifest['jobs'] if j['task']==task)
    assert job['pid']==pid and os.getpgid(pid)==pid and os.getpgid(worker)==pid
    assert str(Path(job['run_dir'])/'run.sh') in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
    if Path('/proc/931296').exists():
        assert os.getpgid(931296) != pid  # Never stop the separate Sokoban pilot.
    marker=Path(job['checkpoint_dir'])/'latest_checkpointed_iteration.txt'
    stopped.append(dict(task=task,pid=pid,worker=worker,stopped_at=time.time(),
        checkpoint=marker.read_text().strip() if marker.exists() else None,
        run_dir=job['run_dir'],reason='Superseded event-target attempt; validate current whole-return PLAN with bounded official training first'))
rec.write_text(json.dumps(dict(stopped=stopped,status='stopping'),indent=2)+'\n')
for job in stopped:
    os.killpg(job['pid'],signal.SIGTERM)
time.sleep(3)
for job in stopped:
    path=Path(f"/proc/{job['worker']}/stat")
    assert not path.exists() or path.read_text().split(') ',1)[1].split()[0]=='Z', 'Wait for the same stop; never launch a duplicate GPU owner'
for name in ['formal-training.json','active-training.json']:
    p=root/name;m=json.loads(p.read_text())
    for j in m['jobs']:
        if j['task'] in expected:
            j.update(status='stopped_for_linear_return_validation',stop_receipt=str(rec),stopped_at=time.time())
    p.write_text(json.dumps(m,indent=2)+'\n')
jobs=[]
for task,gpu in [('Webshop',5),('AppWorld',6)]:
    original=next(j for j in manifest['jobs'] if j['task']==task)
    directory=root/'runs/linear-return-pilot'/task
    directory.mkdir(exist_ok=False)
    settings={**original['settings'],'DT_ROOT':str(release),'DT_ENVIRONMENT_JSON':str(release/'environment.json'),
        'TRAIN_SIZE':'1','GROUP_SIZE':'4','TOTAL_EPOCHS':'2','VAL_SIZE':'1','VAL_DATA_SIZE':'1',
        'DT_RAY_NUM_CPUS':'8','TEST_FREQ':'-1','RESUME_MODE':'disable',
        'DATA_ROOT':str(directory/'data'),'ROLLOUT_DATA_DIR':str(directory/'rollouts'),
        'CHECKPOINT_DIR':str(directory/'checkpoints')}
    script=['#!/bin/bash']+['export '+k+'='+shlex.quote(v) for k,v in settings.items()]
    script+=['source "$DT_ROOT/experiments/rl/environments/metax.env.sh"',
             'bash "$DT_ROOT/experiments/rl/run_verl_agent.sh" '+shlex.join(original['native_hydra_overrides']),
             'rc=$?',"printf '%s\\n' \"$rc\" > "+shlex.quote(str(directory/'exit-code')),'exit "$rc"']
    (directory/'run.sh').write_text('\n'.join(script)+'\n')
    subprocess.run(['bash','-n',str(directory/'run.sh')],check=True)
    with (directory/'train.log').open('w') as log:
        proc=subprocess.Popen(['bash',str(directory/'run.sh')],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job=dict(task=task,pid=proc.pid,started=time.time(),gpu=gpu,run_dir=str(directory),
        log=str(directory/'train.log'),checkpoint_dir=settings['CHECKPOINT_DIR'],settings=settings,
        status='initializing',scope='Two native iterations x four actual episodes. Full original task horizon and response limits, 32768 total, actor/DT B4; not formal budget or performance evaluation.')
    (directory/'job.json').write_text(json.dumps(job,indent=2)+'\n');jobs.append(job)
rec.write_text(json.dumps(dict(stopped=stopped,status='new_pilots_launched',jobs=jobs),indent=2)+'\n')
for name in ['active-training.json','active-source.json']:
    p=root/name;m=json.loads(p.read_text());m['bounded_validation']=dict(
        manifests=[str(root/'runs/linear-return-pilot'/task/'job.json') for task in ['Sokoban','Webshop','AppWorld']],
        release=str(release),scope='Current PLAN, three bounded real-task checks; old formal attempts stopped with artifacts preserved')
    p.write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(dict(stopped=stopped,jobs=[{k:j[k] for k in ['task','pid','gpu','run_dir']} for j in jobs])))
