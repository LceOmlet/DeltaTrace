"""Replace only the user's current DTPO with prepared, verified fresh v3."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
PREPARED=OUT/'DTPO-variance-v3-prepared'
record=json.loads((PREPARED/'launch.json').read_bytes())
source=json.loads((PREPARED/'source.json').read_bytes())
verification=PREPARED/'interface-verification.json'
checked=json.loads(verification.read_bytes());assert all(checked['checks'].values())
commit=sys.argv[1];assert re.fullmatch('[0-9a-f]{40}',commit)
job_path=OUT/'DTPO-variance-v3-job.json';run=Path(record['run'])
assert not job_path.exists() and not run.exists(),'Read the submitted job rather than submitting twice'
for path,digest in record['source_bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
old=json.loads((OUT/'DT-group-multiplier-v40-job.json').read_bytes())
ppo=json.loads((OUT/'native-PPO-control-v9-job.json').read_bytes())
protected=[];protected_pids=set()
protected_identities=[(3174448,1791793629.73),(ppo['pid'],ppo['birth'])]
if psutil.pid_exists(2614045) and psutil.Process(2614045).create_time()==1791949034.31:
    protected_identities.append((2614045,1791949034.31))
for pid,birth in protected_identities:
    p=psutil.Process(pid);assert p.create_time()==birth and p.status()!=psutil.STATUS_ZOMBIE
    protected.append(dict(pid=pid,birth=birth));protected_pids.add(pid)
    protected_pids.update(c.pid for c in p.children(recursive=True))
stop=PREPARED/'previous-DTPO-stop.json'
if not stop.exists():
    p=psutil.Process(old['pid']);assert p.create_time()==old['birth']
    assert os.getpgid(p.pid)==os.getsid(p.pid)==p.pid
    checkpoint=Path(old['run'])/'textcraft-dt/checkpoints'
    marker=checkpoint/'latest_checkpointed_iteration.txt';step=int(marker.read_text())
    for rank in (0,1):
        for kind in ('model','optim','extra_state'):
            f=checkpoint/f'global_step_{step}/actor/{kind}_world_size_2_rank_{rank}.pt'
            assert f.is_file() and f.stat().st_size>0
    assert (checkpoint/f'global_step_{step}/data.pt').is_file()
    members={p.pid:p};members.update({c.pid:c for c in p.children(recursive=True)})
    for q in psutil.process_iter():
        try:
            if os.getpgid(q.pid)==p.pid:members[q.pid]=q
        except (psutil.NoSuchProcess,ProcessLookupError):pass
    assert not protected_pids.intersection(members)
    identities=[dict(pid=q.pid,birth=q.create_time()) for q in members.values()]
    report=dict(unix=time.time(),job=old,checkpoint_step_preserved=step,
        checkpoint_directory=str(checkpoint/f'global_step_{step}'),members=identities,
        instruction='User requests fresh DTPO v3 restart with variance-only normalization.',
        protected=protected,new_checkpoint_created=False,status='validated_before_stop')
    stop.write_text(json.dumps(report,indent=2)+'\n')
    os.killpg(p.pid,signal.SIGTERM)
    _,alive=psutil.wait_procs(list(members.values()),timeout=12)
    for q in alive:
        try:
            birth=next(v['birth'] for v in identities if v['pid']==q.pid)
            if q.create_time()==birth and q.status()!=psutil.STATUS_ZOMBIE:
                assert q.pid not in protected_pids;q.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(alive,timeout=5)
    survivors=[]
    for identity in identities:
        try:
            q=psutil.Process(identity['pid'])
            if q.create_time()==identity['birth'] and q.status()!=psutil.STATUS_ZOMBIE:survivors.append(identity)
        except psutil.NoSuchProcess:pass
    report.update(status='stopped_for_user_requested_fresh_v3',survivors=survivors)
    stop.write_text(json.dumps(report,indent=2)+'\n');assert not survivors
else:
    assert json.loads(stop.read_bytes())['status']=='stopped_for_user_requested_fresh_v3'
    assert not psutil.pid_exists(old['pid']) or psutil.Process(old['pid']).create_time()!=old['birth']
for identity in protected:assert psutil.Process(identity['pid']).create_time()==identity['birth']
physical=subprocess.check_output(['mx-smi'],text=True,timeout=30)
(PREPARED/'before-v3-launch-mx-smi.txt').write_text(physical)
assert not re.search(r'\|\s*[01]\s+\d+\s+',physical.split('| Process:')[-1]),'GPU0/1 remain occupied'
env=dict(os.environ);env.update(record['effective_environment']);env['CUDA_VISIBLE_DEVICES']='0,1'
env.pop('RAY_ADDRESS',None);env.pop('RAY_TMPDIR',None)
assert record['argv'][0]==env['VENV_PYTHON'] and 'VERL_GLOBAL_TOKEN_MEAN' not in env
run.mkdir(parents=True);(run/'effective-config.yaml').write_bytes((PREPARED/'effective-config.yaml').read_bytes())
record.update(local_patch_commit=commit,status='submitted_v3_initialization_pending',
    interface_verification=str(verification),interface_verification_SHA256=hashlib.sha256(verification.read_bytes()).hexdigest(),
    prior_job=dict(pid=old['pid'],birth=old['birth'],run=old['run'],status='stopped_not_resumed'))
with (run/'driver.stdout.log').open('wb') as stdout,(run/'driver.stderr.log').open('wb') as stderr:
    p=subprocess.Popen(record['argv'],cwd=env['VERL_ROOT'],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
record.update(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time())
source.update(local_patch_commit=commit,prepared_only=False,deployment=dict(status=record['status'],pid=p.pid,birth=record['birth']))
(run/'launch.json').write_text(json.dumps(record,indent=2)+'\n');(run/'source.json').write_text(json.dumps(source,indent=2)+'\n')
identity={k:record[k] for k in ('pid','birth','started_unix','run','version','method','physical_gpus','local_patch_commit','upstream_commit','source_bindings','status')}
identity.update(protected_jobs=protected,rootcause_repair_verified=False,hourly_automation='PAUSED')
job_path.write_text(json.dumps(identity,indent=2)+'\n');(run/'job.json').write_bytes(job_path.read_bytes())
snapshots=run/'deployment-receipts';snapshots.mkdir()
for name in ('formal-training.json','active-training.json','active-source.json'):
    path=ROOT/name;raw=path.read_bytes();manifest=json.loads(raw)
    matches=[j for j in manifest['jobs'] if j.get('task')=='TextCraft' and j.get('method')=='dt']
    assert len(matches)==1 and matches[0]['pid']==old['pid']
    j=matches[0];unaffected=[x.copy() for x in manifest['jobs'] if x is not j]
    (snapshots/(name+'.before-v3.json')).write_bytes(raw)
    j.clear();j.update(task='TextCraft',method='dt',pid=p.pid,birth=record['birth'],
        observed_process_created_unix=record['birth'],started_unix=record['started_unix'],devices=[0,1],
        output=str(run/'textcraft-dt'),log=str(run/'driver.stdout.log'),
        checkpoints=record['options']['trainer.default_local_dir'],budget=record['budget'],
        version=record['version'],status=record['status'],source_receipt=str(run/'source.json'),
        source_bindings=record['source_bindings'],local_patch_commit=commit,save_freq=5,
        fresh_base_model=True,resume_checkpoint=None,variance_standardization=record['variance_standardization'],
        PPO_reduction=record['PPO_reduction'],prior_driver=record['prior_job'],rootcause_repair_verified=False)
    assert [x for x in manifest['jobs'] if x is not j]==unaffected
    path.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(receipt=str(job_path),report=identity)))
