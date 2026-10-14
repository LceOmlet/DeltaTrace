"""User-authorized temporary PPO stop; preserve native checkpoint and other jobs."""
import hashlib
import json
import os
from pathlib import Path
import psutil
import signal
import subprocess
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
dest=OUT/'PPO-temporary-stop-for-v47-v54.json'
assert not dest.exists(), 'Inspect the existing stop receipt; do not repeat signals'
job=json.loads((OUT/'native-PPO-control-v9-job.json').read_bytes())
run=Path(job['run']);launch=json.loads((run/'launch.json').read_bytes())
directory=Path(launch['options']['trainer.default_local_dir'])
marker=directory/'latest_checkpointed_iteration.txt';step=int(marker.read_text().strip())
assert step>=15
checkpoint=directory/f'global_step_{step}'
for role in ('actor','critic'):
    for rank in (0,1):
        for kind in ('model','optim','extra_state'):
            path=checkpoint/role/f'{kind}_world_size_2_rank_{rank}.pt'
            assert path.is_file() and path.stat().st_size>0, path
assert (checkpoint/'data.pt').is_file()
protected=[];protected_pids=set()
for pid,birth in [(1419849,1791977138.31),(3174448,1791793629.73),(2614045,1791949034.31)]:
    p=psutil.Process(pid);assert p.create_time()==birth and p.status()!=psutil.STATUS_ZOMBIE
    protected.append(dict(pid=pid,birth=birth));protected_pids.add(pid)
    protected_pids.update(c.pid for c in p.children(recursive=True))
p=psutil.Process(job['pid']);assert p.create_time()==job['birth']
assert os.getsid(p.pid)==p.pid==os.getpgid(p.pid)
members={p.pid:p}
members.update({c.pid:c for c in p.children(recursive=True)})
for q in psutil.process_iter():
    try:
        if os.getpgid(q.pid)==p.pid:members[q.pid]=q
    except (psutil.NoSuchProcess,ProcessLookupError):pass
assert not protected_pids.intersection(members)
identities=[dict(pid=q.pid,birth=q.create_time(),name=q.name()) for q in members.values()]
report=dict(started_unix=time.time(),instruction='User explicitly chose temporary PPO stop and full native restore after the bounded diagnostic.',
    job=job,checkpoint=str(checkpoint),completed_iteration=step,marker=str(marker),
    data_SHA256=hashlib.sha256((checkpoint/'data.pt').read_bytes()).hexdigest(),
    protected_jobs=protected,members_before=identities,signals=[],new_checkpoint_created=False,
    checkpoint_files_deleted=False,status='validated_before_stop')
dest.write_text(json.dumps(report,indent=2)+'\n')
os.killpg(p.pid,signal.SIGTERM)
report['signals'].append(dict(signal='SIGTERM',process_group=p.pid,unix=time.time()))
_,alive=psutil.wait_procs(list(members.values()),timeout=12)
for q in alive:
    old=next(x for x in identities if x['pid']==q.pid)
    try:
        if q.create_time()==old['birth'] and q.status()!=psutil.STATUS_ZOMBIE:
            assert q.pid not in protected_pids
            q.kill();report['signals'].append(dict(signal='SIGKILL',pid=q.pid,birth=old['birth']))
    except psutil.NoSuchProcess:pass
psutil.wait_procs(alive,timeout=5)
survivors=[]
for old in identities:
    try:
        q=psutil.Process(old['pid'])
        if q.create_time()==old['birth'] and q.status()!=psutil.STATUS_ZOMBIE:survivors.append(old)
    except psutil.NoSuchProcess:pass
report['survivors']=survivors;assert not survivors
for old in protected:
    q=psutil.Process(old['pid']);assert q.create_time()==old['birth'] and q.status()!=psutil.STATUS_ZOMBIE
assert marker.read_text().strip()==str(step)
report.update(finished_unix=time.time(),status='temporarily_stopped_native_full_restore_required',
              physical_mx_smi=subprocess.check_output(['mx-smi'],text=True,timeout=30))
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
