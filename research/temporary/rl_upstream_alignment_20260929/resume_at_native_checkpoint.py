"""One-time deployment at an original completed checkpoint, without rerunning a batch.

Waits only for the recorded job's native completion marker. It stops that
identified process tree, then calls the existing submission helper. VERL owns
checkpoint creation, dataloader state, loading and all training behavior.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['AppWorld','TextCraft'],required=True)
    parser.add_argument('--minimum-step',type=int,required=True)
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--receipt',required=True)
    args=parser.parse_args()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,psutil,signal,time
root=Path('@ROOT@');read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
task=@TASK@;prepared_path=Path(@PREPARED@);prepared=read(prepared_path)
active=read(root/'active-training.json');old=next(j for j in active['jobs'] if j['task']==task)
assert old['pid']==prepared['prior_driver_pid']
assert old['entry']==prepared['prior_entry'] and old['verl_root']==prepared['prior_verl_root']
assert old['checkpoints']==prepared['future_checkpoint_root']
parent=psutil.Process(old['pid'])
assert abs(parent.create_time()-old['observed_process_created_unix'])<.02
expected_owner=prepared.get('owner_sha256',prepared.get('owner_head_sha256'))
for name,h in prepared['entry_sha256'].items():assert sha(Path(prepared['entry'])/name)==h,name
for name,h in expected_owner.items():assert sha(Path(prepared['verl_root'])/name)==h,name
for name,h in prepared['dt_source_sha256'].items():assert sha(Path(prepared['dt_root'])/name)==h,name
receipt=Path(@RECEIPT@);receipt.mkdir(parents=True,exist_ok=True)
assert not (receipt/'completed-stop.json').exists(), 'Inspect the completed transition instead of repeating it'
identity=dict(task=task,prior_driver_pid=parent.pid,prior_created_unix=parent.create_time(),
    helper_repository_commit=@REVISION@,helper_source_sha256=@SCRIPT_SHA@,
    prepared_receipt=str(prepared_path),prepared_receipt_sha256=sha(prepared_path),
    minimum_checkpoint_step=@STEP@,started_unix=time.time())
(receipt/'waiting.json').write_text(json.dumps(dict(identity,status='waiting_original_checkpoint',
    observer_pid=psutil.Process().pid,observer_created_unix=psutil.Process().create_time()),indent=2)+'\n')
marker=Path(old['checkpoints'])/'latest_checkpointed_iteration.txt'
while True:
    assert parent.is_running() and parent.status()!=psutil.STATUS_ZOMBIE, 'Original job exited before the required checkpoint'
    current=next(j for j in read(root/'active-training.json')['jobs'] if j['task']==task)
    assert current['pid']==parent.pid and current['observed_process_created_unix']==old['observed_process_created_unix']
    try:step=int(marker.read_text().strip())
    except (FileNotFoundError,ValueError):step=-1
    if step>=@STEP@:break
    time.sleep(1)
checkpoint=marker.parent/f'global_step_{step}'
files=[checkpoint/'data.pt']+[checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt'
    for rank in range(2) for kind in ('model','optim','extra_state')]
assert all(p.is_file() and p.stat().st_size for p in files), 'Incomplete native checkpoint; job not stopped'
record=dict(identity,checkpoint=str(checkpoint),marker_step=step,marker_sha256=sha(marker),
    checkpoint_files={str(p.relative_to(checkpoint)):p.stat().st_size for p in files},
    checkpoint_observed_unix=time.time())
processes=[parent]+parent.children(recursive=True)
record['processes']=[dict(pid=p.pid,created_unix=p.create_time(),name=p.name()) for p in processes]
# Another authorized transition may have happened while this observer waited.
# Observe the authoritative identities at this boundary, not the old ones.
active_at_boundary=read(root/'active-training.json')
other={j['task']:(j['pid'],j['observed_process_created_unix']) for j in active_at_boundary['jobs'] if j['task']!=task}
(receipt/'stopping.json').write_text(json.dumps(record,indent=2)+'\n')
for p in reversed(processes):
    try:p.send_signal(signal.SIGTERM)
    except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(processes,timeout=8)
for p in alive:
    try:p.kill()
    except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(alive,timeout=4)
remaining=[]
for p in alive:
    try:
        if p.status()!=psutil.STATUS_ZOMBIE:remaining.append(p.pid)
    except psutil.NoSuchProcess:pass
record.update(finished_unix=time.time(),remaining_non_zombie=remaining)
record['other_jobs_unchanged']={}
for name,(pid,created) in other.items():
    try:record['other_jobs_unchanged'][name]=abs(psutil.Process(pid).create_time()-created)<.02
    except psutil.NoSuchProcess:record['other_jobs_unchanged'][name]=False
record['other_jobs_observation_scope']='Identity observations across this stop only; another task exiting or restarting does not prevent this task from resuming.'
(receipt/'completed-stop.json').write_text(json.dumps(record,indent=2)+'\n')
assert not remaining, 'Recorded job has remaining processes; no new job submitted'
print(json.dumps(dict(task=task,checkpoint=str(checkpoint),marker_step=step,stopped_driver=parent.pid)),flush=True)
PY
'''.replace('@ENTRY@',ENTRY).replace('@ROOT@',ROOT).replace('@TASK@',repr(args.task))
       .replace('@PREPARED@',repr(args.prepared)).replace('@RECEIPT@',repr(args.receipt))
       .replace('@REVISION@',repr(revision)).replace('@SCRIPT_SHA@',repr(script_sha))
       .replace('@STEP@',str(args.minimum_step)))
    target=AUDIT/'checkpoint-boundary-20261002';target.mkdir(exist_ok=True)
    stop_file=target/(args.task.lower()+'-completed-stop.json')
    subprocess.run(SCP+[f'{SSH[-1]}:{args.receipt}/completed-stop.json',str(stop_file)],check=True)
    import json
    stopped=json.loads(stop_file.read_text())
    subprocess.run([sys.executable,str(AUDIT/'submit_prepared_appworld_resume.py'),
        '--task',args.task,'--checkpoint',stopped['checkpoint'],
        '--prepared',args.prepared,'--run-dir',args.run_dir],cwd=REPO,check=True)


if __name__=='__main__':
    main()
