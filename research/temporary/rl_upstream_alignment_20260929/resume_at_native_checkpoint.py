"""One-time deployment at an original completed checkpoint, without rerunning a batch.

Waits only for the recorded job's native completion marker. It stops that
identified process tree, then calls the existing submission helper. VERL owns
checkpoint creation, dataloader state, loading and all training behavior.
--stop-only preserves the latest completed native checkpoint and does not
submit another job. It cannot be combined with --reuse-loaded-checkpoint.
--stop-only --stop-now stops without waiting for, creating or restoring a
checkpoint. An existing completion marker is recorded without validation.
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
    parser.add_argument('--run-dir',help='Required for the default stop-and-resume mode.')
    parser.add_argument('--receipt',required=True)
    parser.add_argument('--stop-only',action='store_true',
                        help='Stop only the current prepared task after its latest native checkpoint; do not resubmit.')
    parser.add_argument('--stop-now',action='store_true',
                        help='Only with --stop-only: stop now without waiting for, creating or restoring a checkpoint.')
    parser.add_argument('--reuse-loaded-checkpoint',action='store_true',
                        help='Use the originally loaded complete checkpoint only while its next training rollout is still unfinished.')
    args=parser.parse_args()
    if args.stop_now and not args.stop_only:
        parser.error('--stop-now requires --stop-only; no checkpoint is restored or job submitted')
    if args.stop_only and args.reuse_loaded_checkpoint:
        parser.error('--stop-only cannot use --reuse-loaded-checkpoint; preserve the latest native checkpoint instead')
    if not args.stop_only and args.run_dir is None:
        parser.error('--run-dir is required unless --stop-only is selected')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,hashlib,json,psutil,signal,subprocess,time
root=Path('@ROOT@');read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
task=@TASK@;prepared_path=Path(@PREPARED@);prepared=read(prepared_path)
active=read(root/'active-training.json');old=next(j for j in active['jobs'] if j['task']==task)
if @STOP_ONLY@:
    assert old['entry']==prepared['entry'] and old['verl_root']==prepared['verl_root']
    assert old['dt_root']==prepared['dt_root']
else:
    assert old['pid']==prepared['prior_driver_pid']
    assert old['entry']==prepared['prior_entry'] and old['verl_root']==prepared['prior_verl_root']
    assert old['checkpoints']==prepared['future_checkpoint_root']
parent=psutil.Process(old['pid'])
assert abs(parent.create_time()-old['observed_process_created_unix'])<.02
expected_owner=prepared.get('owner_sha256',prepared.get('owner_head_sha256'))
for name,h in prepared['entry_sha256'].items():assert sha(Path(prepared['entry'])/name)==h,name
for name,h in expected_owner.items():assert sha(Path(prepared['verl_root'])/name)==h,name
for name,h in prepared['dt_source_sha256'].items():assert sha(Path(prepared['dt_root'])/name)==h,name
current_source=None
if @STOP_ONLY@:
    source_path=Path(old['source_receipt'])
    source=read(source_path)
    assert source['dt_root']==old['dt_root'] and source['verl_root']==old['verl_root']
    assert source['entry_sha256']==prepared['entry_sha256']
    for name,h in source['verl_sha256'].items():assert sha(Path(old['verl_root'])/name)==h,name
    for name,h in source.get('owner_head_sha256',{}).items():
        assert expected_owner[name]==h and sha(Path(old['verl_root'])/name)==h,name
    current_source=dict(path=str(source_path),sha256=sha(source_path),
        entry=old['entry'],verl_root=old['verl_root'],dt_root=old['dt_root'],
        scope='Current job startup source receipt and frozen file paths/hashes; not a new live module introspection.')
receipt=Path(@RECEIPT@);receipt.mkdir(parents=True,exist_ok=True)
assert not (receipt/'completed-stop.json').exists(), 'Inspect the completed transition instead of repeating it'
identity=dict(task=task,prior_driver_pid=parent.pid,prior_created_unix=parent.create_time(),
    helper_repository_commit=@REVISION@,helper_source_sha256=@SCRIPT_SHA@,
    prepared_receipt=str(prepared_path),prepared_receipt_sha256=sha(prepared_path),
    minimum_checkpoint_step=@STEP@,started_unix=time.time())
if @STOP_ONLY@:
    identity.update(stop_only=True,current_source=current_source)
(receipt/'waiting.json').write_text(json.dumps(dict(identity,status=('stopping_without_checkpoint_wait' if @STOP_NOW@ else 'waiting_original_checkpoint'),
    observer_pid=psutil.Process().pid,observer_created_unix=psutil.Process().create_time()),indent=2)+'\n')
phase=None
if @STOP_NOW@:
    assert parent.is_running() and parent.status()!=psutil.STATUS_ZOMBIE, 'Original job exited before stopping'
    current=next(j for j in read(root/'active-training.json')['jobs'] if j['task']==task)
    assert current['pid']==parent.pid and current['observed_process_created_unix']==old['observed_process_created_unix']
    marker=Path(old['checkpoints'])/'latest_checkpointed_iteration.txt'
    try:marker_bytes=marker.read_bytes()
    except FileNotFoundError:marker_bytes=None
    step=None
    if marker_bytes is not None:
        try:step=int(marker_bytes.decode().strip())
        except (UnicodeDecodeError,ValueError):pass
    checkpoint=None if step is None else marker.parent/f'global_step_{step}'
    record=dict(identity,stop_now=True,checkpoint=None if checkpoint is None else str(checkpoint),
        marker_step=step,marker_path=str(marker),
        marker_sha256=None if marker_bytes is None else hashlib.sha256(marker_bytes).hexdigest(),
        marker_text=None if marker_bytes is None else marker_bytes.decode(errors='replace'),
        reused_loaded_checkpoint=False,unfinished_rollout_phase=None,checkpoint_files={},
        checkpoint_observed_unix=time.time(),
        checkpoint_observation_scope='Existing marker only, if present; no completion validation, waiting, checkpoint creation or restore. No minimum-step condition is applied in stop-now mode.')
elif @REUSE_LOADED@:
    loaded=Path(old['resume_from'])
    marker=loaded.parent/'latest_checkpointed_iteration.txt'
    step=int(marker.read_text().strip())
    assert step==old['resume_completed_step'] and step>=@STEP@
    assert loaded==marker.parent/f'global_step_{step}'
    runner=next(p for p in parent.children(recursive=True) if 'TaskRunner' in p.name())
    trainer=Path(old['verl_root'])/'verl/trainer/ppo/ray_trainer.py'
    tree=ast.parse(trainer.read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
              and ast.unparse(n.func)=='self.traj_collector.multi_turn_loop'
              and any(k.arg=='is_train' and isinstance(k.value,ast.Constant)
                      and k.value.value is True for k in n.keywords))
    observed=subprocess.run(['/opt/conda/bin/py-spy','dump','--nonblocking','--json',
        '--full-filenames','-p',str(runner.pid)],capture_output=True,text=True,timeout=12)
    assert observed.returncode==0,observed.stderr
    stack=json.loads(observed.stdout)
    frames=[f for t in stack if t['thread_name']=='MainThread' for f in t['frames']]
    assert any(f['name']=='fit' and f['filename']==str(trainer) and f['line']==call.lineno
               for f in frames), 'New rollout already advanced; wait for the next native checkpoint instead'
    assert any(f['name']=='collect_native_trajectories' for f in frames)
    phase=dict(task_runner_pid=runner.pid,trainer=str(trainer),trainer_sha256=sha(trainer),
        original_training_rollout_call_line=call.lineno,observed_unix=time.time(),stack=stack,
        meaning='Original fit is still collecting its next training rollout before any new update; restore the originally loaded native state.')
else:
    marker=Path(old['checkpoints'])/'latest_checkpointed_iteration.txt'
    while True:
        assert parent.is_running() and parent.status()!=psutil.STATUS_ZOMBIE, 'Original job exited before the required checkpoint'
        current=next(j for j in read(root/'active-training.json')['jobs'] if j['task']==task)
        assert current['pid']==parent.pid and current['observed_process_created_unix']==old['observed_process_created_unix']
        try:step=int(marker.read_text().strip())
        except (FileNotFoundError,ValueError):step=-1
        if step>=@STEP@:break
        time.sleep(1)
if not @STOP_NOW@:
    checkpoint=marker.parent/f'global_step_{step}'
    files=[checkpoint/'data.pt']+[checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt'
        for rank in range(2) for kind in ('model','optim','extra_state')]
    assert all(p.is_file() and p.stat().st_size for p in files), 'Incomplete native checkpoint; job not stopped'
    record=dict(identity,checkpoint=str(checkpoint),marker_step=step,marker_sha256=sha(marker),
        reused_loaded_checkpoint=@REUSE_LOADED@,unfinished_rollout_phase=phase,
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
print(json.dumps(dict(task=task,checkpoint=None if checkpoint is None else str(checkpoint),marker_step=step,stopped_driver=parent.pid)),flush=True)
PY
'''.replace('@ENTRY@',ENTRY).replace('@ROOT@',ROOT).replace('@TASK@',repr(args.task))
       .replace('@PREPARED@',repr(args.prepared)).replace('@RECEIPT@',repr(args.receipt))
       .replace('@REVISION@',repr(revision)).replace('@SCRIPT_SHA@',repr(script_sha))
       .replace('@REUSE_LOADED@',repr(args.reuse_loaded_checkpoint))
       .replace('@STOP_ONLY@',repr(args.stop_only))
       .replace('@STOP_NOW@',repr(args.stop_now))
       .replace('@STEP@',str(args.minimum_step)))
    target=AUDIT/'checkpoint-boundary-20261002';target.mkdir(exist_ok=True)
    stop_file=target/(args.task.lower()+'-'+Path(args.receipt).name+'-completed-stop.json')
    assert not stop_file.exists(), 'Preserve each original transition receipt'
    subprocess.run(SCP+[f'{SSH[-1]}:{args.receipt}/completed-stop.json',str(stop_file)],check=True)
    if args.stop_only:
        return
    import json
    stopped=json.loads(stop_file.read_text())
    submit_args=[sys.executable,str(AUDIT/'submit_prepared_appworld_resume.py'),
        '--task',args.task,'--checkpoint',stopped['checkpoint'],
        '--prepared',args.prepared,'--run-dir',args.run_dir]
    if args.reuse_loaded_checkpoint:
        submit_args.extend(['--completed-stop-receipt',args.receipt+'/completed-stop.json'])
    subprocess.run(submit_args,cwd=REPO,check=True)


if __name__=='__main__':
    main()
