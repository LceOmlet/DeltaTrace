"""Submit the frozen AppWorld candidate after its old job has stopped.

This script does not stop a job or implement checkpoint loading. The completed
checkpoint is passed to the prepared launcher and original VERL loader.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

from stage_environment_entry import remote, ROOT, ENTRY, REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True,
                        help='Original completed AppWorld global_step directory.')
    parser.add_argument('--prepared', default=ROOT+'/receipts/owner-b8-dispatch-20260930/appworld-balanced-padding-resume/prepared.json',
                        help='Remote immutable preparation receipt for this exact prior job.')
    parser.add_argument('--run-dir', default=ROOT+'/runs/appworld-balanced-padding-resume-20261001',
                        help='Remote output directory for this submission; must not already exist.')
    args = parser.parse_args()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
root=Path('@ROOT@');read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
active=read(root/'active-training.json');old=next(j for j in active['jobs'] if j['task']=='AppWorld')
prepared_path=Path(@PREPARED@)
prepared=read(prepared_path)
assert old['pid']==prepared['prior_driver_pid'] and old['devices']==[2,3]
assert old['entry']==prepared['prior_entry'] and old['verl_root']==prepared['prior_verl_root']
assert old['checkpoints']==prepared['future_checkpoint_root']
if psutil.pid_exists(old['pid']):
    p=psutil.Process(old['pid'])
    assert p.create_time()==old['observed_process_created_unix'], 'PID identity changed'
    assert p.status()==psutil.STATUS_ZOMBIE, 'Original job still running; no stop or mutation performed'

checkpoint=Path(@CHECKPOINT@).resolve()
marker=Path(old['checkpoints'])/'latest_checkpointed_iteration.txt'
step=int(marker.read_text().strip())
assert checkpoint==(Path(old['checkpoints'])/f'global_step_{step}').resolve()
assert (checkpoint/'data.pt').is_file()
for rank in range(2):
    for kind in ('model','optim','extra_state'):
        assert (checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').is_file()
entry=Path(prepared['entry']);verl=Path(prepared['verl_root'])
for name,expected in prepared['entry_sha256'].items():assert sha(entry/name)==expected,name
for name,expected in prepared['owner_head_sha256'].items():assert sha(verl/name)==expected,name
for name,expected in prepared['dt_source_sha256'].items():assert sha(Path(prepared['dt_root'])/name)==expected,name
state=subprocess.check_output(['mx-smi'],text=True)
for device in old['devices']:
    assert not re.search(r'\|\s*'+str(device)+r'\s+\d+\s+\S',state),state

base=Path(@RUN_DIR@)
output=base/'appworld-dt'
assert not base.exists(), 'Keep submissions immutable; inspect any existing attempt'
output.mkdir(parents=True)
for name in ('active-training.json','active-source.json'):
    (base/('prior-'+name)).write_bytes((root/name).read_bytes())
prior_source=read(Path(old.get('source_receipt',str(Path(old['output'])/'source.json'))))
source=dict(unix=time.time(),dt_root=prepared['dt_root'],verl_root=str(verl),loop_root=prepared['loop_root'],
    lora_rank=8,lora_alpha=16,entry_sha256=prepared['entry_sha256'],
    verl_sha256={name:sha(verl/name) for name in prior_source['verl_sha256']},
    owner_head_sha256=prepared['owner_head_sha256'],author_sha256=prior_source['author_sha256'],
    prepared_receipt=str(prepared_path),prepared_receipt_sha256=sha(prepared_path),
    submission_repository_commit=@REVISION@,submission_script_sha256=@SCRIPT_SHA@,
    prior_driver_pid=old['pid'],resume_from=str(checkpoint),
    completed_checkpoint_marker=dict(path=str(marker),value=step,sha256=sha(marker)),
    actor_fix_commit='dc4e4d7',dt_dispatch_commit='4c0cbdd',resume_entry_commit='2036246',
    actor_padding_sha256=prepared['owner_head_sha256']['verl/workers/actor/dp_actor.py'],
    padding_comparison_receipt=prepared['padding_comparison_receipt'],
    padding_comparison_receipt_sha256=prepared['padding_comparison_receipt_sha256'])
for name in ('request_dispatch_commit','request_dispatch_receipt','request_dispatch_receipt_sha256'):
    if name in prepared:source[name]=prepared[name]
(output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
env=os.environ.copy()
env.update(VERL_ROOT=str(verl),LOOP_ROOT=prepared['loop_root'],DT_ROOT=prepared['dt_root'],
    DT_ENTRY_ROOT=str(entry),CUDA_VISIBLE_DEVICES='2,3',
    APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'))
env.pop('MACA_VISIBLE_DEVICES',None)
env['PYTHONPATH']=':'.join([str(entry),str(verl),prepared['loop_root'],env['PYTHONPATH']])
argv=[env['VENV_PYTHON'],str(entry/'launch_appworld_native.py'),'--output',str(output),
      '--resume-from',str(checkpoint)]
with (output/'train.log').open('wb') as log:
    proc=subprocess.Popen(argv,env=env,cwd=str(output),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(task='AppWorld',method='dt',pid=proc.pid,devices=[2,3],started_unix=time.time(),
    observed_process_created_unix=psutil.Process(proc.pid).create_time(),argv=argv,
    entry=str(entry),verl_root=str(verl),loop_root=prepared['loop_root'],dt_root=prepared['dt_root'],
    output=str(output),log=str(output/'train.log'),checkpoints=str(output/'checkpoints'),
    source_receipt=str(output/'source.json'),status='formal_resume_submitted_not_yet_verified',
    lora_rank=8,lora_alpha=16,actor_microbatch=4,log_prob_micro_batch_size_per_gpu=4,
    budget=old['budget'],resume_from=str(checkpoint),resume_completed_step=step,prior_driver_pid=old['pid'])
(output/'job.json').write_text(json.dumps(job,indent=2)+'\n')
manifest_path=base/'formal-training.json'
manifest=dict(active,manifest=str(manifest_path),jobs=[job if j['task']=='AppWorld' else j for j in active['jobs']])
manifest.setdefault('retired_jobs',[]).append(dict(old,status='replaced_after_completed_checkpoint',replacement_pid=proc.pid))
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
(root/'active-training.json').write_text(json.dumps(manifest,indent=2)+'\n')
sources=read(root/'active-source.json');sources.update(unix=time.time(),manifest=str(manifest_path))
sources['jobs']=[dict(task=job['task'],pid=job['pid'],entry=job['entry'],verl_root=job['verl_root'],
    source_receipt=job['source_receipt'],runtime_override=None,actor_microbatch=4,lora_rank=8,lora_alpha=16)
    if item['task']=='AppWorld' else item for item in sources['jobs']]
(root/'active-source.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps(job,indent=2),flush=True)
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY)
       .replace('@CHECKPOINT@', repr(args.checkpoint)).replace('@REVISION@', repr(revision))
       .replace('@SCRIPT_SHA@', repr(script_sha)).replace('@PREPARED@',repr(args.prepared))
       .replace('@RUN_DIR@',repr(args.run_dir)))


if __name__ == '__main__':
    main()
