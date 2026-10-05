"""Submit a frozen AppWorld or TextCraft release after its old job stops.

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
    parser.add_argument('--task', choices=['AppWorld', 'TextCraft'], default='AppWorld')
    parser.add_argument('--checkpoint', required=True,
                        help='Original completed global_step directory.')
    parser.add_argument('--prepared', required=True,
                        help='Remote immutable preparation receipt for this exact prior job.')
    parser.add_argument('--run-dir', required=True,
                        help='Remote output directory for this submission; must not already exist.')
    parser.add_argument('--completed-stop-receipt',
                        help='Original loaded checkpoint reuse after a recorded stop during the unfinished next rollout.')
    parser.add_argument('--initialization-failure-receipt',
                        help='Recorded terminal init_workers failure before fit; reuse its original complete resume checkpoint.')
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
task=@TASK@
active=read(root/'active-training.json');old=next(j for j in active['jobs'] if j['task']==task)
prepared_path=Path(@PREPARED@)
prepared=read(prepared_path)
devices={'AppWorld':[2,3],'TextCraft':[4,5]}[task]
assert old['pid']==prepared['prior_driver_pid'] and old['devices']==devices
assert old['entry']==prepared['prior_entry'] and old['verl_root']==prepared['prior_verl_root']
assert old['checkpoints']==prepared['future_checkpoint_root']
if psutil.pid_exists(old['pid']):
    p=psutil.Process(old['pid'])
    assert p.create_time()==old['observed_process_created_unix'], 'PID identity changed'
    assert p.status()==psutil.STATUS_ZOMBIE, 'Original job still running; no stop or mutation performed'

checkpoint=Path(@CHECKPOINT@).resolve()
stop_receipt=@COMPLETED_STOP@
init_failure_receipt=@INIT_FAILURE@
stop_record=None
init_failure=None
assert not (stop_receipt and init_failure_receipt)
if init_failure_receipt:
    init_failure=read(Path(init_failure_receipt))
    assert init_failure['failed_driver_pid']==old['pid']
    assert init_failure['failed_driver_created_unix']==old['observed_process_created_unix']
    assert init_failure['driver_terminal'] and not psutil.pid_exists(old['pid'])
    assert init_failure['failed_source_sha256']==sha(Path(old['source_receipt']))
    assert init_failure['log_sha256']==sha(Path(old['log']))
    assert checkpoint==Path(old['resume_from']).resolve()==Path(init_failure['resume_checkpoint']).resolve()
    marker=checkpoint.parent/'latest_checkpointed_iteration.txt'
elif stop_receipt:
    stop_record=read(Path(stop_receipt))
    assert stop_record['reused_loaded_checkpoint'] and stop_record['unfinished_rollout_phase']
    assert stop_record['prior_driver_pid']==old['pid'] and stop_record['prior_created_unix']==old['observed_process_created_unix']
    assert not stop_record['remaining_non_zombie']
    assert checkpoint==Path(old['resume_from']).resolve()==Path(stop_record['checkpoint']).resolve()
    marker=checkpoint.parent/'latest_checkpointed_iteration.txt'
else:
    marker=Path(old['checkpoints'])/'latest_checkpointed_iteration.txt'
step=int(marker.read_text().strip())
assert checkpoint==(marker.parent/f'global_step_{step}').resolve()
assert (checkpoint/'data.pt').is_file()
for rank in range(2):
    for kind in ('model','optim','extra_state'):
        assert (checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').is_file()
entry=Path(prepared['entry']);verl=Path(prepared['verl_root'])
for name,expected in prepared['entry_sha256'].items():assert sha(entry/name)==expected,name
owner_files=prepared.get('owner_sha256',prepared.get('owner_head_sha256'))
for name,expected in owner_files.items():assert sha(verl/name)==expected,name
for name,expected in prepared['dt_source_sha256'].items():assert sha(Path(prepared['dt_root'])/name)==expected,name
state=subprocess.check_output(['mx-smi'],text=True)
for device in old['devices']:
    assert not re.search(r'\|\s*'+str(device)+r'\s+\d+\s+\S',state),state

base=Path(@RUN_DIR@)
output=base/({'AppWorld':'appworld-dt','TextCraft':'textcraft-dt'}[task])
assert not base.exists(), 'Keep submissions immutable; inspect any existing attempt'
output.mkdir(parents=True)
for name in ('active-training.json','active-source.json'):
    (base/('prior-'+name)).write_bytes((root/name).read_bytes())
prior_source=read(Path(old.get('source_receipt',str(Path(old['output'])/'source.json'))))
source=dict(unix=time.time(),dt_root=prepared['dt_root'],verl_root=str(verl),
    lora_rank=8,lora_alpha=16,entry_sha256=prepared['entry_sha256'],
    verl_sha256={name:sha(verl/name) for name in prior_source['verl_sha256']},
    owner_head_sha256=owner_files,
    prepared_receipt=str(prepared_path),prepared_receipt_sha256=sha(prepared_path),
    submission_repository_commit=@REVISION@,submission_script_sha256=@SCRIPT_SHA@,
    prior_driver_pid=old['pid'],resume_from=str(checkpoint),
    completed_checkpoint_marker=dict(path=str(marker),value=step,sha256=sha(marker)),
    actor_fix_commit='dc4e4d7',dt_dispatch_commit=prepared.get('dt_dispatch_commit','4c0cbdd'),
    actor_padding_sha256=owner_files['verl/workers/actor/dp_actor.py'],
    padding_comparison_receipt=prepared['padding_comparison_receipt'],
    padding_comparison_receipt_sha256=prepared['padding_comparison_receipt_sha256'])
if stop_receipt:
    source['unfinished_rollout_restart']=dict(receipt=stop_receipt,sha256=sha(Path(stop_receipt)),
        loaded_original_checkpoint=str(checkpoint),completed_step=step,
        scope='No newly completed iteration is discarded; original model/optimizer/RNG/reader loader used again, with unchanged workload.')
if init_failure_receipt:
    source['initialization_retry']=dict(receipt=init_failure_receipt,
        sha256=sha(Path(init_failure_receipt)),original_checkpoint=str(checkpoint),
        scope='Recorded init_workers failure occurred before fit/checkpoint loading/updates. Original VERL loader reuses the same complete checkpoint; no new training state discarded.')
for name in ('request_dispatch_commit','request_dispatch_receipt','request_dispatch_receipt_sha256',
             'rollout_scope_commit','rollout_scope_comparison','rollout_scope_comparison_sha256',
             'completion_transport_code_commit','completion_transport_receipt',
             'completion_transport_receipt_sha256','completion_transport_sources',
             'completed_overlay_receipt','completed_overlay_sha256',
             'prior_source_receipt','prior_source_sha256'):
    if name in prepared:source[name]=prepared[name]
env=os.environ.copy()
env.update(VERL_ROOT=str(verl),DT_ROOT=prepared['dt_root'],
    DT_ENTRY_ROOT=str(entry),CUDA_VISIBLE_DEVICES=','.join(map(str,devices)))
env.pop('MACA_VISIBLE_DEVICES',None)
extra_path=[]
if task=='AppWorld':
    source.update(loop_root=prepared['loop_root'],author_sha256=prior_source['author_sha256'])
    env.update(LOOP_ROOT=prepared['loop_root'],
        APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'))
    extra_path.append(prepared['loop_root'])
    launcher=entry/'launch_appworld_native.py'
    argv=[env['VENV_PYTHON'],str(launcher),'--output',str(output),'--resume-from',str(checkpoint)]
else:
    prior_launch=read(Path(old['output'])/'launch.json')
    data=old['argv'][old['argv'].index('--data')+1]
    agentgym=prior_launch['options']['+env.textcraft']['owner_root']
    env['AGENTGYM_RL_ROOT']=agentgym
    source.update(agentgym_root=agentgym,prior_source_receipt=str(Path(old['output'])/'source.json'),
        prior_source_sha256=sha(Path(old['output'])/'source.json'))
    launcher=entry/'launch_textcraft_native.py'
    argv=[env['VENV_PYTHON'],str(launcher),'--data',data,'--output',str(output),
          '--resume-from',str(checkpoint)]
# The provisioned environment expanded its DT paths when it was sourced.
# Repoint those exact owner paths when a frozen DT release is selected.
# Preserve the original ordering and all unrelated environment dependencies.
provisioned_dt_root=Path(os.environ['DT_ROOT'])
dt_paths={str(provisioned_dt_root):prepared['dt_root'],
          str(provisioned_dt_root/'experiments/rl'):str(Path(prepared['dt_root'])/'experiments/rl')}
inherited_paths=[dt_paths.get(path,path) for path in env['PYTHONPATH'].split(':')]
env['PYTHONPATH']=':'.join([str(entry),str(verl),*extra_path,*inherited_paths])
source['pythonpath']=env['PYTHONPATH']
source['resume_launcher']=dict(path=str(launcher),sha256=sha(launcher))
(output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
with (output/'train.log').open('wb') as log:
    proc=subprocess.Popen(argv,env=env,cwd=str(output),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(task=task,method='dt',pid=proc.pid,devices=devices,started_unix=time.time(),
    observed_process_created_unix=psutil.Process(proc.pid).create_time(),argv=argv,
    entry=str(entry),verl_root=str(verl),dt_root=prepared['dt_root'],
    output=str(output),log=str(output/'train.log'),checkpoints=str(output/'checkpoints'),
    source_receipt=str(output/'source.json'),status='formal_resume_submitted_not_yet_verified',
    lora_rank=8,lora_alpha=16,actor_microbatch=4,log_prob_micro_batch_size_per_gpu=4,
    budget=old['budget'],resume_from=str(checkpoint),resume_completed_step=step,prior_driver_pid=old['pid'])
(output/'job.json').write_text(json.dumps(job,indent=2)+'\n')
manifest_path=base/'formal-training.json'
manifest=dict(active,manifest=str(manifest_path),jobs=[job if j['task']==task else j for j in active['jobs']])
manifest.setdefault('retired_jobs',[]).append(dict(old,
    status=('replaced_after_initialization_failure' if init_failure_receipt else
            'replaced_during_unfinished_rollout_from_loaded_checkpoint' if stop_receipt else 'replaced_after_completed_checkpoint'),
    replacement_pid=proc.pid))
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
(root/'active-training.json').write_text(json.dumps(manifest,indent=2)+'\n')
sources=read(root/'active-source.json');sources.update(unix=time.time(),manifest=str(manifest_path))
sources['jobs']=[dict(task=job['task'],pid=job['pid'],entry=job['entry'],verl_root=job['verl_root'],
    source_receipt=job['source_receipt'],runtime_override=None,actor_microbatch=4,lora_rank=8,lora_alpha=16)
    if item['task']==task else item for item in sources['jobs']]
(root/'active-source.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps(job,indent=2),flush=True)
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY)
       .replace('@CHECKPOINT@', repr(args.checkpoint)).replace('@REVISION@', repr(revision))
       .replace('@SCRIPT_SHA@', repr(script_sha)).replace('@PREPARED@',repr(args.prepared))
       .replace('@RUN_DIR@',repr(args.run_dir)).replace('@TASK@',repr(args.task))
       .replace('@COMPLETED_STOP@',repr(args.completed_stop_receipt))
       .replace('@INIT_FAILURE@',repr(args.initialization_failure_receipt)))


if __name__ == '__main__':
    main()
