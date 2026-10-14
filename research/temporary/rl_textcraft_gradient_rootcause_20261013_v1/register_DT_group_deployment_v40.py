"""Select the actually submitted DT version without touching other jobs."""
import copy
import hashlib
import json
from pathlib import Path
import psutil
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
job=json.loads((OUT/'DT-group-multiplier-v40-job.json').read_bytes())
run=Path(job['run']);launch=json.loads((run/'launch.json').read_bytes())
old=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
original=json.loads((old/'source.json').read_bytes())
state_path=run/'runtime-overrides/initial-state-and-existing-passive-records-v40.json'
state=json.loads(state_path.read_bytes())
assert state['complete'] and state['initial_optimizer0_scheduler0_verified']
assert psutil.Process(job['pid']).create_time()==job['birth']
source=json.loads((run/'source.json').read_bytes())
assert not (run/'manifest-registration-v40.json').exists(), 'Already registered; read existing receipt'
bindings={}
old_verl=original['verl_root'];new_verl=launch['effective_environment']['VERL_ROOT']
for path,expected in original.get('source_bindings',{}).items():
    current=path.replace(old_verl,new_verl)
    if Path(current).is_file():
        value=hashlib.sha256(Path(current).read_bytes()).hexdigest()
        # The only changed upstream file was already checked in preparation.
        if current != new_verl+'/verl/trainer/ppo/ray_trainer.py':
            assert value==expected, current
        bindings[current]=value
bindings.update(launch['source_bindings'])
for k in ('entry_sha256','verl_sha256','white_patch_commit','entry_changes','added_entry_sources'):
    if k in source:source['inherited_'+k]=source.pop(k)
source.update(source_bindings=bindings,entry=launch['effective_environment']['PYTHONPATH'].split(':')[0],
    pythonpath=launch['effective_environment']['PYTHONPATH'],prepared_only=False,
    method_patch_commit='ea2a470c',submission_repository_commit=job['local_patch_commit'],
    DT_whitening=False,multiplier_owner='fixed native VERL compute_advantage(GRPO) -> compute_grpo_outcome_advantage',
    actual_initial_worker_state_receipt=str(state_path),actual_initial_worker_state_verified=True,
    actual_initial_worker_states=state['workers'],passive_credit_recording_runtime_override=str(state_path),
    passive_recording_status='existing original recording mounted on actual new workers; first output pending',
    runtime_verification_status='initial bindings verified; first full iteration pending',
    actual_import_scope='Actual new worker imports and initial optimizer/scheduler read via original RPC.',
    deployment=dict(status='submitted_actual_initial_workers_verified',pid=job['pid'],birth=job['birth'],
                    rootcause_repair_verified=False))
(run/'source.json').write_text(json.dumps(source,indent=2)+'\n')
new=dict(task='TextCraft',method='dt',pid=job['pid'],birth=job['birth'],
    observed_process_created_unix=job['birth'],started_unix=job['started_unix'],devices=[0,1],
    argv=launch['argv'],entry=source['entry'],dt_root=launch['effective_environment']['DT_ROOT'],verl_root=new_verl,
    output=str(run/'textcraft-dt'),log=str(run/'driver.stdout.log'),checkpoints=launch['options']['trainer.default_local_dir'],
    source_receipt=str(run/'source.json'),status='submitted_actual_initial_workers_verified_first_full_iteration_pending',
    version=job['version'],lora_rank=8,lora_alpha=16,actor_microbatch=4,log_prob_micro_batch_size_per_gpu=4,
    budget=launch['budget'],fresh_base_model=True,resume_checkpoint=None,completed_iterations=0,save_freq=5,
    actual_initial_worker_state_receipt=str(state_path),actual_initial_worker_state_verified=True,
    actual_initial_worker_states=state['workers'],passive_credit_recording_runtime_override=str(state_path),
    passive_recording_status=source['passive_recording_status'],workers=state['workers'],
    local_patch_commit=job['local_patch_commit'],source_bindings=launch['source_bindings'],
    prior_driver=dict(pid=3172194,birth=1791793609.09,status='stopped_preserved_not_resumed'),
    rootcause_repair_verified=False)
snapshots=run/'deployment-receipts';snapshots.mkdir(exist_ok=True)
changed=[]
for name in ('formal-training.json','active-training.json','active-source.json'):
    path=ROOT/name;raw=path.read_bytes();manifest=json.loads(raw)
    rows=[i for i,j in enumerate(manifest['jobs']) if j.get('task')=='TextCraft' and j.get('method')=='dt']
    assert len(rows)==1 and manifest['jobs'][rows[0]]['pid']==3172194
    unaffected=[j for i,j in enumerate(manifest['jobs']) if i!=rows[0]]
    snapshot=snapshots/(name+'.before-v40.json');assert not snapshot.exists();snapshot.write_bytes(raw)
    manifest['jobs'][rows[0]]=copy.deepcopy(new);manifest['unix']=time.time()
    assert [j for i,j in enumerate(manifest['jobs']) if i!=rows[0]]==unaffected
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    changed.append(dict(path=str(path),snapshot=str(snapshot),new_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        non_DT_jobs_exactly_unchanged=True))
for pid,birth in [(3174448,1791793629.73),(2671393,1791949551.27)]:assert psutil.Process(pid).create_time()==birth
report=dict(unix=time.time(),job=job,manifest_updates=changed,actual_worker_receipt=str(state_path),
            source_binding_files=len(bindings),rootcause_repair_verified=False)
dest=run/'manifest-registration-v40.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
