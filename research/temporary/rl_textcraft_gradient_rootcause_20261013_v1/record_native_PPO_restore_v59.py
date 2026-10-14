"""Record actual native full PPO restore and preserve previous process identities."""
import hashlib
import json
from pathlib import Path
import psutil
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
job_path=OUT/'native-PPO-resume-after-v47-v55-job.json'
job=json.loads(job_path.read_bytes());RUN=Path(job['run'])
binding_path=RUN/'runtime-overrides/native-PPO-bindings-v56.json'
binding=json.loads(binding_path.read_bytes())
checkpoint=Path(job['resume_from_path'])
assert psutil.Process(job['pid']).create_time()==job['birth']
states=[]
for worker in binding['workers']:
    for role in worker['roles']:
        if role['role'] not in ('actor_rollout','critic'):continue
        assert role['scheduler_last_epoch']==15 and role['optimizer_state_steps']==[60.]
        if role['role']=='actor_rollout':
            assert role['rollout_rng_restored_variable_present'] and role['rollout_rng_owner_is_native_manager']
            assert role['checkpoint_manager']['SHA256']=='f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01'
        states.append(dict(pid=worker['pid'],birth=worker['birth'],role=role['role'],rank=role['rank'],
            scheduler=15,optimizer_step=60,optimizer_entries=role['optimizer_state_entries']))
assert len(states)==4
prepared=json.loads((OUT/'native-PPO-resume-after-v47-v55-prepared.json').read_bytes())
assert set(prepared['configuration_changes'])=={'trainer.resume_mode','trainer.resume_from_path'}
report=dict(unix=time.time(),status='native_PPO_full_checkpoint15_restore_verified',job=job,
    actual_worker_binding=str(binding_path),actual_worker_binding_SHA256=hashlib.sha256(binding_path.read_bytes()).hexdigest(),
    actual_role_states=states,completed_checkpoint=str(checkpoint),native_data_state=str(checkpoint/'data.pt'),
    native_data_SHA256=hashlib.sha256((checkpoint/'data.pt').read_bytes()).hexdigest(),
    native_rng_storage='actor/critic extra_state rng(cpu,numpy,random,cuda); actor extra rollout_gen_random_states -> original rollout manager',
    native_RNG_owner_verified=True,checkpoint_owner_SHA256='f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01',
    restore_helper_path=str(OUT/'resume_native_PPO_after_diagnostic_v55.py'),
    restore_helper_SHA256=hashlib.sha256((OUT/'resume_native_PPO_after_diagnostic_v55.py').read_bytes()).hexdigest(),
    resume_configuration_changes=prepared['configuration_changes'],training_hyperparameters_changed=False,
    diagnostic_v55_RNG_field_correction='v55 incorrectly inspected the worker attribute; the native variable belongs to rollout_sharding_manager. Actual owner pointer and presence are verified in v56. No training RNG defect was observed.',
    limits=['The current generation consumes restored rollout RNG normally. This is not a claim that current consumed RNG still equals checkpoint15.',
        'Native model/optimizer/scheduler/RNG/data loading and actual counters are verified; a restarted full iteration has not yet been reported here.'],
    rootcause_goal_status='active',rootcause_repair_verified=False,hourly_automation='PAUSED')
dest=RUN/'runtime-overrides/native-PPO-restore-verification-v59.json';assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
job.update(status=report['status'],actual_native_restore_receipt=str(dest),actual_native_restore_observed_unix=report['unix'])
for p in (job_path,OUT/'native-PPO-control-v9-job.json',RUN/'resume-v55-job.json',RUN/'job.json'):
    backup=OUT/(p.name+'.before-verified-restore-v59.json')
    # Distinguish equal basenames from separate receipt paths.
    if backup.exists():backup=OUT/(p.parent.name+'-'+p.name+'.before-verified-restore-v59.json')
    assert not backup.exists()
    if p.exists():backup.write_bytes(p.read_bytes())
    p.write_text(json.dumps(job,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
