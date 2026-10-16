"""Reuse the existing native PPO resume launcher after the measured global OOM."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-formal-training-20261017-v1'
CANONICAL = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json'
OWNER = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/resume_native_PPO_after_diagnostic_v55.py'
VERSION = 'textcraft-PPO-global-OOM-recovery-20261017-v1'
INPUT = AUDIT / 'TextCraft-PPO-native-OOM-recovery-input.json'
PREFIX = 'TextCraft-PPO-native-global-OOM-recovery-v1'
source = OWNER.read_text()
assert hashlib.sha256(OWNER.read_bytes()).hexdigest() == '6263554c449a5db5f3e552a81d4e81861bd46ddcf2104e88f37288d0e9ba740a'

if '--prepare' in sys.argv:
    assert not INPUT.exists()
    failure = json.loads((AUDIT / 'TextCraft-PPO-global-OOM-exit.json').read_bytes())
    job = json.loads(CANONICAL.read_bytes())
    assert failure['kernel_class'] == 'global_oom' and failure['driver_pid'] == job['pid']
    assert not psutil.pid_exists(job['pid']), 'The recorded driver has exited; do not stop or duplicate a live run'
    run = Path(job['run'])
    launch = json.loads((run / 'launch.json').read_bytes())
    checkpoint_root = Path(launch['options']['trainer.default_local_dir'])
    step = int((checkpoint_root / 'latest_checkpointed_iteration.txt').read_text().strip())
    checkpoint = checkpoint_root / f'global_step_{step}'
    env = launch['effective_environment']
    sys.path[:0] = env['PYTHONPATH'].split(':')
    os.environ.update(env)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    states = []
    for role in ('actor', 'critic'):
        for rank in (0, 1):
            paths = {kind:checkpoint / role / f'{kind}_world_size_2_rank_{rank}.pt'
                     for kind in ('model', 'optim', 'extra_state')}
            assert all(p.is_file() and p.stat().st_size > 0 for p in paths.values())
            extra = torch.load(paths['extra_state'], map_location='cpu', weights_only=False)
            assert extra['lr_scheduler']['last_epoch'] == step
            assert {'cpu','cuda','numpy','random'}.issubset(extra['rng'])
            if role == 'actor':
                assert extra.get('rollout_gen_random_states') is not None
            optim = torch.load(paths['optim'], map_location='cpu', weights_only=False, mmap=True)
            assert optim['state']
            steps = sorted({float(s['step']) for s in optim['state'].values() if 'step' in s})
            states.append(dict(role=role, rank=rank, scheduler=step, optimizer_entries=len(optim['state']),
                               optimizer_steps=steps, rng_keys=sorted(extra['rng']),
                               rollout_rng_present=extra.get('rollout_gen_random_states') is not None,
                               files={k:dict(path=str(p),bytes=p.stat().st_size) for k,p in paths.items()}))
            del optim, extra
    data_path = checkpoint / 'data.pt'
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    assert data and not torch.cuda.is_initialized()
    INPUT.write_text(json.dumps(dict(status='native_PPO_exited_from_global_OOM_full_restore_required',
        collected=time.time(), job=job, checkpoint=str(checkpoint), completed_iteration=step,
        data_SHA256=hashlib.sha256(data_path.read_bytes()).hexdigest(), data_state_keys=list(data),
        saved_role_states=states, kernel_failure_receipt=str(AUDIT / 'TextCraft-PPO-global-OOM-exit.json'),
        model_calls=0, optimizer_updates=0, checkpoint_create_calls=0), indent=2) + '\n')

replacements = {
    "OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'": "OUT=ROOT/'receipts/appworld-formal-training-20261017-v1'",
    "stop=json.loads((OUT/'PPO-temporary-stop-for-v47-v54.json').read_bytes())": "stop=json.loads((OUT/'TextCraft-PPO-native-OOM-recovery-input.json').read_bytes())",
    "assert stop['status']=='temporarily_stopped_native_full_restore_required'": "assert stop['status']=='native_PPO_exited_from_global_OOM_full_restore_required'",
    'native-PPO-resume-after-v47-v55': PREFIX,
    "diag=OUT/'native-common-update-feedback-v47'\nresult=json.loads((diag/'result.json').read_bytes())\nassert result['phase']=='failed' or result['phase'].startswith('complete_'), 'Preserve running diagnostic'":
        "relocation=json.loads((OUT/'inactive-diagnostic-checkpoint-storage-relocation.json').read_bytes())\nassert relocation['status']=='passed_preserved_diagnostic_checkpoint_storage_relocation' and relocation['SHA256_verified']\nassert not psutil.pid_exists(stop['job']['pid'])",
    r"r'\|\s*[23]\s+\d+\s+'": r"r'\|\s*[45]\s+\d+\s+'",
    'GPU2/3 still occupied': 'GPU4/5 still occupied',
    '[(1419849,1791977138.31),(3174448,1791793629.73)]':
        '[(878971,1792091830.03),(1395665,1792177213.42),(1577251,1792178946.21)]',
    "env['CUDA_VISIBLE_DEVICES']='2,3'": "env['CUDA_VISIBLE_DEVICES']='4,5'",
    "    prepared.write_text(json.dumps(report,indent=2)+'\\n')":
        "    report['effective_environment']=dict(report['effective_environment'],CUDA_VISIBLE_DEVICES='4,5')\n    report['physical_gpus']=[4,5]\n    report['execution_version']='textcraft-PPO-global-OOM-recovery-20261017-v1'\n    prepared.write_text(json.dumps(report,indent=2)+'\\n')",
    '-v55.stdout.log': '-global-OOM-v1.stdout.log',
    '-v55.stderr.log': '-global-OOM-v1.stderr.log',
    'before-resume-v55.json': 'before-global-OOM-recovery-v1.json',
    'resume-v55-job.json': 'resume-global-OOM-recovery-v1-job.json',
    "(OUT/'native-PPO-control-v9-job.json')": "(ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json')",
    "job.update(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),":
        "job.update(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),physical_gpus=[4,5],execution_version='textcraft-PPO-global-OOM-recovery-20261017-v1',execution_patch_commit=os.environ['RL_RECOVERY_SOURCE_COMMIT'],actual_native_restore_receipt=None,actual_native_restore_observed_unix=None,actual_native_binding_receipt=None,",
    "user_authorized_temporary_stop_and_restore=True,hourly_automation='PAUSED',rootcause_repair_verified=False":
        "user_requested_preserve_existing_TextCraft_PPO=True,recovery_trigger='kernel_global_oom',hourly_automation='ACTIVE_PLOTTING_ONLY',rootcause_repair_verified=False",
}
for old,new in replacements.items():
    assert source.count(old) >= 1, old
    source = source.replace(old,new)
with contextlib.redirect_stdout(io.StringIO()):
    try:
        exec(compile(source,str(OWNER),'exec'),globals())
    except SystemExit as exc:
        assert '--prepare' in sys.argv and exc.code == 0
prepared_path = AUDIT / (PREFIX + '-prepared.json')
record = json.loads(prepared_path.read_bytes())
if '--prepare' in sys.argv:
    print(json.dumps(dict(status=record['status'],checkpoint=record['checkpoint'],
                          changes=record['configuration_changes'],physical_gpus=record['physical_gpus'],
                          owner_path=str(OWNER),owner_SHA256=hashlib.sha256(OWNER.read_bytes()).hexdigest())))
else:
    destination = AUDIT / (PREFIX + '-job.json')
    job = json.loads(destination.read_bytes())
    for name in ('formal-training.json','active-training.json'):
        path = ROOT / name
        manifest = json.loads(path.read_bytes())
        row, = [j for j in manifest['jobs'] if j.get('task') == 'TextCraft' and j.get('method') == 'ppo']
        assert row['pid'] == job['previous_job_identity']['pid'] and row['birth'] == job['previous_job_identity']['birth']
        row.update(pid=job['pid'],birth=job['birth'],observed_process_created_unix=job['birth'],
                   started_unix=job['started_unix'],status=job['status'],active=True,
                   execution_version=VERSION,execution_patch_commit=job['execution_patch_commit'],
                   resume_from_path=job['resume_from_path'],resumed_completed_iteration=job['resumed_completed_iteration'],
                   log=job['stdout'],actual_native_restore_receipt=None,actual_native_binding_receipt=None,
                   native_recovery_launch_receipt=str(destination),previous_failed_identity=job['previous_job_identity'])
        row.pop('ray_session',None)
        path.write_text(json.dumps(manifest,indent=2)+'\n')
    path = ROOT / 'active-source.json'
    active_source = json.loads(path.read_bytes())
    active_source['TextCraft_PPO_global_OOM_recovery_20261017_v1'] = dict(
        status=job['status'],pid=job['pid'],birth=job['birth'],execution_patch_commit=job['execution_patch_commit'],
        original_launch=job['run']+'/launch.json',original_source=job['run']+'/source.json',
        checkpoint=job['resume_from_path'],receipt=str(destination),method_or_training_hyperparameters_changed=False)
    path.write_text(json.dumps(active_source,indent=2)+'\n')
    print(json.dumps(dict(status=job['status'],pid=job['pid'],birth=job['birth'],physical_gpus=job['physical_gpus'],
                          checkpoint=job['resume_from_path'],receipt=str(destination))))
