"""Prepare an isolated source candidate from the actual approved v3 owners."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4'
VERSION='textcraft-dt-weighted-reward-targets-20261015-v4'
DEST=ROOT/'candidates'/VERSION
RUN=ROOT/'runs'/VERSION
OLD=ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
launch=json.loads((OLD/'launch.json').read_bytes())
old_env=launch['effective_environment']
assert not DEST.exists(), 'Do not replace a prepared or deployed candidate.'
DEST.mkdir()
for key,name in (('DT_ROOT','deltatrace'),('VERL_ROOT','verl-owner')):
    shutil.copytree(old_env[key],DEST/name,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
entry=DEST/'entry';entry.mkdir()
# This owner reads its unchanged native configuration beside __file__.
for directory in old_env['PYTHONPATH'].split(':'):
    original_config=Path(directory)/'owner_environment_configs.json'
    if original_config.is_file():
        shutil.copyfile(original_config,entry/original_config.name)
        break
else:
    raise FileNotFoundError('Original TextCraft owner configuration was not found.')
for path in (AUDIT/'source/candidate-source').glob('*.py'):
    target=(DEST/'deltatrace/clean/qwen35'/path.name if path.name.startswith('qwen35_')
            else DEST/'deltatrace/experiments/rl'/path.name if path.name=='deltatrace_credit.py'
            else entry/path.name)
    shutil.copyfile(path,target)
env=dict(old_env)
env.update(DT_ROOT=str(DEST/'deltatrace'),VERL_ROOT=str(DEST/'verl-owner'),
           DT_ENTRY_ROOT=str(entry),DT_CREDIT_MODE='weighted_reward_target_finite_v4',
           DT_FSDP_EVENT_WAIT_RECEIPTS=str(RUN/'worker-runtime-overrides'))
env['PYTHONPATH']=str(entry)+':'+env['PYTHONPATH'].replace(old_env['DT_ROOT'],env['DT_ROOT']).replace(old_env['VERL_ROOT'],env['VERL_ROOT'])
for key in ('RAY_ADDRESS','RAY_TMPDIR','MACA_VISIBLE_DEVICES'):env.pop(key,None)
options={key:value.replace(str(OLD),str(RUN)) if isinstance(value,str) else value for key,value in launch['options'].items()}
options['+ray_init.runtime_env.env_vars.DT_CREDIT_MODE']='weighted_reward_target_finite_v4'
argv=[arg.replace(str(OLD),str(RUN)) for arg in launch['argv']]
argv.append('+ray_init.runtime_env.env_vars.DT_CREDIT_MODE="weighted_reward_target_finite_v4"')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
original_files=json.loads((AUDIT/'baseline-manifest.json').read_text())
bindings={}
for name,record in original_files.items():
    original=Path(record['path'])
    if name.startswith('qwen35_'):current=DEST/'deltatrace/clean/qwen35'/name
    elif name in ('textcraft_owner_rollout.py','dt_training_batch.py','counterfactual.py'):current=entry/name
    elif name=='native_ray_trainer.py':current=DEST/'verl-owner/verl/trainer/ppo/ray_trainer.py'
    elif name=='native_fsdp_workers.py':current=DEST/'verl-owner/verl/workers/fsdp_workers.py'
    else:current=original
    bindings[str(current)]=sha(current)
for path in entry.glob('*.py'):bindings[str(path)]=sha(path)
bindings[str(entry/'owner_environment_configs.json')]=sha(entry/'owner_environment_configs.json')
bindings[str(DEST/'deltatrace/experiments/rl/deltatrace_credit.py')]=sha(DEST/'deltatrace/experiments/rl/deltatrace_credit.py')
fixed=('actor_rollout_ref.model.lora_rank','actor_rollout_ref.model.lora_alpha',
       'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu','actor_rollout_ref.actor.ppo_mini_batch_size',
       'actor_rollout_ref.actor.ppo_epochs','actor_rollout_ref.rollout.max_model_len',
       'trainer.save_freq','trainer.max_actor_ckpt_to_keep','trainer.total_epochs',
       'data.train_batch_size','env.rollout.n','trainer.resume_from_path','trainer.resume_mode')
assert all(options[key]==launch['options'][key] for key in fixed)
assert options['trainer.resume_from_path'] is None
record=dict(version=VERSION,status='prepared_only_not_deployed',run=str(RUN),argv=argv,
            options=options,effective_environment=env,physical_gpus=[0,1],budget=launch['budget'],
            fresh_base_model=True,resume_checkpoint=None,upstream_commit=launch['upstream_commit'],
            parent_patch_commit=launch['local_patch_commit'],local_patch_commit=None,
            previous_launch=str(OLD/'launch.json'),previous_launch_SHA256=sha(OLD/'launch.json'),
            source_bindings=bindings,original_verified_files=original_files,
            method_changes=['Only native reward-settlement action spans are targets, including score 0.',
                            'Per-span affine credit seeds, linear source credit; no exponentiation.',
                            'Original GRPO multiplier, variance-only standardization, self-target and PPO preserved.'])
(AUDIT/'prepared-launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],candidate=str(DEST),prepared=str(AUDIT/'prepared-launch.json'),fixed_configuration={key:options[key] for key in fixed})))
