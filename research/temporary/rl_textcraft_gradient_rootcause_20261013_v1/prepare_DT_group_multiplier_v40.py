"""Prepare precisely the authorized native GRPO multiplier and no DT whitening.

No GRPO, PPO, optimizer, scheduler, generation or DT algorithm is implemented
here. The two changed source files only connect the existing native GRPO
result to the existing DT composition and remove its final whitening.
"""
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from omegaconf import OmegaConf

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
OLD = ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
VERSION = 'textcraft-dt-grpo-multiplier-no-whiten-20261014-v1'
RUN = ROOT/'runs'/VERSION
DEST = ROOT/'candidates'/VERSION
PREPARED = OUT/'DT-group-multiplier-v40-prepared'
assert not RUN.exists(), 'Preserve existing runs'
launch = json.loads((OLD/'launch.json').read_bytes())
source = json.loads((OLD/'source.json').read_bytes())
old_verl = Path(launch['effective_environment']['VERL_ROOT'])
old_entry = next(Path(p) for p in launch['effective_environment']['PYTHONPATH'].split(':')
                 if (Path(p)/'dt_training_batch.py').is_file())
expected = {
    'verl/workers/actor/dp_actor.py': '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
    'verl/trainer/ppo/core_algos.py': 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
    'verl/workers/fsdp_workers.py': '9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3',
    'verl/utils/checkpoint/fsdp_checkpoint_manager.py': 'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01',
    'verl/trainer/ppo/ray_trainer.py': '7366557b482e604d66f47bdc4841ea147ffaac7c80538fa000544bb4e92eb619',
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert {k:sha(old_verl/k) for k in expected} == expected
assert sha(old_entry/'dt_training_batch.py') == '1a3f39d2ae11add49b1edb9077e915bf5f28d3f7fc9da929dd00fba79413004e'

def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)

trainer = (old_verl/'verl/trainer/ppo/ray_trainer.py').read_text()
trainer_new = replace_once(trainer,
    '        advantages = verl_F.masked_whiten(data.batch["dt_token_advantages"], response_mask)\n',
    '        advantages = data.batch["dt_token_advantages"]\n')
anchor = ('                        if self.config.algorithm.adv_estimator == AdvantageEstimator.DELTATRACE:\n'
          '                            from dt_training_batch import compute_training_credit\n')
replacement = ('                        if self.config.algorithm.adv_estimator == AdvantageEstimator.DELTATRACE:\n'
    '                            # Native GRPO owns group membership, reward reduction and normalization.\n'
    '                            group_view = compute_advantage(\n'
    '                                batch.select(batch_keys=["responses", "token_level_rewards", "response_mask", "loss_mask"],\n'
    '                                             non_tensor_batch_keys=["uid", "traj_uid"]),\n'
    '                                adv_estimator=AdvantageEstimator.GRPO,\n'
    '                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,\n'
    '                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,\n'
    '                            )\n'
    '                            # GRPO broadcasts one coefficient over each row\'s valid action tokens.\n'
    '                            first_policy = batch.batch["response_mask"].long().argmax(-1, keepdim=True)\n'
    '                            batch.batch["dt_grpo_multipliers"] = group_view.batch["advantages"].gather(1, first_policy).squeeze(1)\n'
    '                            from dt_training_batch import compute_training_credit\n')
trainer_new = replace_once(trainer_new, anchor, replacement)
batch = (old_entry/'dt_training_batch.py').read_text()
batch_new = replace_once(batch,
    "    rewards = data.batch['token_level_rewards'].double().sum(-1).cpu()[first]\n",
    "    # The trainer supplies the unchanged official GRPO result before deduplication.\n"
    "    # This replaces the return multiplier; raw environment rewards remain untouched.\n"
    "    rewards = data.batch['dt_grpo_multipliers'].double().cpu()[first]\n")
compile(trainer_new, 'ray_trainer.py', 'exec')
compile(batch_new, 'dt_training_batch.py', 'exec')
verl = DEST/'verl-owner'
entry = DEST/'entry'
if not DEST.exists():
    DEST.mkdir(parents=True)
    shutil.copytree(old_verl, verl, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    (verl/'verl/trainer/ppo/ray_trainer.py').write_text(trainer_new, newline='\n')
    entry.mkdir()
    (entry/'dt_training_batch.py').write_text(batch_new, newline='\n')
else:
    # The first preparation stopped at a traversal check: pathlib.rglob does
    # not descend through the baseline's symlinked package directories.
    # Reuse exactly those prepared bytes, without recopying or repatching.
    assert not (PREPARED/'launch.json').exists()
    assert (verl/'verl/trainer/ppo/ray_trainer.py').read_text() == trainer_new
    assert (entry/'dt_training_batch.py').read_text() == batch_new
assert {k:sha(old_verl/k) for k in expected} == expected, 'Baseline must remain unchanged'
PREPARED.mkdir(exist_ok=True)
for name, before, after in [('ray_trainer.py',trainer,trainer_new), ('dt_training_batch.py',batch,batch_new)]:
    (PREPARED/(name+'.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                              fromfile='baseline/'+name, tofile=VERSION+'/'+name)))
unchanged = {str(verl/k):sha(verl/k) for k in expected if k != 'verl/trainer/ppo/ray_trainer.py'}
assert {k:sha(verl/k) for k in expected if k != 'verl/trainer/ppo/ray_trainer.py'} == {k:v for k,v in expected.items() if k != 'verl/trainer/ppo/ray_trainer.py'}
changed_owner_files = []
for directory, dirs, files in os.walk(old_verl, followlinks=True):
    dirs[:] = [d for d in dirs if d not in ('.git','__pycache__')]
    for name in files:
        if name.endswith('.py'):
            p = Path(directory)/name
            if sha(p) != sha(verl/p.relative_to(old_verl)):
                changed_owner_files.append(str(p.relative_to(old_verl)))
assert changed_owner_files == ['verl/trainer/ppo/ray_trainer.py'], changed_owner_files
options = copy.deepcopy(launch['options'])
argv = [a.replace(str(OLD), str(RUN)) for a in launch['argv']]
for k,v in list(options.items()):
    if isinstance(v,str): options[k] = v.replace(str(OLD),str(RUN))
changes = {k:dict(before=launch['options'][k],after=v) for k,v in options.items() if launch['options'][k] != v}
assert set(changes) == {'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir',
                        '+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS'}, changes
config = OmegaConf.load(verl/'verl/trainer/config/ppo_trainer.yaml')
for k,v in options.items(): OmegaConf.update(config,k.lstrip('+'),v,force_add=True)
assert config.algorithm.get('norm_adv_by_std_in_grpo',True) is True
fixed = {'actor_rollout_ref.model.lora_rank':8,'actor_rollout_ref.model.lora_alpha':16,
         'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':4,'actor_rollout_ref.actor.ppo_mini_batch_size':64,
         'actor_rollout_ref.actor.ppo_epochs':1,'actor_rollout_ref.rollout.max_model_len':32768,
         'actor_rollout_ref.actor.ppo_max_token_len_per_gpu':32768,'env.rollout.n':8,'data.train_batch_size':32,
         'trainer.total_epochs':30,'trainer.save_freq':5,'trainer.max_actor_ckpt_to_keep':2,
         'trainer.total_training_steps':None,'trainer.resume_from_path':None,
         'actor_rollout_ref.rollout.enforce_eager':False,'actor_rollout_ref.rollout.free_cache_engine':False}
assert {k:options[k] for k in fixed} == fixed
env = dict(launch['effective_environment'])
env.update(VERL_ROOT=str(verl), CUDA_VISIBLE_DEVICES='0,1',
           PYTHONPATH=str(entry)+':'+env['PYTHONPATH'].replace(str(old_verl),str(verl)),
           DT_FSDP_EVENT_WAIT_RECEIPTS=str(RUN/'worker-runtime-overrides'))
for k in list(env):
    if k in ('RAY_ADDRESS','RAY_TMPDIR','MACA_VISIBLE_DEVICES'):
        env.pop(k)
assert 'VERL_GLOBAL_TOKEN_MEAN' not in env
for k in env:
    assert not k.startswith(('DT_HOLD','DT_STOP_AFTER','DT_ACTOR_INCIDENT')), k
bindings = dict(unchanged)
bindings.update({str(verl/'verl/trainer/ppo/ray_trainer.py'):sha(verl/'verl/trainer/ppo/ray_trainer.py'),
                 str(entry/'dt_training_batch.py'):sha(entry/'dt_training_batch.py')})
record = dict(unix=time.time(),status='prepared_user_authorized_two_changes_not_yet_submitted',method='deltatrace',
    version=VERSION,run=str(RUN),argv=argv,options=options,effective_environment=env,physical_gpus=[0,1],
    budget=launch['budget'],fresh_base_model=True,resume_checkpoint=None,
    upstream_commit='20bd331b',parent_deployed_patch_commit=launch['local_patch_commit'],
    local_patch_commit=None,source_bindings=bindings,unchanged_owner_files=unchanged,
    changed_owner_files=changed_owner_files,configuration_changes=changes,fixed_configuration=fixed,
    previous_launch=str(OLD/'launch.json'),previous_launch_SHA256=sha(OLD/'launch.json'),
    semantic_changes=['Original GRPO group multiplier replaces raw return in existing token DT composition.',
                      'No masked_whiten after DT composition; no replacement normalization.'],
    analytic_credit='source=g_Y*(-expm1(-d_i));self_target=g_Y;observation=padding=0',
    checkpoint_contract='Original model/optimizer/scheduler/training RNG/data/global_step plus accepted rollout RNG extension; save5/keep2 unchanged.',
    inherited_checkpoint_verification=str(ROOT/'receipts/textcraft-native-graph-deployment-20261012-v1/integration-result.json'),
    rootcause_repair_verified=False,GRPO_rerun=False,PPO_rerun=False,hourly_automation='PAUSED')
new_source = copy.deepcopy(source)
for k in ('runtime_override','deployment','passive_credit_recording_runtime_override','initial_worker_state_verification',
          'actual_initial_worker_state_receipt','actual_initial_worker_state_verified','actual_initial_worker_states'):
    new_source.pop(k,None)
new_source.update(version=VERSION,verl_root=str(verl),environment=env,local_patch_commit=None,
    inherited_source=str(OLD/'source.json'),inherited_source_SHA256=sha(OLD/'source.json'),
    source_bindings=bindings,semantic_changes=record['semantic_changes'],fresh_base_model=True,resume_from_path=None,
    deployment=dict(status=record['status']),actual_import_scope='Prepared disk identity; actual worker imports pending.')
(PREPARED/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
(PREPARED/'source.json').write_text(json.dumps(new_source,indent=2)+'\n')
(PREPARED/'effective-config.yaml').write_text(OmegaConf.to_yaml(config,resolve=True))
manifest_shapes = {}
for name in ('formal-training.json','active-training.json','active-source.json'):
    value = json.loads((ROOT/name).read_bytes())
    manifest_shapes[name] = {k:type(v).__name__ for k,v in value.items()} if isinstance(value,dict) else dict(type=type(value).__name__)
report = dict(prepared=str(PREPARED),run=str(RUN),version=VERSION,status=record['status'],
              source_bindings=bindings,configuration_changes=changes,manifest_shapes=manifest_shapes)
dest = OUT/'DT-group-multiplier-preparation-v40.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
