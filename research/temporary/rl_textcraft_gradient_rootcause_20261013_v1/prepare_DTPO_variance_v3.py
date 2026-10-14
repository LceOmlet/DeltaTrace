"""Prepare user-approved v3 using native variance and unchanged native PPO."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
from omegaconf import OmegaConf

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
OLD=ROOT/'runs/textcraft-dt-grpo-multiplier-no-whiten-20261014-v1'
VERSION='textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
RUN=ROOT/'runs'/VERSION
DEST=ROOT/'candidates'/VERSION
PREPARED=OUT/'DTPO-variance-v3-prepared'
assert not RUN.exists() and not DEST.exists() and not PREPARED.exists()
launch=json.loads((OLD/'launch.json').read_bytes())
parent_source=json.loads((OLD/'source.json').read_bytes())
old_verl=Path(launch['effective_environment']['VERL_ROOT'])
old_entry=Path(launch['effective_environment']['PYTHONPATH'].split(':')[0])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

expected={'verl/workers/actor/dp_actor.py':'3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
    'verl/trainer/ppo/core_algos.py':'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
    'verl/workers/fsdp_workers.py':'9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3',
    'verl/utils/checkpoint/fsdp_checkpoint_manager.py':'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01',
    'verl/utils/torch_functional.py':'079a6d20696a861340687d6e61fb4162cc1d436837ce747ac9ef318b738c1701',
    'verl/trainer/ppo/ray_trainer.py':'63b0d623380bab3173eaf2c7148bddf21b557827625bd5eff25c5b4faa05daa4'}
assert {k:sha(old_verl/k) for k in expected}==expected
patcher_path=OUT/'canonical-patch_verl_agent2-DTPO-v3.py'
spec=importlib.util.spec_from_file_location('canonical_DTPO_v3',patcher_path)
patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)
before=(old_verl/'verl/trainer/ppo/ray_trainer.py').read_text()
after=patcher.patch_dt_advantage_preprocessing(before)
assert after!=before and patcher.patch_dt_advantage_preprocessing(after)==after
assert 'advantages = advantages * torch.rsqrt(verl_F.masked_var(advantages, response_mask) + 1e-8)' in after
compile(after,'ray_trainer.py','exec')
verl=DEST/'verl-owner';entry=DEST/'entry'
DEST.mkdir(parents=True)
shutil.copytree(old_verl,verl,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
(verl/'verl/trainer/ppo/ray_trainer.py').write_text(after,newline='\n')
entry.mkdir();shutil.copyfile(old_entry/'dt_training_batch.py',entry/'dt_training_batch.py')
assert sha(entry/'dt_training_batch.py')=='a4e1f70b42082972324199b328975fb87ec6600ab9a572d5ca2b2040bebc54f7'
changed=[]
for directory,dirs,files in os.walk(old_verl,followlinks=True):
    dirs[:]=[d for d in dirs if d not in ('.git','__pycache__')]
    for name in files:
        if name.endswith('.py'):
            p=Path(directory)/name
            if sha(p)!=sha(verl/p.relative_to(old_verl)):changed.append(str(p.relative_to(old_verl)))
assert changed==['verl/trainer/ppo/ray_trainer.py'],changed
options=copy.deepcopy(launch['options'])
for k,v in list(options.items()):
    if isinstance(v,str):options[k]=v.replace(str(OLD),str(RUN))
argv=[a.replace(str(OLD),str(RUN)) for a in launch['argv']]
changes={k:dict(before=launch['options'][k],after=v) for k,v in options.items() if launch['options'][k]!=v}
assert set(changes)=={'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir',
                      '+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS'}
fixed={'actor_rollout_ref.model.lora_rank':8,'actor_rollout_ref.model.lora_alpha':16,
 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':4,'actor_rollout_ref.actor.ppo_mini_batch_size':64,
 'actor_rollout_ref.actor.ppo_epochs':1,'actor_rollout_ref.rollout.max_model_len':32768,
 'actor_rollout_ref.actor.ppo_max_token_len_per_gpu':32768,'env.rollout.n':8,'data.train_batch_size':32,
 'trainer.total_epochs':30,'trainer.save_freq':5,'trainer.max_actor_ckpt_to_keep':2,
 'trainer.total_training_steps':None,'trainer.resume_from_path':None,
 'actor_rollout_ref.rollout.enforce_eager':False,'actor_rollout_ref.rollout.free_cache_engine':False}
assert {k:options[k] for k in fixed}==fixed
config=OmegaConf.load(verl/'verl/trainer/config/ppo_trainer.yaml')
for k,v in options.items():OmegaConf.update(config,k.lstrip('+'),v,force_add=True)
env=dict(launch['effective_environment'])
env.update(VERL_ROOT=str(verl),PYTHONPATH=env['PYTHONPATH'].replace(str(old_entry),str(entry)).replace(str(old_verl),str(verl)),
           DT_FSDP_EVENT_WAIT_RECEIPTS=str(RUN/'worker-runtime-overrides'),CUDA_VISIBLE_DEVICES='0,1')
assert 'VERL_GLOBAL_TOKEN_MEAN' not in env
for k in ('RAY_ADDRESS','RAY_TMPDIR','MACA_VISIBLE_DEVICES'):env.pop(k,None)
bindings={}
for path,digest in parent_source['source_bindings'].items():
    current=path.replace(str(old_verl),str(verl)).replace(str(old_entry),str(entry))
    if Path(current).is_file():
        if current!=str(verl/'verl/trainer/ppo/ray_trainer.py'):assert sha(current)==digest,current
        bindings[current]=sha(current)
bindings.update({str(verl/k):sha(verl/k) for k in expected})
bindings[str(entry/'dt_training_batch.py')]=sha(entry/'dt_training_batch.py')
record=dict(unix=time.time(),status='prepared_only_v3_not_deployed',version=VERSION,method='deltatrace',
 run=str(RUN),argv=argv,options=options,effective_environment=env,physical_gpus=[0,1],budget=launch['budget'],
 fresh_base_model=True,resume_checkpoint=None,upstream_commit='20bd331b',local_patch_commit=None,
 parent_deployed_patch_commit=launch['local_patch_commit'],source_bindings=bindings,
 changed_owner_files=changed,configuration_changes=changes,fixed_configuration=fixed,
 canonical_patcher=dict(path=str(patcher_path),SHA256=sha(patcher_path)),
 previous_launch=str(OLD/'launch.json'),previous_launch_SHA256=sha(OLD/'launch.json'),
 semantic_changes=['Keep official GRPO group multiplier and original DT composition.',
                   'Scale DT advantages by inverse native masked standard deviation, without subtracting mean.'],
 variance_standardization=dict(owner='verl.utils.torch_functional.masked_var',epsilon=1e-8,subtract_mean=False,
   expression='A * torch.rsqrt(verl_F.masked_var(A, response_mask) + 1e-8)',
   official_shift_mean_false_is_different=True),
 PPO_reduction='unchanged official token-mean; 1/16 sum_j(S_j/N_j)',
 rootcause_repair_verified=False,GRPO_rerun=False,PPO_rerun=False,hourly_automation='PAUSED')
source=dict(version=VERSION,upstream_commit='20bd331b',local_patch_commit=None,prepared_only=True,
 inherited_source=str(OLD/'source.json'),inherited_source_SHA256=sha(OLD/'source.json'),
 dt_root=env['DT_ROOT'],verl_root=str(verl),entry=str(entry),pythonpath=env['PYTHONPATH'],
 environment=env,source_bindings=bindings,semantic_changes=record['semantic_changes'],
 variance_standardization=record['variance_standardization'],PPO_reduction=record['PPO_reduction'],
 fresh_base_model=True,resume_from_path=None,actual_import_scope='Prepared disk identity; actual worker imports pending.',
 deployment=dict(status=record['status']))
PREPARED.mkdir();(PREPARED/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
(PREPARED/'source.json').write_text(json.dumps(source,indent=2)+'\n')
(PREPARED/'effective-config.yaml').write_text(OmegaConf.to_yaml(config,resolve=True))
import difflib
(PREPARED/'ray_trainer.py.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='v1/ray_trainer.py',tofile='v3/ray_trainer.py')))
print(json.dumps(dict(receipt=str(PREPARED/'launch.json'),report=record)))
