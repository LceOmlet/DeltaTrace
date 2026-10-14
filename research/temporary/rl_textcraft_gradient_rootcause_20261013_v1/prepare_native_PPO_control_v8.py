"""Prepare the user-authorized native PPO control without changing any owner."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from omegaconf import OmegaConf

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
OLD=ROOT/'runs/textcraft-grpo-formal-native-graph-fresh-20261012-v1'
RUN=ROOT/'runs/textcraft-ppo-control-fixed-verl-fresh-20261014-v1'
PREPARED=OUT/'native-PPO-control-v8-prepared'
assert not RUN.exists(),'Preserve existing runs; no duplicate'
if PREPARED.exists():
    previous=json.loads((PREPARED/'launch.json').read_bytes())
    assert previous['status']=='prepared_only_user_authorized_native_PPO_control'
    assert not (PREPARED/'launch-draft-v8.json').exists(),'Inspect the final preparation before changing it again'
    (PREPARED/'launch-draft-v8.json').write_bytes((PREPARED/'launch.json').read_bytes())
    (PREPARED/'source-draft-v8.json').write_bytes((PREPARED/'source.json').read_bytes())
launch=json.loads((OLD/'launch.json').read_bytes())
source=json.loads((OLD/'source.json').read_bytes())
VERL=Path(launch['effective_environment']['VERL_ROOT'])
expected={'verl/workers/actor/dp_actor.py':'3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
    'verl/trainer/ppo/core_algos.py':'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
    'verl/workers/fsdp_workers.py':'9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3',
    'verl/utils/checkpoint/fsdp_checkpoint_manager.py':'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert {key:sha(VERL/key) for key in expected}==expected
options=copy.deepcopy(launch['options'])
for key,value in list(options.items()):
    if isinstance(value,str):options[key]=value.replace(str(OLD),str(RUN))
for key in ('trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir'):
    options[key]=options[key].replace('/textcraft-grpo/','/textcraft-ppo/')
model=options['actor_rollout_ref.model.path']
changes={
    'algorithm.adv_estimator':'gae',
    'trainer.experiment_name':'textcraft-ppo-native-gae-20261014-v1',
    'critic.strategy':'fsdp2',
    'critic.model.path':model,
    'critic.ppo_micro_batch_size_per_gpu':4,
    'critic.model.enable_activation_offload':True,
    'critic.model.fsdp_config.param_offload':True,
    'critic.model.fsdp_config.optimizer_offload':True,
    'critic.model.fsdp_config.offload_policy':True,
    'trainer.max_critic_ckpt_to_keep':2,
}
options.update(changes)
config=OmegaConf.load(VERL/'verl/trainer/config/ppo_trainer.yaml')
original_config=OmegaConf.load(VERL/'verl/trainer/config/ppo_trainer.yaml')
for key,value in launch['options'].items():OmegaConf.update(original_config,key.lstrip('+'),value,force_add=True)
for key,value in options.items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
assert config.algorithm.gamma==1 and config.algorithm.lam==1
assert config.critic.optim.lr==1e-5 and config.critic.model.lora_rank==0
assert config.critic.ppo_mini_batch_size==64 and config.critic.ppo_epochs==1
assert config.critic.forward_micro_batch_size_per_gpu==4
assert config.critic.ppo_max_token_len_per_gpu==32768
fixed={'actor_rollout_ref.model.lora_rank':8,'actor_rollout_ref.model.lora_alpha':16,
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':4,'actor_rollout_ref.actor.ppo_mini_batch_size':64,
    'actor_rollout_ref.actor.ppo_epochs':1,'actor_rollout_ref.rollout.max_model_len':32768,
    'actor_rollout_ref.actor.ppo_max_token_len_per_gpu':32768,'env.rollout.n':8,
    'data.train_batch_size':32,'trainer.total_epochs':30,'trainer.save_freq':5,
    'trainer.total_training_steps':None,'trainer.resume_from_path':None,
    'actor_rollout_ref.rollout.enforce_eager':False,'actor_rollout_ref.rollout.free_cache_engine':False}
assert {key:options[key] for key in fixed}==fixed
for key in ('actor_rollout_ref.actor.entropy_coeff','actor_rollout_ref.actor.clip_ratio_c',
    'actor_rollout_ref.actor.kl_loss_coef','actor_rollout_ref.actor.use_kl_loss',
    'actor_rollout_ref.actor.optim.lr'):
    assert OmegaConf.select(config,key)==OmegaConf.select(original_config,key),key

def hydra_value(value):
    # The original argv is preserved except changed/additional overrides.
    # JSON quoted strings/lists and native bool/null literals are Hydra syntax.
    return json.dumps(value,separators=(',',':'))
argv=[];found=set()
for arg in launch['argv']:
    key=arg.split('=',1)[0]
    if key in options and options[key]!=launch['options'].get(key):
        argv.append(key+'='+hydra_value(options[key]));found.add(key)
    else:
        argv.append(arg.replace(str(OLD),str(RUN)))
for key in options:
    if key not in launch['options']:
        assert key in changes,key
        argv.append(key+'='+hydra_value(options[key]));found.add(key)
env=dict(launch['effective_environment'])
env['CUDA_VISIBLE_DEVICES']='2,3'
env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(RUN/'worker-runtime-overrides')
for key in list(env):
    if key in ('RAY_ADDRESS','RAY_TMPDIR','MACA_VISIBLE_DEVICES','VERL_GLOBAL_TOKEN_MEAN','DTPO_BEHAVIOR_PROBE') or key.startswith(('DT_HOLD','DT_STOP_AFTER','DT_ACTOR_INCIDENT')):
        env.pop(key)
binding={str(VERL/key):value for key,value in expected.items()}
for relative in ('verl/workers/critic/dp_critic.py','verl/trainer/config/ppo_trainer.yaml'):
    binding[str(VERL/relative)]=sha(VERL/relative)
from transformers import Qwen3_5ForTokenClassification
import inspect
native_classifier=Path(inspect.getsourcefile(Qwen3_5ForTokenClassification))
binding[str(native_classifier)]=sha(native_classifier)
record={'unix':time.time(),'status':'prepared_only_user_authorized_native_PPO_control','method':'ppo','task':'TextCraft PPO',
    'run':str(RUN),'argv':argv,'options':options,'effective_environment':env,'physical_gpus':[2,3],
    'fresh_base_model':True,'resume_checkpoint':None,'budget':launch.get('budget'),
    'algorithm':'Native fixed VERL PPO/GAE with its learned critic; not a DT advantage substitute.',
    'parent_launch':str(OLD/'launch.json'),'parent_launch_SHA256':sha(OLD/'launch.json'),
    'source_bindings':binding,'local_patch_commit':launch.get('local_patch_commit'),
    'changes_from_GRPO_launch':{key:{'before':launch['options'].get(key),'after':value} for key,value in options.items() if value!=launch['options'].get(key)},
    'fixed_actor_configuration':fixed,
    'critic_native_defaults':{'learning_rate':1e-5,'lora_rank':0,'gamma':1.0,'lam':1.0,
        'grad_clip':float(config.critic.grad_clip),'cliprange_value':float(config.critic.cliprange_value),
        'initial_storage_dtype':str(config.critic.model.fsdp_config.get('model_dtype','fp32')),
        'forward_dtype':'Native original FSDP2 mixed precision; unchanged owner.',
        'activation_and_parameter_optimizer_offload':'Native capability enabled to share the two freed GPUs; no algorithm change.'},
    'no_quality_gate':True,'GRPO_untouched':True,'old_DTPO_stays_stopped':True,
    'hourly_automation_stays_paused':True,'formal_manifests_changed':False,
    'rootcause_repair_acceptance':False,'actual_worker_bindings':'pending launch and readback'}
new_source=copy.deepcopy(source)
for key in ('runtime_override','deployment','passive_credit_recording_runtime_override','initial_worker_state_verification',
    'actual_initial_worker_state_receipt','actual_initial_worker_state_verified','actual_initial_worker_states'):
    new_source.pop(key,None)
new_source.update(algorithm='gae',method='ppo',task='TextCraft PPO',DT_credit_used=False,environment=env,
    source_bindings=binding,inherited_source=str(OLD/'source.json'),inherited_source_SHA256=sha(OLD/'source.json'),
    deployment={'status':record['status']},fresh_base_model=True,resume_from_path=None,
    actual_import_scope='Prepared original disk bindings; no worker launched yet.')
PREPARED.mkdir(exist_ok=True)
(PREPARED/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
(PREPARED/'source.json').write_text(json.dumps(new_source,indent=2)+'\n')
(PREPARED/'effective-config.yaml').write_text(OmegaConf.to_yaml(config,resolve=True))
report={'status':record['status'],'prepared':str(PREPARED),'run':str(RUN),'argv_prefix':argv[:4],
    'source_bindings':binding,'fixed_actor_configuration':fixed,'critic_native_defaults':record['critic_native_defaults'],
    'changes_from_GRPO_launch':record['changes_from_GRPO_launch'],'disk':subprocess.run(['df','-h',str(ROOT),'/tmp'],capture_output=True,text=True).stdout,
    'cgroup_memory':{str(p):p.read_text() for p in [Path('/sys/fs/cgroup/memory.current'),Path('/sys/fs/cgroup/memory.max'),
        Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'),Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')] if p.exists()}}
dest=OUT/'native-PPO-control-preparation-v8.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
