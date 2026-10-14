"""Restore only the native critic activation-offload default after its assertion."""
import hashlib
import json
from pathlib import Path
import time
import psutil
from omegaconf import OmegaConf

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
old=json.loads((OUT/'native-PPO-control-v8-job.json').read_bytes())
assert not psutil.pid_exists(old['pid']) or psutil.Process(old['pid']).create_time()!=old['birth']
failed=Path(old['run'])
new=ROOT/'runs/textcraft-ppo-control-fixed-verl-fresh-20261014-v2'
prepared=OUT/'native-PPO-control-v9-prepared'
assert not new.exists(), 'Preserve prior runs.'
if prepared.exists():
    prior=json.loads((prepared/'launch.json').read_bytes())
    assert prior['status']=='prepared_only_native_PPO_critic_default_activation_offload'
    draft=prepared/'draft-before-metadata-finalization-v9'
    draft.mkdir()
    for name in ('launch.json','source.json','effective-config.yaml'):
        (draft/name).write_bytes((prepared/name).read_bytes())
stderr=(failed/'driver.stderr.log').read_text(errors='replace')
assert 'activation_offload.py' in stderr and 'assert not isinstance(tensor, tuple)' in stderr

def rewrite(value):
    if isinstance(value,str):return value.replace(str(failed),str(new))
    if isinstance(value,list):return [rewrite(v) for v in value]
    if isinstance(value,dict):return {k:rewrite(v) for k,v in value.items()}
    return value

record=rewrite(json.loads((failed/'launch.json').read_bytes()))
record['options']['critic.model.enable_activation_offload']=False
record['options']['trainer.experiment_name']='textcraft-ppo-native-gae-20261014-v2'
key='critic.model.enable_activation_offload='
assert sum(a.startswith(key) for a in record['argv'])==1
record['argv']=[key+'false' if a.startswith(key) else a for a in record['argv']]
record['argv']=['trainer.experiment_name=textcraft-ppo-native-gae-20261014-v2'
                if a.startswith('trainer.experiment_name=') else a for a in record['argv']]
config=OmegaConf.load(failed/'effective-config.yaml')
config=OmegaConf.create(rewrite(OmegaConf.to_container(config,resolve=True)))
assert config.critic.model.enable_activation_offload is True
config.critic.model.enable_activation_offload=False
config.trainer.experiment_name='textcraft-ppo-native-gae-20261014-v2'
assert config.critic.model.enable_gradient_checkpointing is True
assert config.actor_rollout_ref.model.enable_activation_offload is True
assert config.algorithm.adv_estimator=='gae'
assert config.critic.model.lora_rank==0 and config.critic.optim.lr==1e-5
record.update(unix=time.time(),status='prepared_only_native_PPO_critic_default_activation_offload',
              run=str(new),parent_failed_run=str(failed),
              changes_from_v1={'critic.model.enable_activation_offload':{'before':True,'after':False}},
              actual_worker_bindings='pending new v2 launch and readback',
              compatibility_scope='Use original critic nonreentrant gradient checkpointing without the optional async activation offloader. No training owner source or mathematical configuration change.',
              no_new_offloader_implementation=True)
record['critic_native_defaults']['activation_and_parameter_optimizer_offload']='Original critic activation offload defaultFalse; existing native parameter/optimizer offload remains enabled.'
for path,sha in record['source_bindings'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
source=rewrite(json.loads((failed/'source.json').read_bytes()))
source.update(deployment={'status':record['status']},parent_failed_run=str(failed),
              configuration_change=record['changes_from_v1'],
              actual_import_scope='Original owner disk bindings preserved; v2 workers not launched yet.')
failure={'unix':time.time(),'job':old,'status':'failed_native_critic_activation_offload_before_first_complete_iteration',
         'exception':'AsyncDoubleBufferGroupOffloadHandler.tensor_pop: assert not isinstance(tensor, tuple)',
         'source':{'path':str(failed/'driver.stderr.log'),'SHA256':hashlib.sha256((failed/'driver.stderr.log').read_bytes()).hexdigest()},
         'completed_training_iterations':0,'actor_update_reached':False,
         'critic_optimizer_updates_at_failure':'Not read back after process exit; do not infer from empty initial state.',
         'not_DTPO_gradient_rootcause_evidence':True,'owner_code_modified':False}
if not (failed/'completion-status-v9.json').exists():
    (failed/'completion-status-v9.json').write_text(json.dumps(failure,indent=2)+'\n')
prepared.mkdir(exist_ok=True)
(prepared/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
(prepared/'source.json').write_text(json.dumps(source,indent=2)+'\n')
(prepared/'effective-config.yaml').write_text(OmegaConf.to_yaml(config,resolve=True))
report={'unix':time.time(),'scope':__doc__,'failure':failure,'prepared':str(prepared),'run':str(new),
        'source_bindings':record['source_bindings'],'changes':record['changes_from_v1'],
        'source_changes':0,'actor_training_hyperparameter_changes':0,'GPU_model_calls':0,
        'native_critic_gradient_checkpointing':True,'native_critic_activation_offload':False,
        'full_32k_critic_training_capacity':'Unverified; the configured32k/B4 limit is retained.'}
dest=OUT/'native-PPO-offload-default-v9-preparation.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
