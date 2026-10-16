"""Prepare the same original native AppWorld PPO trial with verified 32k memory support.

Only candidate owner, optional critic offload, and independent execution paths
change. Do not launch until numerical and matched efficiency receipts pass.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION='appworld-native-PPO-32k-speed-20261016-v2'
AUDIT=ROOT/'receipts'/VERSION
DEST=ROOT/'candidates'/VERSION
NEW=ROOT/'candidates/PPO-32k-native-efficiency-20261016-v2/verl-owner'
OLD=ROOT/'candidates/textcraft-dt-reward-target-exponential-20261016-v5/verl-owner'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not DEST.exists(),'Preserve an existing trial'
previous=ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json'
p=json.loads(previous.read_bytes())
assert all(sha(path)==expected for path,expected in p['source_bindings'].items())
numeric=ROOT/'receipts/PPO-32k-numeric-reference-20261016-v3/same-session-status.json'
n=json.loads(numeric.read_bytes());assert n['numerical_accepted'] and n['native_async_post_values_max_abs']==0
assert sha(NEW/'verl/workers/fsdp_workers.py')=='76f3efbf53e25bd9919ae79de061cd6a47401d8e93df3d5c64bf397e09aa94dd'
assert sha(NEW/'verl/utils/activation_offload.py')=='f62bd44a203f8ee27480792769281f4259d299beb74db4a215031bd6b6d5b423'
authorities={str(ROOT/name):sha(ROOT/name) for name in ('formal-training.json','active-training.json','active-source.json')}
run=ROOT/'runs'/VERSION/'appworld';oldrun=p['run']
p['options']={key:value.replace(oldrun,str(run)) if isinstance(value,str) else value for key,value in p['options'].items()}
p['options']['trainer.experiment_name']=VERSION
p['options']['critic.model.enable_activation_offload']=True
env={key:value.replace(str(OLD),str(NEW)).replace(oldrun,str(run)) if isinstance(value,str) else value
    for key,value in p['effective_environment'].items()}
env['VERL_ROOT']=str(NEW);env['FLASH_ATTENTION_DETERMINISTIC']='0'
env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(run/'worker-runtime-overrides')
env['CUDA_VISIBLE_DEVICES']='2,3';env.pop('MACA_VISIBLE_DEVICES',None)
p['effective_environment']=env
p['source_bindings']={path.replace(str(OLD),str(NEW)):sha(path.replace(str(OLD),str(NEW))) for path in p['source_bindings']}
p['source_bindings'][str(NEW/'verl/utils/activation_offload.py')]=sha(NEW/'verl/utils/activation_offload.py')
for name in ('verl/workers/critic/dp_critic.py','verl/workers/actor/dp_actor.py','verl/trainer/ppo/core_algos.py',
             'verl/utils/checkpoint/fsdp_checkpoint_manager.py','verl/utils/experimental/torch_functional.py'):
    assert sha(NEW/name)==sha(OLD/name)
DEST.mkdir(parents=True);(AUDIT/'source').mkdir(parents=True,exist_ok=True)
cpu=dict(os.environ);cpu.update(env);cpu['CUDA_VISIBLE_DEVICES']=''
builder='import json,sys;sys.path.insert(0,sys.argv[1]);from owner_runtime_options import owner_command;print(json.dumps(owner_command(json.loads(sys.stdin.read()))))'
p['argv']=json.loads(subprocess.check_output([cpu['VENV_PYTHON'],'-c',builder,p['entry']],
    input=json.dumps(p['options']),text=True,env=cpu))
assert p['argv'][0]==cpu['VENV_PYTHON']
resolved=subprocess.run(p['argv']+['--cfg','job','--resolve'],env=cpu,cwd=p['entry'],capture_output=True)
(DEST/'effective-config.yaml').write_bytes(resolved.stdout);(DEST/'effective-config.stderr.log').write_bytes(resolved.stderr)
resolved.check_returncode()
p.update(version=VERSION,run=str(run),physical_gpus=[2,3],status='prepared_only_not_launched',source_commit=sys.argv[1],
    previous_prepared=dict(path=str(previous),SHA256=sha(previous)),
    execution_changes=['Independent trial version/output','Verified v2 native critic activation offload'],
    method_or_training_hyperparameter_changes=[],numerical_receipt=dict(path=str(numeric),SHA256=sha(numeric)),
    production_FA_deterministic=False,formal=False,existing_formal_training_mutations=0)
path=DEST/'prepared.json';path.write_text(json.dumps(p,indent=2)+'\n')
(AUDIT/'prepared-index.json').write_text(json.dumps(dict(version=VERSION,tasks=[dict(path=str(path),task='AppWorld')]),indent=2)+'\n')
prior=ROOT/'receipts/appworld-native-ppo-speed-20261016-v1'
shutil.copyfile(prior/'SQL-boundary-tests.xml',AUDIT/'SQL-boundary-tests.xml')
(AUDIT/'CPU-verification.json').write_text(json.dumps(dict(status='prepared_native_same_config_critic_memory_only',
    numerical_receipt=p['numerical_receipt'],source_bindings=len(p['source_bindings']),model_DT_optimizer_calls=0,
    method_or_training_hyperparameter_changes=[],performance_gate_pending=True),indent=2)+'\n')
for name in ('run_trials.py','submit_trials.py'):
    owner=ROOT/'receipts/appworld-sql-native-speed-20261016-v1/source'/name
    code=owner.read_text().replace('appworld-sql-native-speed-20261016-v1',VERSION)
    if name=='run_trials.py':
        anchor="index=read(AUDIT/'prepared-index.json')"
        assert code.count(anchor)==1
        gate="""performance=read(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/status.json')
assert performance['status']=='completed' and performance['exact_transport']=='passed'
assert all(item['median_ratio']<=1.0 for item in performance['comparisons']), 'No efficiency regression accepted'
"""
        code=code.replace(anchor,gate+anchor)
    (AUDIT/'source'/name).write_text(code)
assert authorities=={path:sha(path) for path in authorities}
print(json.dumps(dict(version=VERSION,status='CPU_prepared_only_performance_gate_pending',prepared=str(path),
    physical_gpus=[2,3],native_iterations=2,FA_deterministic=False,model_DT_optimizer_calls=0),indent=2))
