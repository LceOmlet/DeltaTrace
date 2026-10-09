set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'

import contextlib, hashlib, importlib, importlib.util, io, json, os, runpy, subprocess, sys, time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'); candidate=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-stable-memory-entry-20261010-v3'); base=candidate/'appworld'; entry=base/'entry'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def put(p,x):
    p.write_text(json.dumps(x,indent=2)+'\n');p.chmod(0o600)
original=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert sha(original)=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
s=json.loads(original.read_bytes());old_entry=Path(s['entry']);old_dt=Path(s['dt_root'])
dt=root/'candidates/direct-target-consumed-cache-release-20261007-v1/deltatrace'
config=dt.parent/'environment.json'
assert sha(config)=='cd28a6e2140190ffcba4d64229b5c62549b85473645fea249899014a034d6db2'
authorities={str(root/n):sha(root/n) for n in ('formal-training.json','active-training.json','active-source.json')}
assert not (base/'prepared.json').exists(),'Inspect the existing preparation instead of repeating it'
base.mkdir(parents=True,exist_ok=True);entry.mkdir()
for p in old_entry.iterdir():
    if p.name in ('launch_appworld_native.py','__pycache__'):continue
    (entry/p.name).symlink_to(p,target_is_directory=p.is_dir())
old_launcher=old_entry/'launch_appworld_native.py';assert sha(old_launcher)==s['entry_sha256'][old_launcher.name]
text=old_launcher.read_text();needle="numerical_runtime='c9cd147/fc2e6c2'";assert text.count(needle)==1
launcher=entry/old_launcher.name;launcher.write_text(text.replace(needle,'numerical_runtime='+repr('fla-early-output-scale-20261009-v1')))
env=dict(s['environment']);env.update(DT_ROOT=str(dt),DT_ENVIRONMENT_JSON=str(config),DT_ENTRY_ROOT=str(entry),CUDA_VISIBLE_DEVICES='4,5')
env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None)
def remap(p):
    p=p.replace(str(old_entry),str(entry))
    return p.replace(str(old_dt),str(dt))
env['PYTHONPATH']=remap(s['pythonpath'])
assert not any(k.startswith(('DT_HOLD','DT_STOP_AFTER','DT_ACTOR_INCIDENT')) for k in env)
output=root/'runs/appworld-stable-memory-20261010-v1/appworld-dt'
inspection=r"""
import contextlib,hashlib,importlib,io,json,os,runpy,torch
from pathlib import Path
from omegaconf import OmegaConf
from launch_appworld_native import options_for
base=Path(os.environ['INSPECTION_BASE']);old=json.loads(Path(os.environ['ORIGINAL_SOURCE']).read_bytes())
output=Path(os.environ['PREPARED_OUTPUT']);options,sampling=options_for(output)
def redirect(value):
    if isinstance(value,str):return value.replace(str(Path(os.environ['ORIGINAL_SOURCE']).parent),str(output)).replace(old['entry'],os.environ['DT_ENTRY_ROOT'])
    if isinstance(value,dict):return {k:redirect(v) for k,v in value.items()}
    if isinstance(value,list):return [redirect(v) for v in value]
    return value
assert options==redirect(old['startup_options']),'Original task or trainer options changed'
cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
for key,value in options.items():OmegaConf.update(cfg,key.lstrip('+'),value,force_add=True)
(base/'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
buffer=io.StringIO()
with contextlib.redirect_stdout(buffer):runpy.run_path(os.environ['PROBE_PATH'])
probe=json.loads(buffer.getvalue());modules=probe['modules']
for name in ('reward_readout','counterfactual','dt_training_batch','owner_trajectory_batch',
             'executed_target_spans','native_prefix_leases','verl.trainer.ppo.ray_trainer',
             'verl.workers.actor.dp_actor','verl.trainer.ppo.core_algos',
             'verl.workers.sharding_manager.fsdp_vllm','loop_owner_rollout','official_parser'):
    m=importlib.import_module('phi_agents.utils.appworld' if name=='official_parser' else name);p=Path(m.__file__)
    modules[name]=dict(path=str(p),resolved=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    if name in old['actual_CPU_imports']:assert modules[name]['sha256']==old['actual_CPU_imports'][name]['sha256'],name
    if name.startswith('verl.'):
        relative=str(p.relative_to(Path(old['verl_root'])))
        assert modules[name]['sha256']==old['verl_sha256'][relative],name
dataset=Path(options['data.custom_cls.path'])
assert hashlib.sha256(dataset.read_bytes()).hexdigest()==old['entry_sha256'][dataset.name]
modules['loop_iteration_dataset']=dict(path=str(dataset),resolved=str(dataset.resolve()),sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
assert cfg.trainer.resume_mode=='disable' and cfg.trainer.total_training_steps==200
assert cfg.actor_rollout_ref.model.lora_rank==8 and cfg.actor_rollout_ref.model.lora_alpha==16
assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu==4
assert cfg.actor_rollout_ref.actor.entropy_coeff==0.001 and cfg.actor_rollout_ref.actor.clip_ratio_c==3
assert not torch.cuda.is_initialized()
result=dict(options=options,sampling=sampling,actual_CPU_imports=modules,probe=probe,
    original_task_and_training_options_equal_except_entry_and_output_paths=True,CUDA_initialized=False,
    model_DT_optimizer_episode_calls=0)
(base/'CPU-imports.json').write_text(json.dumps(result,indent=2)+'\n')
"""
runenv={**os.environ,**env,'CUDA_VISIBLE_DEVICES':'-1','INSPECTION_BASE':str(base),
    'ORIGINAL_SOURCE':str(original),'PREPARED_OUTPUT':str(output),'PROBE_PATH':str(candidate/'probe_appworld_memory_stable_imports.py')}
run=subprocess.run([env['VENV_PYTHON'],'-c',inspection],cwd=entry,env=runenv,capture_output=True,timeout=60)
(base/'CPU-imports.stdout').write_bytes(run.stdout);(base/'CPU-imports.stderr').write_bytes(run.stderr)
if run.returncode:print(run.stderr.decode(errors='replace')[-4000:]);run.check_returncode()
cpu=json.loads((base/'CPU-imports.json').read_bytes())
bindings={item['path']:item['sha256'] for item in cpu['actual_CPU_imports'].values()}
bindings.update({str(original):sha(original),str(launcher):sha(launcher),str(config):sha(config)})
source=dict(environment=env,entry=str(entry),verl_root=s['verl_root'],dt_root=str(dt),pythonpath=env['PYTHONPATH'],
    startup_options=cpu['options'],local_patch_commit='57ad451f611c0577ff17775f198fdddf7a2caa82',source_bindings=bindings,
    checkpoint_restore_requested=False,resume_mode='disable',prepared_only=True,fresh_base_model=True,
    numerical_runtime='fla-early-output-scale-20261009-v1',numerical_source_commit='26bef6c8b2e49db118f46e3c05e86944dcf8e293',
    memory_source_commit='799224868e0a9c0f8031b6012bb71505ab801a35',base_source=binding(original))
put(base/'source-template.json',source)
oldjob=next(j for j in json.loads((root/'formal-training.json').read_bytes())['jobs'] if j['task']=='AppWorld')
plan=dict(task='AppWorld',devices=[4,5],environment=env,entry=str(entry),working_directory=str(output),
    argv=[env['VENV_PYTHON'],str(launcher),'--output',str(output)],budget=oldjob['budget'],
    checkpoint_restore_requested=False,resume_mode='disable')
put(base/'launch-plan.json',plan)
prep=dict(status='prepared_CPU_imports_passed_not_submitted',task='AppWorld',devices=[4,5],
    launch_plan=binding(base/'launch-plan.json'),source_template=binding(base/'source-template.json'),
    CPU_imports=binding(base/'CPU-imports.json'),repository_commit='57ad451f611c0577ff17775f198fdddf7a2caa82',
    numerical_runtime='fla-early-output-scale-20261009-v1',formal_started=False,source_changes='Original launcher provenance label only',
    numerical_or_training_implementation_changes=False,observed_unix=time.time())
put(base/'prepared.json',prep)
spec=importlib.util.spec_from_file_location('existing_submit_owner',candidate/'submit_prepared_direct_targets.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
checked=owner.check_prepared(candidate,'AppWorld')
assert authorities=={p:sha(p) for p in authorities},'Active training metadata changed'
result=dict(status='prepared_verified_selection_not_started',unix=time.time(),version='fla-early-output-scale-20261009-v1',
    candidate=str(candidate),prepared=binding(base/'prepared.json'),launch_plan=binding(base/'launch-plan.json'),
    source_template=binding(base/'source-template.json'),CPU_imports=binding(base/'CPU-imports.json'),
    effective_config=binding(base/'effective-config.yaml'),existing_checker=binding(candidate/'submit_prepared_direct_targets.py'),
    source_bindings_verified=checked['source_bindings_verified'],source_commit='57ad451f611c0577ff17775f198fdddf7a2caa82',
    original_task_and_training_options_equal_except_entry_and_output_paths=True,
    formal_started=False,root_authorities_unchanged=authorities,
    model_DT_optimizer_episode_calls=0,CPU=cpu)
put(base/'result.json',result);print(json.dumps(result))

PY
