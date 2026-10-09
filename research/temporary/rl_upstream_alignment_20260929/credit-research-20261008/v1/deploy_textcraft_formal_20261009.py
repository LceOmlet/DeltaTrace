"""Deploy the accepted numerical release through the existing formal launcher.

No trainer, worker, model or credit implementation is provided here. The only
entry-file change corrects an obsolete provenance label. No diagnostic stop,
iteration limit, timeout, checkpoint restore or runtime observer is installed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
VERSION = 'fla-early-output-scale-20261009-v1'
OUT = transport.ROOT+'/runs/textcraft-formal-stable-20261009-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    repo = HERE.parents[4]
    verification_path = repo/'experiments/rl/results_fla_early_output_scale_deployment_20261009.json'
    verification = json.loads(verification_path.read_bytes())
    owner = next(x for x in verification['deployment']['owners'] if x['task']=='textcraft')
    numerical = {x['path']: x['sha256'] for x in verification['deployment']['unchanged_numerical_files']
                 if x['path'].startswith(owner['root']+'/')}
    numerical[owner['path']] = owner['new_sha256']
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote_code = r'''
import hashlib, importlib, json, os, re, shutil, subprocess, time, urllib.request
from pathlib import Path
import psutil
root=Path(__ROOT__); out=Path(__OUT__); version=__VERSION__
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2)+'\n'); tmp.replace(p)
base=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
assert sha(base)=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
s=json.loads(base.read_bytes()); old_entry=Path(s['environment']['DT_ENTRY_ROOT']); entry=out/'entry'
runtime_configuration=Path(s['environment']['DT_ENVIRONMENT_JSON'])
assert sha(runtime_configuration)==s['candidate_environment']['sha256']
if (out/'deployment.json').exists():
    previous=json.loads((out/'deployment.json').read_bytes())
    print(json.dumps({'already_deployed':True,'receipt':previous})); raise SystemExit(0)
out.mkdir(parents=True,exist_ok=True); entry.mkdir(exist_ok=True)
numeric=__NUMERIC__
for path,expected in numeric.items(): assert sha(path)==expected, path
expected_gdn=__GDN__
assert str(Path(expected_gdn['path']).resolve())==expected_gdn['versioned_path']
owner_files={}
for relative in ['verl/workers/actor/dp_actor.py','verl/trainer/ppo/core_algos.py',
                 'verl/workers/fsdp_workers.py','verl/trainer/main_ppo.py',
                 'verl/trainer/ppo/ray_trainer.py','verl/workers/sharding_manager/fsdp_vllm.py']:
    p=Path(s['verl_root'])/relative
    assert sha(p)==s['verl_sha256'][relative], str(p)
    owner_files[str(p)]=sha(p)
for name,meta in s['actual_CPU_imports'].items():
    expected=numeric.get(meta['path'],meta['sha256'])
    assert sha(meta['path'])==expected, meta['path']
for p in old_entry.iterdir():
    if p.name=='launch_textcraft_native.py' or p.name=='__pycache__': continue
    target=entry/p.name
    if not target.exists(): target.symlink_to(p, target_is_directory=p.is_dir())
old_launcher=old_entry/'launch_textcraft_native.py'
assert sha(old_launcher)==s['entry_sha256']['launch_textcraft_native.py']
old=old_launcher.read_text()
needle="numerical_runtime='c9cd147/fc2e6c2'"
assert old.count(needle)==1
launcher=entry/old_launcher.name
launcher.write_text(old.replace(needle,'numerical_runtime='+repr(version)))
env=dict(os.environ,**s['environment'])
env.pop('RAY_ADDRESS',None); env.pop('MACA_VISIBLE_DEVICES',None)
for key in list(env):
    if key.startswith('DT_ACTOR_INCIDENT') or key.startswith('DT_HOLD') or key.startswith('DT_STOP_AFTER'):
        env.pop(key)
env['CUDA_VISIBLE_DEVICES']='2,3'; env['DT_ENTRY_ROOT']=str(entry)
env['PYTHONPATH']=':'.join([str(entry),s['pythonpath']])
argv=[env['VENV_PYTHON'],str(launcher),'--data',str(root/'datasets/textcraft-native-20260930'),
      '--output',str(out/'textcraft-dt')]
inspection=r"""
import hashlib,importlib,json,os,sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from launch_textcraft_native import options_for
out=Path(os.environ['FORMAL_INSPECTION_OUT'])
options,sampling=options_for(Path(os.environ['FORMAL_DATA']),out/'textcraft-dt')
config=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
for key,value in options.items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
(out/'effective-config.yaml').write_text(OmegaConf.to_yaml(config))
dt=Path(os.environ['DT_ROOT']);sys.path.insert(0,str(dt/'clean/qwen35'));sys.path.insert(0,str(dt))
modules={}
for name in ['verl.trainer.main_ppo','verl.workers.actor.dp_actor','verl.trainer.ppo.core_algos',
             'deltatrace_rollout','reward_readout','counterfactual','dt_training_batch',
             'owner_trajectory_batch','qwen35_gdn_finite','vllm','transformers']:
    m=importlib.import_module(name);p=Path(m.__file__)
    modules[name]=dict(path=str(p),resolved=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert not torch.cuda.is_initialized()
record=dict(options=options,sampling=sampling,actual_CPU_imports=modules,CUDA_initialized=False,
            total_epochs=config.trainer.total_epochs,total_training_steps=config.trainer.total_training_steps,
            entropy_coeff=config.actor_rollout_ref.actor.entropy_coeff,
            clip_ratio_c=config.actor_rollout_ref.actor.clip_ratio_c,resume_mode=config.trainer.resume_mode)
(out/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
"""
check_env=dict(env,CUDA_VISIBLE_DEVICES='-1',FORMAL_INSPECTION_OUT=str(out),
               FORMAL_DATA=str(root/'datasets/textcraft-native-20260930'))
checked=subprocess.run([env['VENV_PYTHON'],'-c',inspection],cwd=entry,env=check_env,capture_output=True,timeout=120)
(out/'preflight.stdout').write_bytes(checked.stdout);(out/'preflight.stderr').write_bytes(checked.stderr)
if checked.returncode:print(checked.stderr.decode(errors='replace')[-3000:]);checked.check_returncode()
pre=json.loads((out/'preflight.json').read_bytes())
expected={k:v for k,v in s['startup_options'].items() if k not in [
    'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir']}
actual={k:v for k,v in pre['options'].items() if k not in [
    'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir']}
assert actual==expected,'Formal training options changed'
assert pre['actual_CPU_imports']['qwen35_gdn_finite']['sha256']==expected_gdn['new_sha256']
assert pre['total_epochs']==30 and pre['total_training_steps'] is None
assert pre['resume_mode']=='disable' and pre['entropy_coeff']==0.001 and pre['clip_ratio_c']==3
with urllib.request.urlopen('http://127.0.0.1:36005/docs',timeout=5) as r:assert r.status==200
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\|\s*[23]\s+\d+\s+\S',physical,re.M),'Physical2/3 occupied'
(out/'before-physical.txt').write_text(physical)
metadata=dict(version=version,numerical_source_commit=__NUMERIC_COMMIT__,deployment_source_commit=__COMMIT__,
    runtime_configuration_path=str(runtime_configuration),runtime_configuration_sha256=sha(runtime_configuration),
    base_source_path=str(base),base_source_sha256=sha(base),upstream_commit=s['upstream_commit'],
    numeric_files={p:dict(sha256=h,resolved=str(Path(p).resolve())) for p,h in numeric.items()},
    verified_deployment_receipt=__VERIFICATION__,native_owner_files=owner_files,
    entry=str(entry),entry_change='Provenance label only; original launcher/owner command unchanged.',
    launch_path=str(launcher),launch_sha256=sha(launcher),preflight_path=str(out/'preflight.json'),
    preflight_sha256=sha(out/'preflight.json'),effective_config_path=str(out/'effective-config.yaml'),
    effective_config_sha256=sha(out/'effective-config.yaml'),actual_CPU_imports=pre['actual_CPU_imports'],
    options=pre['options'],sampling=pre['sampling'],checkpoint_restore=False,
    diagnostic_stop=False,first_update_hold=False,training_timeout=None,
    per_card_microbatch=4,effective_two_card_microbatch=8,lora_rank=8,lora_alpha=16,
    host_available_bytes=psutil.virtual_memory().available,disk_free_bytes=shutil.disk_usage(out).free,
    prepared_unix=time.time())
put(out/'prepared.json',metadata)
if not __DEPLOY__:
    print(json.dumps({'prepared':True,'output':str(out),'version':version,'config_unchanged':True}));raise SystemExit(0)
assert not (out/'textcraft-dt/checkpoints/latest_checkpointed_iteration.txt').exists()
with (out/'train.log').open('xb') as stream:
    p=subprocess.Popen(argv,cwd=entry,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
metadata.update(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),
                devices=[2,3],argv=argv,log=str(out/'train.log'),formal_deployment=True,
                status='submitted_initializing_not_yet_update_verified')
put(out/'deployment.json',metadata)
recorded_env={k:env[k] for k in s['environment'] if k in env}
source=dict(s,unix=time.time(),environment=recorded_env,pythonpath=env['PYTHONPATH'],entry=str(entry),
    startup_options=pre['options'],actual_CPU_imports=pre['actual_CPU_imports'],prepared_only=False,
    numerical_runtime=version,accepted_numeric_files=metadata['numeric_files'],
    checkpoint_restore_requested=False,resume_mode='disable',fresh_base_model=True,
    local_patch_commit=__COMMIT__,deployment=str(out/'deployment.json'))
source['entry_sha256']=dict(s['entry_sha256'],launch_textcraft_native_py=sha(launcher))
source['entry_sha256']['launch_textcraft_native.py']=sha(launcher)
source['entry_sha256'].pop('launch_textcraft_native_py')
put(out/'source.json',source)
manifest_path=out/'formal-training.json'
old_manifest=json.loads((root/'formal-training.json').read_bytes())
jobs=[j for j in old_manifest['jobs'] if j['task']!='TextCraft']
for j in jobs:
    try:
        process=psutil.Process(j['pid']);expected_birth=j.get('observed_process_created_unix')
        alive=expected_birth is not None and abs(process.create_time()-expected_birth)<0.05
    except psutil.NoSuchProcess:alive=False
    j.update(observed_process_alive=alive,observed_unix=time.time())
    if not alive:j['status']='terminal_observed_not_restarted'
jobs.append(dict(task='TextCraft',method='dt',pid=p.pid,observed_process_created_unix=metadata['birth'],
    started_unix=metadata['started_unix'],devices=[2,3],argv=argv,entry=str(entry),
    verl_root=s['verl_root'],dt_root=s['dt_root'],output=str(out/'textcraft-dt'),log=metadata['log'],
    checkpoints=str(out/'textcraft-dt/checkpoints'),source_receipt=str(out/'source.json'),
    numerical_runtime=version,status=metadata['status'],actor_microbatch=4,lora_rank=8,lora_alpha=16,
    budget=dict(epochs=30,total_training_steps=None,prompt_groups_per_batch=32,samples_per_prompt=8,
                global_optimizer_minibatch=64,ppo_epochs=1),runtime_override=None))
manifest=dict(manifest=str(manifest_path),jobs=jobs,unix=time.time())
put(manifest_path,manifest);put(root/'formal-training.json',manifest);put(root/'active-training.json',manifest)
active_source=json.loads((root/'active-source.json').read_bytes())
active_source.update(manifest=str(manifest_path),unix=time.time(),jobs=[dict(
    task=j['task'],pid=j['pid'],birth=j.get('observed_process_created_unix'),entry=j['entry'],
    verl_root=j['verl_root'],source_receipt=j['source_receipt'],status=j['status'],
    numerical_runtime=j.get('numerical_runtime'),runtime_override=None) for j in jobs])
put(root/'active-source.json',active_source)
print(json.dumps({'deployed':True,'receipt':metadata,'source_sha256':sha(out/'source.json')}))
'''
    replacements = dict(__ROOT__=repr(transport.ROOT), __OUT__=repr(OUT), __VERSION__=repr(VERSION),
        __NUMERIC__=repr(numerical), __GDN__=repr(owner), __NUMERIC_COMMIT__=repr(verification['source_commit']),
        __COMMIT__=repr(commit), __DEPLOY__=repr(args.deploy),
        __VERIFICATION__=repr(dict(path=str(verification_path),sha256=hashlib.sha256(verification_path.read_bytes()).hexdigest())))
    for key,value in replacements.items(): remote_code=remote_code.replace(key,value)
    script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+remote_code+'\nPY\n'
    result=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=150)
    label='deploy' if args.deploy else 'prepare'
    (HERE/f'formal-stable-{label}-stdout-20261009.txt').write_bytes(result.stdout)
    (HERE/f'formal-stable-{label}-stderr-20261009.txt').write_bytes(result.stderr)
    if result.returncode:print(result.stdout.decode(errors='replace')[-4000:]);print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    receipt=json.loads(result.stdout)
    (HERE/f'formal-stable-{label}-20261009.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    if 'receipt' in receipt:
        r=receipt['receipt'];print(json.dumps({k:r[k] for k in ['version','pid','birth','devices','log','status']}))
    else:print(json.dumps(receipt))


if __name__=='__main__': main()
