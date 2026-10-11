"""Transport the verified native-owner fix and delegate complete resume to VERL."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
RAW = HERE.parent/'direct-credit-records-20261009-v1'
SSH = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=12', '-oStrictHostKeyChecking=yes',
       '-oHostKeyAlias=[ssh.v5000-prod-gw.nhss.zhejianglab.com]:32036', '-p', '31146',
       'root@ssh.v5000-prod-gw.nhss.zhejianglab.com', 'bash', '-s']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    files = {name: base64.b64encode((REPO/'experiments/rl'/name).read_bytes()).decode()
             for name in ['metax_fsdp_event_wait.py', 'launch_textcraft_eventwait.py']}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    body = r'''
from pathlib import Path
import base64,hashlib,json,os,psutil,re,runpy,subprocess,time,urllib.request
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
previous=root/'runs/textcraft-formal-stable-20261009-v1'
out=root/'runs/textcraft-formal-eventwait-resume25-20261011-v1';entry=out/'entry'
diagnostic=root/'receipts/incident37-metax-eventwait-owner-fix-20261011-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
assert sha(previous/'source.json')=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
s=json.loads((previous/'source.json').read_text());checkpoint=previous/'textcraft-dt/checkpoints/global_step_25'
assert (checkpoint.parent/'latest_checkpointed_iteration.txt').read_text().strip()=='25'
check=json.loads((diagnostic/'checkpoint25-CPU.json').read_text())
assert len(check['files'])==7 and all(not f['nonfinite'] for f in check['files'])
assert next(f for f in check['files'] if f['path'].endswith('model_world_size_2_rank_1.pt'))['head']['row159048_SHA256']=='4defedf718a10ac463b43d89a6b0b0f9d5e98ec0a459eeab066f8673b125423a'
hook=json.loads((diagnostic/'production-hook-verification/verification.json').read_text())
assert hook['native_Ray_hook_result']['receipt']['patched_code_SHA256']=='1609439cddb3ffa0b30e18fa55ad6119c1527e009515335eafca5b24dcb3abfc'
for rank in [0,1]:
 r=json.loads((diagnostic/f'result/rank{rank}.json').read_text())
 assert r['phase']=='complete' and r['actual_optimizer_steps']==1
 assert len(r['microbatches'])==8 and all(m['gradient_finite'] for m in r['microbatches'])
for relative,expected in s['verl_sha256'].items():assert sha(Path(s['verl_root'])/relative)==expected,relative
for name,meta in s['actual_CPU_imports'].items():assert sha(meta['path'])==meta['sha256'],name
if (out/'deployment.json').exists():
 print(json.dumps(dict(already_deployed=True,receipt=json.loads((out/'deployment.json').read_text()))));raise SystemExit(0)
out.mkdir(parents=True,exist_ok=True);entry.mkdir(exist_ok=True)
for p in (previous/'entry').iterdir():
 if p.name in ['__pycache__','outputs']:continue
 target=entry/p.name
 if not target.exists():target.symlink_to(p,target_is_directory=p.is_dir())
newfiles=__FILES__
for name,encoded in newfiles.items():
 data=base64.b64decode(encoded);compile(data,str(entry/name),'exec');(entry/name).write_bytes(data)
assert sha(entry/'metax_fsdp_event_wait.py')==sha(diagnostic/'metax_fsdp_event_wait.py')
env=dict(os.environ,**s['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
for key in list(env):
 if key.startswith(('DT_ACTOR_INCIDENT','DT_HOLD','DT_STOP_AFTER')):env.pop(key)
env.update(CUDA_VISIBLE_DEVICES='2,3',DT_ENTRY_ROOT=str(entry),
 DT_FSDP_OWNER_LAUNCHER=str(previous/'entry/launch_textcraft_native.py'),
 DT_FSDP_EVENT_WAIT_RECEIPTS=str(out/'worker-runtime-overrides'),
 PYTHONPATH=str(entry)+':'+s['pythonpath'])
inspection=r"""
import os,runpy,json,hashlib,sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
import launch_textcraft_eventwait as wrapper
wrapper.configure_worker_setup()
native=runpy.run_path(str(wrapper.owner_launcher()),run_name='verified_native_launcher')
out=Path(os.environ['INSPECT_OUT']);checkpoint=Path(os.environ['INSPECT_CHECKPOINT'])
options,sampling=native['options_for'](Path(os.environ['INSPECT_DATA']),out/'textcraft-dt',resume_from=checkpoint)
config=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
for k,v in options.items():OmegaConf.update(config,k.lstrip('+'),v,force_add=True)
(out/'effective-config.yaml').write_text(OmegaConf.to_yaml(config))
assert not torch.cuda.is_initialized()
(out/'preflight.json').write_text(json.dumps(dict(options=options,sampling=sampling,CUDA_initialized=False,
 total_epochs=config.trainer.total_epochs,total_training_steps=config.trainer.total_training_steps,
 resume_mode=config.trainer.resume_mode,resume_from_path=config.trainer.resume_from_path),indent=2)+'\n')
"""
checked=subprocess.run([env['VENV_PYTHON'],'-c',inspection],cwd=entry,
 env=dict(env,CUDA_VISIBLE_DEVICES='-1',INSPECT_OUT=str(out),INSPECT_CHECKPOINT=str(checkpoint),
 INSPECT_DATA=str(root/'datasets/textcraft-native-20260930')),capture_output=True,timeout=120)
(out/'preflight.stdout').write_bytes(checked.stdout);(out/'preflight.stderr').write_bytes(checked.stderr)
if checked.returncode:print(checked.stderr.decode(errors='replace')[-3000:]);checked.check_returncode()
pre=json.loads((out/'preflight.json').read_text());options=pre['options']
allowed={'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir',
 'trainer.resume_mode','trainer.resume_from_path','+ray_init.runtime_env.worker_process_setup_hook',
 '+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS'}
assert {k:v for k,v in options.items() if k not in allowed}=={k:v for k,v in s['startup_options'].items() if k not in allowed}
assert pre['total_epochs']==30 and pre['total_training_steps'] is None and pre['resume_mode']=='resume_path'
assert options['actor_rollout_ref.model.lora_rank']==8 and options['actor_rollout_ref.model.lora_alpha']==16
assert options['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']==4
argv=[env['VENV_PYTHON'],str(entry/'launch_textcraft_eventwait.py'),'--data',str(root/'datasets/textcraft-native-20260930'),
 '--output',str(out/'textcraft-dt'),'--resume-from',str(checkpoint)]
meta=dict(prepared_unix=time.time(),source_commit=__COMMIT__,upstream_commit=s['upstream_commit'],
 previous_source=str(previous/'source.json'),previous_source_sha256=sha(previous/'source.json'),
 resume_from=str(checkpoint),resume_iteration=25,model_optimizer_scheduler_rng_data_owned_by='original VERL',
 checkpoint_CPU_receipt=str(diagnostic/'checkpoint25-CPU.json'),checkpoint_CPU_receipt_sha256=sha(diagnostic/'checkpoint25-CPU.json'),
 original_native_owner_SHA256=s['verl_sha256'],numerical_runtime=s['numerical_runtime'],
 options=options,sampling=pre['sampling'],effective_config_sha256=sha(out/'effective-config.yaml'),
 worker_setup_hook='metax_fsdp_event_wait.worker_setup',
 runtime_override=dict(owner='FSDPParamGroup.finalize_backward',lookup='torch.accelerator -> torch.cuda',
 owner_file_sha256='2798a87b2631091df738afaff6dfcead38ce0cc421585b4331ecaaefb4e784f5',
 patched_code_sha256='1609439cddb3ffa0b30e18fa55ad6119c1527e009515335eafca5b24dcb3abfc'),
 entry_files={n:dict(path=str(entry/n),sha256=sha(entry/n)) for n in newfiles},
 checkpoint_restore_requested=True,restoration_observed=False,formal_budget=330,
 diagnostic_stop=False,training_timeout=None,per_card_microbatch=4,total_microbatch=8,lora_rank=8,lora_alpha=16)
put(out/'prepared.json',meta)
if not __DEPLOY__:
 print(json.dumps(dict(prepared=True,receipt=meta)));raise SystemExit(0)
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\|\s*[23]\s+\d+\s+\S',physical,re.M),'Physical2/3 occupied'
with urllib.request.urlopen('http://127.0.0.1:36005/docs',timeout=5) as response:assert response.status==200
(out/'before-physical.txt').write_text(physical)
with (out/'train.log').open('xb') as stream:
 p=subprocess.Popen(argv,cwd=entry,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
meta.update(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),devices=[2,3],argv=argv,
 log=str(out/'train.log'),status='submitted_native_resume_initializing')
put(out/'deployment.json',meta)
source=dict(s,unix=time.time(),environment={k:env[k] for k in set(s['environment'])|{'DT_FSDP_OWNER_LAUNCHER','DT_FSDP_EVENT_WAIT_RECEIPTS'}},
 entry=str(entry),pythonpath=env['PYTHONPATH'],startup_options=options,checkpoint_restore_requested=True,
 resume_mode='resume_path',resume_from_path=str(checkpoint),fresh_base_model=False,local_patch_commit=__COMMIT__,
 runtime_override=meta['runtime_override'],deployment=str(out/'deployment.json'),prepared_only=False)
source['entry_sha256']=dict(s['entry_sha256'],**{n:sha(entry/n) for n in newfiles});put(out/'source.json',source)
manifest=json.loads((root/'formal-training.json').read_text());jobs=[j for j in manifest['jobs'] if j['task']!='TextCraft']
jobs.append(dict(task='TextCraft',method='dt',pid=p.pid,observed_process_created_unix=meta['birth'],started_unix=meta['started_unix'],
 devices=[2,3],argv=argv,entry=str(entry),verl_root=s['verl_root'],dt_root=s['dt_root'],output=str(out/'textcraft-dt'),
 log=meta['log'],checkpoints=str(out/'textcraft-dt/checkpoints'),source_receipt=str(out/'source.json'),
 numerical_runtime=s['numerical_runtime'],status=meta['status'],actor_microbatch=4,lora_rank=8,lora_alpha=16,
 resume_from=str(checkpoint),runtime_override=meta['runtime_override'],budget=dict(epochs=30,total_training_steps=None,
 prompt_groups_per_batch=32,samples_per_prompt=8,global_optimizer_minibatch=64,ppo_epochs=1)))
manifest.update(jobs=jobs,unix=time.time(),manifest=str(out/'formal-training.json'))
put(out/'formal-training.json',manifest);put(root/'formal-training.json',manifest);put(root/'active-training.json',manifest)
active=json.loads((root/'active-source.json').read_text());active.update(manifest=str(out/'formal-training.json'),unix=time.time(),
 jobs=[dict(task=j['task'],pid=j['pid'],birth=j.get('observed_process_created_unix'),entry=j['entry'],verl_root=j['verl_root'],
 source_receipt=j['source_receipt'],status=j['status'],numerical_runtime=j.get('numerical_runtime'),runtime_override=j.get('runtime_override')) for j in jobs])
put(root/'active-source.json',active)
print(json.dumps(dict(deployed=True,receipt=meta,source_sha256=sha(out/'source.json'))))
'''
    for key, value in dict(__FILES__=repr(files), __COMMIT__=repr(commit), __DEPLOY__=repr(args.deploy)).items():
        body = body.replace(key, value)
    ast.parse(body)
    command = ('source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh\n'
               'CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n' + body + '\nPY\n')
    label = 'deploy' if args.deploy else 'prepare'
    (RAW/f'native-eventwait-formal-{label}-20261011-v1.command.txt').write_text(command, encoding='utf-8')
    result = subprocess.run(SSH, input=command.encode(), capture_output=True, timeout=150)
    (RAW/f'native-eventwait-formal-{label}-20261011-v1.stdout.txt').write_bytes(result.stdout)
    (RAW/f'native-eventwait-formal-{label}-20261011-v1.stderr.txt').write_bytes(result.stderr)
    if result.returncode:
        print(result.stdout.decode(errors='replace')[-3000:]);print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    receipt = json.loads(result.stdout)
    (RAW/f'native-eventwait-formal-{label}-20261011-v1.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt['receipt'].items() if k in ['pid','birth','devices','status','resume_from','source_commit']},indent=2))


if __name__ == '__main__':
    main()
