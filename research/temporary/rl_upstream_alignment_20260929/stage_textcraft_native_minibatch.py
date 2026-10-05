"""Freeze one real author-n8 minibatch for native, zero-step gradient observation.

This does not install an observer in the formal training source. The stopped
TextCraft source and the prepared terminal-reward repair remain separate.
"""
import argparse
import hashlib
import json
import subprocess
import tarfile

from stage_environment_entry import AUDIT, ROOT, SSH, SCP


OUT = ROOT + '/receipts/textcraft-native-minibatch-20261006-v4'
FILES = ['verify_textcraft_native_minibatch.py',
         'observe_native_optimizer_minibatch.py',
         'test_observe_native_optimizer_minibatch.py',
         'observe_native_actor_loss_gradients.py',
         'observe_textcraft_native_batches.py']

SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import ast,hashlib,json,os,pathlib,psutil,re,subprocess,time
root=pathlib.Path('__ROOT__'); out=pathlib.Path('__OUT__')
mode='__MODE__'; hashes=__HASHES__
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/'job.json').exists(), 'Inspect the existing diagnostic instead of starting a duplicate'
prepared=json.loads((root/'candidates/textcraft-truncated-terminal-resume-20261005-v2/prepared.json').read_bytes())
entry=pathlib.Path(prepared['entry']); verl=pathlib.Path(prepared['verl_root']); dt=pathlib.Path(prepared['dt_root'])
checks={}
for name in ['textcraft_owner_rollout.py','dt_training_batch.py','owner_trajectory_batch.py',
             'deltatrace_rollout.py','reward_readout.py','launch_textcraft_native.py']:
 p=entry/name; actual=sha(p); assert actual==prepared['entry_sha256'][name], (str(p),actual)
 checks[str(p)]=actual
for name,expected in [('verl/workers/actor/dp_actor.py','1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd')]:
 p=verl/name; actual=sha(p); assert actual==expected,(str(p),actual); checks[str(p)]=actual
p=dt/'clean/qwen35/qwen35_dense_finite_runner.py'
assert sha(p)=='c7fc969f9f521993f2449ea5f364adcb3e0fdac5b01c38c103963639551516c1'
checks[str(p)]=sha(p)
checkpoint=root/'runs/official-trajectory-20260930-v7/textcraft-dt/checkpoints/global_step_25'
assert all((checkpoint/'actor'/f'model_world_size_2_rank_{i}.pt').is_file() for i in (0,1))
formal=psutil.Process(1856052); assert abs(formal.create_time()-1791197944.7)<.05
env=formal.environ(); env.pop('RAY_ADDRESS',None)
tail=env['PYTHONPATH'].replace(str(root/'candidates/appworld-eval-client-routing-resume-20261005-v1/entry'),str(entry))
tail=tail.replace(str(root/'candidates/appworld-native-prefix-resume-20261005-v2/verl'),str(verl))
tail=tail.replace(str(root/'candidates/appworld-native-prefix-resume-20261005-v2/deltatrace'),str(dt))
env.update(PYTHONPATH=str(out)+':'+tail,VERL_ROOT=str(verl),DT_ROOT=str(dt),DT_TASK='TextCraft',
 DT_MAX_STEPS='30',DT_MAX_LENGTH='32768',DT_TEXTCRAFT_MINIBATCH_ROOT=str(out),
 DT_TEXTCRAFT_NATIVE_OBSERVATION_DIR=str(out/'native-output'),CUDA_VISIBLE_DEVICES='4,5')
for key in ['DT_TEXTCRAFT_GRADIENT_DIAGNOSTIC','DT_TEXTCRAFT_MATCHED_DIAGNOSTIC','DT_TEXTCRAFT_PROBE_ROOT']:
 env.pop(key,None)
if mode=='prepare':
 subprocess.run(['tar','-xf',str(out/'source.tar'),'-C',str(out)],check=True)
 for name,h in hashes.items():
  p=out/name; assert sha(p)==h; ast.parse(p.read_bytes())
 prior=root/'receipts/textcraft-degradation-20261005/launch.json'
 launch=json.loads(prior.read_bytes()); options=launch['options']
 for k,part in [('trainer.default_local_dir','checkpoints'),('trainer.rollout_data_dir','rollouts'),('trainer.validation_data_dir','validation')]:
  options[k]=str(out/part)
 options['trainer.resume_mode']='resume_path'; options['trainer.resume_from_path']=str(checkpoint)
 # The owning command builder preserves actual Hydra quoting and owner defaults.
 builder_script='import json; from owner_runtime_options import owner_command; x=json.load(open("'+str(out/'launch.json')+'")); print(json.dumps(owner_command(x["options"])))'
 (out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 argv=json.loads(subprocess.run([env['VENV_PYTHON'],'-c',builder_script],env=env,capture_output=True,text=True,check=True).stdout)
 launch['argv']=argv; (out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 cpuenv=dict(env,CUDA_VISIBLE_DEVICES='')
 with (out/'owner-inspection.stdout.txt').open('wb') as log:
  result=subprocess.run([env['VENV_PYTHON'],'-u',str(out/'verify_textcraft_native_minibatch.py'),
   '--inspect-only','--config-path',str(verl/'verl/trainer/config')]+launch['argv'][3:],cwd=out,env=cpuenv,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0, 'Inspect owner-inspection.stdout.txt; no GPU job submitted'
 with (out/'scalar-owner-tests.stdout.txt').open('wb') as log:
  result=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q',str(out/'test_observe_native_optimizer_minibatch.py'),
   '--junitxml='+str(out/'scalar-owner-tests.xml')],cwd=out,env=cpuenv,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0, 'Inspect scalar-owner-tests.stdout.txt; no GPU job submitted'
 with (out/'ray-owner-inspection.stdout.txt').open('wb') as log:
  result=subprocess.run([env['VENV_PYTHON'],'-u',str(out/'verify_textcraft_native_minibatch.py'),
   '--inspect-ray-only','--config-path',str(verl/'verl/trainer/config')]+launch['argv'][3:]+['+ray_init.num_gpus=0'],
   cwd=out,env=cpuenv,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0, 'Inspect ray-owner-inspection.stdout.txt; no GPU job submitted'
 receipt=dict(role='Prepared real 8 native prompts x author n8; one complete optimizer minibatch64; no optimizer or scheduler updates',
  source_kind='Prepared terminal-reward repair, not a formal training restart',checkpoint=str(checkpoint),devices=[4,5],
  sources=checks,diagnostic_sources={n:dict(path=str(out/n),sha256=h) for n,h in hashes.items()},
  reused_environment_pid=formal.pid,reused_environment_pid_birth=formal.create_time(),
  inspection=str(out/'native-minibatch-inspect.json'),inspection_sha256=sha(out/'native-minibatch-inspect.json'),
  launch_sha256=sha(out/'launch.json'),observed_unix=time.time(),optimizer_steps=0)
 (out/'prepared-diagnostic.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(status='prepared_cpu_owner_inspection_completed',out=str(out))))
else:
 receipt=json.loads((out/'prepared-diagnostic.json').read_bytes())
 for n,identity in receipt['diagnostic_sources'].items(): assert sha(pathlib.Path(identity['path']))==identity['sha256']
 for p,h in receipt['sources'].items(): assert sha(pathlib.Path(p))==h
 assert sha(out/'launch.json')==receipt['launch_sha256']
 launch=json.loads((out/'launch.json').read_bytes())
 env['DT_SAMPLING_JSON']=json.dumps(dict(temperature=launch['options']['actor_rollout_ref.rollout.temperature'],max_tokens=launch['options']['data.max_response_length']))
 physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
 (out/'physical-before-start.txt').write_text(physical)
 occupied=[line for line in physical.splitlines() if re.match(r'^\|\s+[45]\s+\d+\s+',line)]
 assert not occupied,occupied
 receipt['available_host_bytes']=psutil.virtual_memory().available
 receipt['physical_before_start']=str(out/'physical-before-start.txt')
 argv=[env['VENV_PYTHON'],'-u',str(out/'verify_textcraft_native_minibatch.py'),
       '--config-path',str(verl/'verl/trainer/config')]+launch['argv'][3:]
 with (out/'diagnostic.log').open('wb') as log:
  child=subprocess.Popen(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt.update(pid=child.pid,pid_birth=psutil.Process(child.pid).create_time(),started_unix=time.time(),argv=argv,log=str(out/'diagnostic.log'))
 (out/'job.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(status='zero_step_diagnostic_submitted',out=str(out),pid=child.pid,pid_birth=receipt['pid_birth'],devices=[4,5])))
PY
'''


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','launch'])
    args=p.parse_args()
    hashes={name:hashlib.sha256((AUDIT/name).read_bytes()).hexdigest() for name in FILES}
    if args.mode=='prepare':
        archive=AUDIT/'textcraft-degradation-20261005/native-minibatch-source-20261006-v4.tar'
        with tarfile.open(archive,'w') as tar:
            for name in FILES: tar.add(AUDIT/name,arcname=name)
        subprocess.run(SSH+['mkdir','-p',OUT],check=True)
        subprocess.run(SCP+[str(archive),f'{SSH[-1]}:{OUT}/source.tar'],check=True)
    script=SCRIPT.replace('__ROOT__',ROOT).replace('__OUT__',OUT).replace('__MODE__',args.mode).replace('__HASHES__',repr(hashes))
    result=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True)
    (AUDIT/'textcraft-degradation-20261005'/f'native-minibatch-{args.mode}-20261006.stdout.txt').write_bytes(result.stdout+result.stderr)
    print(result.stdout.decode(errors='replace')); print(result.stderr.decode(errors='replace'))
    result.check_returncode()
