"""Freeze the checked native async seam for the next original checkpoint.

Preparation only: no engine, job stop, checkpoint load or deployment. The
existing checkpoint observer and original VERL loader own the later resume.
"""
import hashlib
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote


def main():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    name = 'appworld-native-async-resume-20261002'
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,hashlib,json,os,psutil,runpy,shutil,sys,time
root=Path('@ROOT@');read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
active=read(root/'active-training.json')
job=next(j for j in active['jobs'] if j['task']=='AppWorld')
driver=psutil.Process(job['pid'])
assert abs(driver.create_time()-job['observed_process_created_unix'])<.02
source=read(Path(job['source_receipt']))
prior_path=Path(source['prepared_receipt']);prior=read(prior_path)
cpu=root/'candidates/appworld-native-async-015-transport-20261002'
cpu_prepared=read(cpu/'prepared.json')
assert cpu_prepared['source_entry']==job['entry']
assert cpu_prepared['source_owner']==job['verl_root']
probe_paths=['probe-interface/interfaces.json','probe-transport/transport.json',
             'probe-lifecycle/lora-delivery.json']
probes={name:read(cpu/name) for name in probe_paths}
assert all(p['passed'] for p in probes.values())
owner=Path(cpu_prepared['owner'])
for name,record in cpu_prepared['sources'].items():
    assert sha(owner/name)==record['after_sha256'],name
    assert sha(Path(job['verl_root'])/name)==record['before_sha256'],name
base=root/'candidates/@NAME@';entry=base/'entry'
assert not base.exists(),'Preserve any existing preparation instead of replacing it'
entry.parent.mkdir()
shutil.copytree(cpu_prepared['entry'],entry,ignore=shutil.ignore_patterns('__pycache__'))
launcher=entry/'launch_appworld_native.py'
before=launcher.read_text()
anchor="        'algorithm.adv_estimator': 'deltatrace',\n"
assert before.count(anchor)==1
after=before.replace(anchor,anchor+"        'actor_rollout_ref.rollout.mode': 'async',\n")
ast.parse(after)
(base/'launch_appworld_native.before-async.py').write_bytes(launcher.read_bytes())
launcher.write_text(after,newline='\n')
os.environ.update(LOOP_ROOT=prior['loop_root'],VERL_ROOT=str(owner),DT_ROOT=prior['dt_root'],
    DT_ENTRY_ROOT=str(entry),APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'))
sys.path[:0]=[str(entry),str(owner),prior['loop_root']]
# Compose the actual two launchers against the same author recipe and paths.
# No main routine, model or inference engine is executed.
old=runpy.run_path(str(Path(job['entry'])/'launch_appworld_native.py'))
new=runpy.run_path(str(launcher))
output=Path('/same-formal-output');checkpoint=Path('/same-completed-checkpoint')
old_options,old_sampling=old['options_for'](output,resume_from=checkpoint)
new_options,new_sampling=new['options_for'](output,resume_from=checkpoint)
assert old_sampling==new_sampling
from omegaconf import OmegaConf
inherited=OmegaConf.load(Path(job['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
old_options.setdefault('actor_rollout_ref.rollout.mode',inherited.actor_rollout_ref.rollout.mode)
new_options['data.custom_cls.path']=old_options['data.custom_cls.path']
changes={k:dict(before=old_options.get(k),after=new_options.get(k))
    for k in old_options.keys()|new_options.keys() if old_options.get(k)!=new_options.get(k)}
assert changes=={'actor_rollout_ref.rollout.mode':{'before':'sync','after':'async'}},changes
assert new_options['actor_rollout_ref.model.lora_rank']==8
assert new_options['actor_rollout_ref.model.lora_alpha']==16
assert new_options['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']==4
verification=dict(observed_unix=time.time(),scope='CPU API/transport/LoRA interface and actual author configuration composition; no new model numerics or performance claim',
    receipts={name:dict(path=str(cpu/name),sha256=sha(cpu/name)) for name in probe_paths},
    configuration_changes=changes,sampling_unchanged=old_sampling,passed=True)
verification_path=base/'cpu-verification.json'
verification_path.write_text(json.dumps(verification,indent=2)+'\n')
owner_files=dict(prior.get('owner_sha256',prior['owner_head_sha256']))
owner_files.update({name:record['after_sha256'] for name,record in cpu_prepared['sources'].items()})
for name,h in owner_files.items():assert sha(owner/name)==h,name
for name,h in prior['dt_source_sha256'].items():assert sha(Path(prior['dt_root'])/name)==h,name
prepared=dict(prior,prepared_unix=time.time(),status='prepared-only, explicit native async resume candidate; GPU runtime unaccepted',
    scope='Native VERL async server/manager and vLLM request scheduling; LOOP owns task/runner/termination and original VERL owns training/checkpoint behavior',
    prior_driver_pid=job['pid'],prior_entry=job['entry'],prior_verl_root=job['verl_root'],
    entry=str(entry),verl_root=str(owner),future_checkpoint_root=job['checkpoints'],
    entry_sha256={p.name:sha(p) for p in entry.glob('*.py')},owner_sha256=owner_files,owner_head_sha256=owner_files,
    prior_prepared_receipt=str(prior_path),prior_prepared_receipt_sha256=sha(prior_path),
    preparation_repository_commit='@REVISION@',preparation_script_sha256='@SCRIPT_SHA@',
    completion_transport_code_commit='e3230713478a10434f30dbd06607e01df6a8d7d1',
    completion_transport_receipt=str(verification_path),completion_transport_receipt_sha256=sha(verification_path),
    completion_transport_sources={name:sha(entry/name) for name in ['loop_owner_rollout.py','loop_async_transport.py','owner_rollout_scope.py']},
    native_async_source_preparation=dict(path=str(cpu/'prepared.json'),sha256=sha(cpu/'prepared.json')),
    configuration_changes=changes,prior_budget=job['budget'],minimum_completed_checkpoint=8)
(base/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
print(json.dumps(dict(prepared=str(base/'prepared.json'),pid_unchanged=driver.pid,
    configuration_changes=changes,status=prepared['status']),indent=2))
PY
'''.replace('@ENTRY@', ENTRY).replace('@ROOT@', ROOT).replace('@NAME@', name)
       .replace('@REVISION@', revision).replace('@SCRIPT_SHA@', script_sha))
    local = AUDIT/'appworld-official-async-20261002/resume-preparation'
    local.mkdir(exist_ok=True)
    for name_in in ['prepared.json','cpu-verification.json','entry/launch_appworld_native.py']:
        destination=local/Path(name_in).name
        subprocess.run(SCP+[SSH[-1]+':'+ROOT+'/candidates/'+name+'/'+name_in,str(destination)],check=True)


if __name__ == '__main__':
    main()
