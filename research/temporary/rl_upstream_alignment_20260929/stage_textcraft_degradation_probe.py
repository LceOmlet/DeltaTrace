"""Freeze a bounded original-owner diagnostic on the authorized free GPU4/5."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from stage_environment_entry import AUDIT, ROOT, SSH, SCP, ENTRY

OUT=ROOT+'/receipts/textcraft-degradation-probe-20261005-v1'
NAMES=['verify_textcraft_credit_degradation.py','observe_native_actor_loss_gradients.py']

SCRIPT=r'''/opt/conda/bin/python - <<'PY'
import ast,hashlib,json,os,pathlib,psutil,subprocess,time
root=pathlib.Path('__ROOT__');out=pathlib.Path('__OUT__')
assert not (out/'job.json').exists(), 'Inspect existing job before launching another diagnostic'
reference=root/'receipts/textcraft-degradation-20261005'
out.mkdir(exist_ok=True)
subprocess.run(['tar','-xf',str(out/'source.tar'),'-C',str(out)],check=True)
for name in ['checkpoint-matched-records.json','launch.json','source.json']:
 (out/name).write_bytes((reference/name).read_bytes())
for name,sha in __HASHES__.items():
 p=out/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;ast.parse(p.read_bytes())
source=json.loads((out/'source.json').read_bytes())
old_entry=root/'candidates/textcraft-rollout-scope-20261001/entry'
old_verl=pathlib.Path(source['verl_root']);old_dt=pathlib.Path(source['dt_root'])
checks={}
for name in ['deltatrace_rollout.py','reward_readout.py','counterfactual.py']:
 p=old_entry/name if name!='counterfactual.py' else old_dt/'experiments/rl'/name
 expected=(source['entry_sha256'][name] if name!='counterfactual.py' else '46937a43dbd7e89287eafe0e5b3344f0920d947c7a8bffc2cc50267de13cd754')
 actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==expected,(name,actual,expected)
 checks[str(p)]=actual
for name in ['verl/workers/actor/dp_actor.py','verl/trainer/ppo/core_algos.py','verl/utils/experimental/torch_functional.py']:
 p=old_verl/name;actual=hashlib.sha256(p.read_bytes()).hexdigest()
 expected=source['verl_sha256'][name];assert actual==expected,(name,actual,expected)
 checks[str(p)]=actual
checkpoint=root/'runs/official-trajectory-20260930-v7/textcraft-dt/checkpoints/global_step_25'
assert all((checkpoint/'actor'/f'model_world_size_2_rank_{i}.pt').is_file() for i in [0,1])
# Preserve the existing platform/dependency/cache environment; replace only the
# diagnostic import owners with the source-bound stopped TextCraft versions.
formal=psutil.Process(1856052);assert abs(formal.create_time()-1791197944.7)<.05
env=formal.environ()
env.pop('RAY_ADDRESS',None)
tail=env['PYTHONPATH']
tail=tail.replace(str(root/'candidates/appworld-eval-client-routing-resume-20261005-v1/entry'),str(old_entry))
tail=tail.replace(str(root/'candidates/appworld-native-prefix-resume-20261005-v2/verl'),str(old_verl))
tail=tail.replace(str(root/'candidates/appworld-native-prefix-resume-20261005-v2/deltatrace'),str(old_dt))
env.update(PYTHONPATH=str(out)+':'+tail,CUDA_VISIBLE_DEVICES='4,5',VERL_ROOT=str(old_verl),DT_ROOT=str(old_dt),
 DT_TASK='TextCraft',DT_MAX_STEPS='30',DT_MAX_LENGTH='32768',
 DT_TEXTCRAFT_PROBE_ROOT=str(out),DT_TEXTCRAFT_CHECKPOINT=str(checkpoint),DT_TEXTCRAFT_RECORD_LINES='[727,730]',
 DT_TEXTCRAFT_GRADIENT_DIAGNOSTIC='1')
launch=json.loads((out/'launch.json').read_bytes())
env['DT_SAMPLING_JSON']=json.dumps(dict(temperature=launch['options']['actor_rollout_ref.rollout.temperature'],
 max_tokens=launch['options']['data.max_response_length']))
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
(out/'physical-before-start.txt').write_text(physical)
# mx-smi's physical process rows identify real GPU users; do not rely on old
# device assignments or virtual allocator counters.
import re
occupied=[line for line in physical.splitlines() if re.match(r'^\|\s+[45]\s+\d+\s+',line)]
assert not occupied, occupied
available=psutil.virtual_memory().available
assert available>150*1024**3, 'Original two-rank loading plus active formal job must have room'
receipt=dict(role='Read-only actual TextCraft credit/gradient diagnostic; no formal training restart or optimizer step',
 checkpoint=str(checkpoint),original_step=26,devices=[4,5],sources=checks,
 diagnostic_sources={name:dict(path=str(out/name),sha256=sha) for name,sha in __HASHES__.items()},
 input_sources={name:dict(path=str(out/name),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in ['checkpoint-matched-records.json','launch.json','source.json']},
 reused_environment_pid=formal.pid,reused_environment_pid_birth=formal.create_time(),
 runtime_environment={k:env.get(k) for k in ['VERL_ROOT','DT_ROOT','DT_ENVIRONMENT_JSON','CUDA_VISIBLE_DEVICES','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','PYTHONPATH']},
 available_host_bytes=available,observed_unix=time.time(),optimizer_steps=0)
(out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (out/'probe.log').open('wb') as log:
 child=subprocess.Popen([env['VENV_PYTHON'],'-u',str(out/'verify_textcraft_credit_degradation.py')],cwd=out,env=env,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt.update(pid=child.pid,pid_birth=psutil.Process(child.pid).create_time(),started_unix=time.time(),log=str(out/'probe.log'))
(out/'job.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(out=str(out),pid=child.pid,pid_birth=receipt['pid_birth'],devices=[4,5])))
PY
'''

if __name__=='__main__':
    hashes={name:hashlib.sha256((AUDIT/name).read_bytes()).hexdigest() for name in NAMES}
    archive=AUDIT/'textcraft-degradation-20261005/diagnostic-source.tar'
    with tarfile.open(archive,'w') as tar:
        for name in NAMES:tar.add(AUDIT/name,arcname=name)
    subprocess.run(SSH+['mkdir','-p',OUT],check=True)
    subprocess.run(SCP+[str(archive),f'{SSH[-1]}:{OUT}/source.tar'],check=True)
    script=SCRIPT.replace('__ROOT__',ROOT).replace('__OUT__',OUT).replace('__HASHES__',repr(hashes))
    result=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True,check=True)
    (AUDIT/'textcraft-degradation-20261005/probe-submission.json').write_bytes(result.stdout)
    print(result.stdout.decode(errors='replace'))
