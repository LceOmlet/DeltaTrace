"""Read-only snapshot of the deployed efficiency call path and existing DT logs."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('strict_entry', HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
code = r'''
import hashlib,json,re,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
run=root/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
launch=json.loads((run/'launch.json').read_bytes())
env=launch['effective_environment']; dt=Path(env['DT_ROOT']); verl=Path(env['VERL_ROOT'])
paths=[Path(env['DT_ENVIRONMENT_JSON'])]
paths += [verl/p for p in ['verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py','verl/trainer/ppo/ray_trainer.py','verl/utils/model.py','verl/utils/torch_functional.py','verl/models/transformers/monkey_patch.py','verl/models/transformers/qwen3_vl.py','verl/models/transformers/dense_common.py','verl/workers/critic/dp_critic.py']]
paths += [dt/p for p in ['clean/qwen35/qwen35_dense_finite_runner.py','clean/qwen35/qwen35_decoder_finite.py','clean/qwen35/qwen35_native_prefix_artifacts.py','accelerated/qwen35/qwen35_retained_capture.py','accelerated/qwen35/qwen35_code_local_capture.py']]
modules=['dt_training_batch.py','reward_readout.py','deltatrace_rollout.py','deltatrace_credit.py','native_prefix_leases.py','patch_verl_environment_entry.py','patch_actor_fused_head.py','patch_actor_entropy_dispatch.py','patch_actor_shared_right_padding.py']
resolved={}
for name in modules:
 for directory in env['PYTHONPATH'].split(':'):
  p=Path(directory)/name
  if p.is_file():
   paths.append(p);resolved[name]=str(p);break
files={}
for p in dict.fromkeys(paths):
 if p.is_file():
  b=p.read_bytes();files[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode('utf-8')}
logs={}
for session in Path('/tmp/ray').glob('session_*_'+str(launch['pid'])):
 for p in sorted((session/'logs').glob('worker-*')):
  if p.suffix not in ('.out','.err'):continue
  evidence=[]
  for n,line in enumerate(p.read_text(errors='replace').splitlines(),1):
   if any(s in line for s in ['[DT direct joint minibatch]','[DT direct target workload]','prefix_capture','capture_rounds','prefix_reuse','capture_and_preparation_seconds','prefix_lease']):
    evidence.append({'line':n,'text':line})
  if evidence:logs[str(p)]=evidence
overrides={}
for name in ['initial-state-and-existing-passive-records-v3.json','actual-native-input-variance-v3-verification.json']:
 p=run/'runtime-overrides'/name
 if p.is_file():overrides[str(p)]=json.loads(p.read_bytes())
print(json.dumps({'unix':time.time(),'resolved_by_recorded_pythonpath':resolved,'files':files,'existing_logs':logs,'overrides':overrides,'options':launch.get('hydra_overrides',launch.get('command',launch.get('argv')))},ensure_ascii=False))
'''
script = 'set -e\nsource '+entry.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
result=subprocess.run(entry.SSH+['bash','-s'],input=script.encode('utf-8'),capture_output=True)
if result.returncode:
 raise RuntimeError(result.stderr.decode('utf-8',errors='replace'))
result_stdout=result.stdout.decode('utf-8')
data=json.loads(result_stdout)
(HERE/'deployed-sources-and-existing-DT-logs.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for remote,item in data['files'].items():
 p=HERE/'sources'/Path(remote).relative_to(entry.ROOT)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(item['text'],encoding='utf-8',newline='\n')
print(json.dumps({'unix':data['unix'],'resolved_by_recorded_pythonpath':data['resolved_by_recorded_pythonpath'],'files':{p:v['sha256'] for p,v in data['files'].items()},'DT_log_rows':{p:len(v) for p,v in data['existing_logs'].items()},'local_receipt':str(HERE/'deployed-sources-and-existing-DT-logs.json')},ensure_ascii=False))
