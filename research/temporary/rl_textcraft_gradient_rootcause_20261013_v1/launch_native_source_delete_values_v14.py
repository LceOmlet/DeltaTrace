"""Stage one bounded native source-value diagnostic through strict SSH."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
out=entry.ROOT+'/receipts/textcraft-gradient-rootcause-20261013-v1'
tag=sys.argv[1] if len(sys.argv)>1 else 'native-source-delete-values-v14'
assert tag in ('native-source-delete-values-v14','native-source-delete-values-v15')
dest=out+'/'+tag
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
source=HERE/'observe_native_source_delete_values_v14.py'
subprocess.run(entry.SSH+['mkdir','-p',dest],check=True)
subprocess.run(entry.SCP+[str(source),entry.SSH[-1]+':'+dest+'/'],check=True)
script=f'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path({entry.ROOT!r});out=Path({out!r});dest=Path({dest!r})
assert not (dest/'job.json').exists() and not (dest/'result.json').exists(),'Do not duplicate a diagnostic.'
if dest.name=='native-source-delete-values-v15':
 previous=out/'native-source-delete-values-v14'
 failed=json.loads((previous/'result.json').read_bytes())
 assert failed['phase']=='failed' and 'native_target_logit_rows' in failed['error']
 old=json.loads((previous/'job.json').read_bytes())
 assert not psutil.pid_exists(old['pid']) or psutil.Process(old['pid']).create_time()!=old['birth']
 manifest=json.loads((previous/'query-manifest.json').read_bytes());manifest['destination']=str(dest)
 manifest['diagnostic_import_order_correction']='Original producer establishes model-interface paths before importing its native target reader; old v14 failed before any endpoint call.'
 (dest/'query-manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
manifest=json.loads((dest/'query-manifest.json').read_bytes())
assert manifest['distinct_source_points']==21
physical=subprocess.check_output(['mx-smi'],text=True)
(dest/'before-mx-smi.txt').write_text(physical)
assert not re.search(r'\|\s*[01]\s+\d+\s+',physical.split('| Process:')[-1]),'GPU0/1 occupied'
grpo=psutil.Process(3174448);assert grpo.create_time()==1791793629.73
ppo=psutil.Process(2671393);assert ppo.create_time()==1791949551.27
launch_path=root/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json'
launch=json.loads(launch_path.read_bytes())
env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']='0,1'
for key in ('RAY_ADDRESS','VERL_GLOBAL_TOKEN_MEAN','DTPO_BEHAVIOR_PROBE','DT_NATIVE_DIAGNOSTIC_MANIFEST'):env.pop(key,None)
env['PYTHONPATH']=str(dest)+os.pathsep+str(out)+os.pathsep+env.get('PYTHONPATH','')
env['DT_SOURCE_DELETE_DESTINATION']=str(dest)
assert env['VENV_PYTHON']=='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
actor=Path(env['VERL_ROOT'])/'verl/workers/actor/dp_actor.py'
assert hashlib.sha256(actor.read_bytes()).hexdigest()=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
with (dest/'stdout.log').open('wb') as stdout,(dest/'stderr.log').open('wb') as stderr:
 p=subprocess.Popen([env['VENV_PYTHON'],'-u',str(dest/'observe_native_source_delete_values_v14.py')],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),physical_gpus=[0,1],
 diagnostic_commit={commit!r},source_SHA256=hashlib.sha256((dest/'observe_native_source_delete_values_v14.py').read_bytes()).hexdigest(),
 query_manifest_SHA256=hashlib.sha256((dest/'query-manifest.json').read_bytes()).hexdigest(),
 question=manifest['question'],evidence=manifest['why_needed'],decision=manifest['decision'],
 scope='21 already observed iteration81 source points, original checkpoint80, identical pairs and single EOS pairs through original native HF/target owners. Two ranks are numerical repeats, not independent task samples.',
 original_launch=str(launch_path),original_launch_SHA256=hashlib.sha256(launch_path.read_bytes()).hexdigest(),
 copied_model_or_PPO_implementations=0,new_optimizer_scheduler_DT_rollout_validation_GRPO_calls=0,
 formal_training_changed=False,accepted_repair=None,deployed_repair=None,
 protected_GRPO=dict(pid=grpo.pid,birth=grpo.create_time()),protected_PPO=dict(pid=ppo.pid,birth=ppo.create_time()))
(dest/'job.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=90)
(HERE/('launch-'+tag+'.stdout.txt')).write_bytes(r.stdout)
(HERE/('launch-'+tag+'.stderr.txt')).write_bytes(r.stderr)
if r.returncode:
    print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
record=json.loads(r.stdout)
(HERE/(tag+'-job.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
