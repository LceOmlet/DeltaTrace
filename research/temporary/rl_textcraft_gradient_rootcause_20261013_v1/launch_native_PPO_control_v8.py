"""Start the prepared native VERL PPO command on the two freed DTPO GPUs.

Only process launch, strict transport and provenance are implemented here.
PPO, GAE, critic, actor, losses, optimizer and checkpointing remain native.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import re

HERE=Path(__file__).resolve().parent
version=sys.argv[1] if len(sys.argv)>1 else 'v8'
assert re.fullmatch(r'v[0-9]+',version)
spec=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=HERE).strip()
out=entry.ROOT+'/receipts/textcraft-gradient-rootcause-20261013-v1'
script=rf'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path({entry.ROOT!r});out=Path({out!r})
prepared=out/{('native-PPO-control-'+version+'-prepared')!r}
record=json.loads((prepared/'launch.json').read_bytes())
source=json.loads((prepared/'source.json').read_bytes())
run=Path(record['run']);job=out/{('native-PPO-control-'+version+'-job.json')!r}
assert not run.exists() and not job.exists(),'No duplicate launch or overwrite'
assert record['method']=='ppo' and record['options']['algorithm.adv_estimator']=='gae'
for path,sha in record['source_bindings'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
physical=subprocess.check_output(['mx-smi'],text=True,timeout=30)
(prepared/'before-launch-mx-smi.txt').write_text(physical)
table=physical.split('| Process:')[-1]
assert not re.search(r'\|\s*[23]\s+\d+\s+',table),'GPU2/3 occupied; do not launch'
grpo=psutil.Process(3174448)
assert grpo.create_time()==1791793629.73,'Protected GRPO identity changed'
for pid,birth in [(3172194,1791793609.09),(2510247,1791907554.02)]:
 assert not psutil.pid_exists(pid) or psutil.Process(pid).create_time()!=birth,'Old DTPO/candidate unexpectedly alive'
env=dict(os.environ);env.update(record['effective_environment']);env['CUDA_VISIBLE_DEVICES']='2,3'
assert 'VERL_GLOBAL_TOKEN_MEAN' not in env
env.pop('RAY_ADDRESS',None);env.pop('RAY_TMPDIR',None)
run.mkdir(parents=True)
for name in ('launch.json','source.json','effective-config.yaml'):
 (run/name).write_bytes((prepared/name).read_bytes())
with (run/'driver.stdout.log').open('wb') as stdout,(run/'driver.stderr.log').open('wb') as stderr:
 process=subprocess.Popen(record['argv'],cwd=env['VERL_ROOT'],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
birth=psutil.Process(process.pid).create_time()
identity={{'pid':process.pid,'birth':birth,'started_unix':time.time(),'physical_gpus':[2,3],
 'run':str(run),'method':'ppo','status':'submitted_native_PPO_control_initialization_pending',
 'launcher_commit':{commit!r},'source_bindings':record['source_bindings'],
 'launch_SHA256':hashlib.sha256((run/'launch.json').read_bytes()).hexdigest(),
 'source_SHA256':hashlib.sha256((run/'source.json').read_bytes()).hexdigest(),
 'protected_GRPO':{{'pid':grpo.pid,'birth':grpo.create_time()}},
 'native_PPO_implementation':True,'GRPO_rerun':False,'old_DTPO_resumed':False,
 'hourly_automation':'PAUSED','rootcause_repair_verified':False}}
job.write_text(json.dumps(identity,indent=2)+'\n')
(run/'job.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity))
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=90)
(HERE/('launch-native-PPO-control-'+version+'.stdout.txt')).write_bytes(r.stdout)
(HERE/('launch-native-PPO-control-'+version+'.stderr.txt')).write_bytes(r.stderr)
if r.returncode:
 print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
result=json.loads(r.stdout)
(HERE/('native-PPO-control-'+version+'-job.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['pid','birth','physical_gpus','run','status','launcher_commit']},ensure_ascii=False))
