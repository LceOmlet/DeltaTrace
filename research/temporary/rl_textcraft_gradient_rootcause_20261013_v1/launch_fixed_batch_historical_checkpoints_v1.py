"""Launch one isolated, read-only-to-training gradient localization experiment."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
out=entry.ROOT+'/receipts/textcraft-gradient-rootcause-20261013-v1'
sources=[HERE/'observe_fixed_batch_historical_checkpoints_v1.py',
         HERE.parent/'rl_upstream_alignment_20260929/observe_native_optimizer_minibatch.py',
         HERE.parent/'rl_upstream_alignment_20260929/observe_native_actor_loss_gradients.py']
subprocess.run(entry.SCP+[str(p) for p in sources]+[entry.SSH[-1]+':'+out+'/'],check=True)
script=rf'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path({entry.ROOT!r});out=Path({out!r});dest=out/'fixed-batch-historical-checkpoints-v1'
dest.mkdir(exist_ok=True)
assert not (dest/'job.json').exists(),'Existing diagnostic retained; do not duplicate'
physical=subprocess.check_output(['mx-smi'],text=True)
(dest/'before-mx-smi.txt').write_text(physical)
assert not re.search(r'\|\s*[01]\s+\d+\s+',physical.split('| Process:')[-1]),'GPU0/1 occupied; no launch'
launch=json.loads((root/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']='0,1'
env.pop('RAY_ADDRESS',None);env.pop('VERL_GLOBAL_TOKEN_MEAN',None);env.pop('DTPO_BEHAVIOR_PROBE',None)
env['PYTHONPATH']=str(out)+os.pathsep+env.get('PYTHONPATH','')
baseline=Path(env['VERL_ROOT'])
actor_SHA=hashlib.sha256((baseline/'verl/workers/actor/dp_actor.py').read_bytes()).hexdigest()
assert actor_SHA=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
with (dest/'stdout.log').open('wb') as stdout,(dest/'stderr.log').open('wb') as stderr:
 p=subprocess.Popen([env['VENV_PYTHON'],'-u',str(out/'observe_fixed_batch_historical_checkpoints_v1.py')],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
r={{'pid':p.pid,'birth':psutil.Process(p.pid).create_time(),'started_unix':time.time(),'physical_gpus':[0,1],
 'scope':'Fixed original DT iteration81 token IDs/mask/advantages, native checkpoint5/40/80 gradient sensitivity at each recomputed old-policy point; no optimizer/scheduler/validation/GRPO/DT calls',
 'actor_SHA256':actor_SHA,'sources':{{}},
 'formal_training_changed':False,'candidate_deployed':False}}
r['sources']={{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['observe_fixed_batch_historical_checkpoints_v1.py','observe_native_optimizer_minibatch.py','observe_native_actor_loss_gradients.py']}}
(dest/'job.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
PY
'''
result=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=70)
(HERE/'launch-fixed-history-v1.stdout.txt').write_bytes(result.stdout)
(HERE/'launch-fixed-history-v1.stderr.txt').write_bytes(result.stderr)
if result.returncode:
    print(result.stderr.decode(errors='replace'));raise SystemExit(result.returncode)
data=json.loads(result.stdout)
(HERE/'fixed-history-v1-job.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(json.dumps(data))
