"""Launch a bounded native PG localization with existing data, no training."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
out=entry.ROOT+'/receipts/textcraft-gradient-rootcause-20261013-v1'
sources=[HERE/'observe_native_PG_groups_v1.py',
    HERE.parent/'rl_upstream_alignment_20260929/observe_native_optimizer_minibatch.py',
    HERE.parent/'rl_upstream_alignment_20260929/observe_native_actor_loss_gradients.py']
subprocess.run(entry.SCP+[str(p) for p in sources]+[entry.SSH[-1]+':'+out+'/'],check=True)
script=rf'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path({entry.ROOT!r});out=Path({out!r});dest=out/'native-PG-groups-v1'
dest.mkdir(exist_ok=True)
assert not (dest/'job.json').exists(),'Do not duplicate an existing bounded diagnostic'
physical=subprocess.check_output(['mx-smi'],text=True)
(dest/'before-mx-smi.txt').write_text(physical)
assert not re.search(r'\|\s*[01]\s+\d+\s+',physical.split('| Process:')[-1]),'GPU0/1 occupied'
launch=json.loads((root/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']='0,1'
for name in ('RAY_ADDRESS','VERL_GLOBAL_TOKEN_MEAN','DTPO_BEHAVIOR_PROBE'):env.pop(name,None)
env['PYTHONPATH']=str(out)+os.pathsep+env.get('PYTHONPATH','')
actor=Path(env['VERL_ROOT'])/'verl/workers/actor/dp_actor.py'
sha=hashlib.sha256(actor.read_bytes()).hexdigest()
assert sha=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
with (dest/'stdout.log').open('wb') as stdout,(dest/'stderr.log').open('wb') as stderr:
 p=subprocess.Popen([env['VENV_PYTHON'],'-u',str(out/'observe_native_PG_groups_v1.py')],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
r={{'pid':p.pid,'birth':psutil.Process(p.pid).create_time(),'started_unix':time.time(),
 'physical_gpus':[0,1],'scope':'Existing iteration81 first global64, three disjoint PG token groups; native actor backward; zero optimizer/scheduler/DT/rollout/validation/GRPO calls',
 'evidence':'Existing four-minibatch output-score bounds place successful-target output L2 above the other two groups combined; actual Jacobian projection remains unknown.',
 'hypothesis':'Successful-target tokens also dominate the large saved parameter PG.',
 'competing_explanation':'Native parameter Jacobians amplify successful-source or zero-reward gradients enough to dominate despite smaller output-score norms.',
 'discriminating_quantity':'Native parameter-gradient Gram matrix of the three original disjoint PG token groups, including signs of their projections onto the sum.',
 'falsifier':'Source/zero-reward parameter gradients dominate or negate the successful-target projection.',
 'decision':'Locate the measured amplification in the appropriate original credit/PPO input path; do not choose a repair from output norms alone.',
 'not_a_temporal_repair_gate':True,'formal_training_changed':False,'owner_actor_SHA256':sha,'sources':{{}}}}
r['sources']={{p:hashlib.sha256((out/p).read_bytes()).hexdigest() for p in ['observe_native_PG_groups_v1.py','observe_native_optimizer_minibatch.py','observe_native_actor_loss_gradients.py']}}
(dest/'job.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=70)
(HERE/'launch-native-PG-groups-v1.stdout.txt').write_bytes(r.stdout)
(HERE/'launch-native-PG-groups-v1.stderr.txt').write_bytes(r.stderr)
if r.returncode:print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
d=json.loads(r.stdout)
(HERE/'native-PG-groups-v1-job.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(json.dumps(d))
