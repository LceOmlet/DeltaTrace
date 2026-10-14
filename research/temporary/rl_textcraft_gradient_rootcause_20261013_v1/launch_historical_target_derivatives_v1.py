"""Launch the bounded historical target derivative accounting; no training."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
out=entry.ROOT+'/receipts/textcraft-gradient-rootcause-20261013-v1'
sources=[HERE/'observe_historical_target_derivatives_v1.py',
    HERE.parent/'rl_upstream_alignment_20260929/observe_native_optimizer_minibatch.py',
    HERE.parent/'rl_upstream_alignment_20260929/observe_native_actor_loss_gradients.py']
subprocess.run(entry.SCP+[str(p) for p in sources]+[entry.SSH[-1]+':'+out+'/'],check=True)
script=rf'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path({entry.ROOT!r});out=Path({out!r});dest=out/'historical-target-derivatives-v1'
assert (dest/'carrier-receipt.json').is_file()
assert not (dest/'job.json').exists(),'Do not duplicate a bounded diagnostic'
assert not (dest/'result.json').exists()
physical=subprocess.check_output(['mx-smi'],text=True)
(dest/'before-mx-smi.txt').write_text(physical)
assert not re.search(r'\|\s*[01]\s+\d+\s+',physical.split('| Process:')[-1]),'GPU0/1 occupied'
grpo=psutil.Process(3174448)
assert abs(grpo.create_time()-1791793629.73)<.02
launch=json.loads((root/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']='0,1'
for name in ('RAY_ADDRESS','VERL_GLOBAL_TOKEN_MEAN','DTPO_BEHAVIOR_PROBE'):env.pop(name,None)
env['PYTHONPATH']=str(out)+os.pathsep+env.get('PYTHONPATH','')
actor=Path(env['VERL_ROOT'])/'verl/workers/actor/dp_actor.py'
sha=hashlib.sha256(actor.read_bytes()).hexdigest()
assert sha=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
assert env['VENV_PYTHON']=='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
with (dest/'stdout.log').open('wb') as stdout,(dest/'stderr.log').open('wb') as stderr:
 p=subprocess.Popen([env['VENV_PYTHON'],'-u',str(out/'observe_historical_target_derivatives_v1.py')],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
r={{'pid':p.pid,'birth':psutil.Process(p.pid).create_time(),'started_unix':time.time(),
 'physical_gpus':[0,1],'scope':'Original checkpoints5/40/80, exact positive IDs/masks/placement from iterations6/41/81; frozen entry-gradient of four global64s; two diagnostic target weighting views; zero optimizer/scheduler/DT/rollout/validation/GRPO calls.',
 'evidence':'Late native target PG dominates; same late input at checkpoint5/40/80 grows only1.35x. Actual B4 target coefficient-L2 bias1.94 early to2.61 late is insufficient alone for native norm14x growth. Late output force concentrates in low-probability action/format tokens within short successful B4s.',
 'hypothesis':'Temporal amplification of the already localized target parameter gradient is explained by the changing real target inputs and their B4-weighted output scores; B4 weights may amplify the late parameter directions more than their average coefficient ratio suggests.',
 'competing_explanation':'Target parameter gradients do not reproduce temporal growth, or common-token weighting retains the same late amplification; coefficient/output summaries then cannot justify the normalizer as sufficient root repair.',
 'discriminating_quantity':'At each original entry checkpoint, actual target parameter gradients and their B4-versus-common inner product/norm, together with native target output score, across all four frozen minibatches.',
 'falsifier':'Historical target parameter growth absent, or late/early corrected gradient growth comparable to legacy despite current coefficient accounting.',
 'decision':'Either retain or reject a quantitatively supported normalizer-dependent temporal mechanism; otherwise locate the remaining input/score/Jacobian term. Never accept repair merely because one norm is smaller.',
 'limits':'Failure counts for6/41 are approximate with independent aggregate/count calibration. Placeholder failure IDs have zero selected derivative. Printed alpha rounding retained. No original complete historical update is claimed. Entry-frozen minibatches do not emulate intervening optimizer updates.',
 'not_a_temporal_repair_gate':True,'formal_training_changed':False,'owner_actor_SHA256':sha,
 'protected_GRPO':{{'pid':grpo.pid,'birth':grpo.create_time()}},'sources':{{}}}}
r['sources']={{p:hashlib.sha256((out/p).read_bytes()).hexdigest() for p in ['observe_historical_target_derivatives_v1.py','observe_native_optimizer_minibatch.py','observe_native_actor_loss_gradients.py']}}
(dest/'job.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=70)
(HERE/'launch-historical-target-derivatives-v1.stdout.txt').write_bytes(r.stdout)
(HERE/'launch-historical-target-derivatives-v1.stderr.txt').write_bytes(r.stderr)
if r.returncode:print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
data=json.loads(r.stdout)
(HERE/'historical-target-derivatives-v1-job.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:data[k] for k in ['pid','birth','physical_gpus','sources','protected_GRPO']}))
