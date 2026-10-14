"""Run one bounded native derivative-chain localization, never training."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
out=entry.ROOT+'/receipts/textcraft-gradient-rootcause-20261013-v1'
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
sources=[HERE/'observe_native_gradient_chain_v4.py',
    HERE.parent/'rl_upstream_alignment_20260929/observe_native_optimizer_minibatch.py',
    HERE.parent/'rl_upstream_alignment_20260929/observe_native_actor_loss_gradients.py']
subprocess.run(entry.SCP+[str(p) for p in sources]+[entry.SSH[-1]+':'+out+'/'],check=True)
script=rf'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path({entry.ROOT!r});out=Path({out!r});dest=out/'native-gradient-chain-v4'
dest.mkdir(exist_ok=True)
assert not (dest/'job.json').exists() and not (dest/'result.json').exists(),'Do not duplicate the bounded localization'
assert json.loads((out/'historical-target-derivatives-v1/result.json').read_bytes())['phase']=='complete_historical_target_derivatives_not_repair'
physical=subprocess.check_output(['mx-smi'],text=True)
(dest/'before-mx-smi.txt').write_text(physical)
assert not re.search(r'\|\s*[01]\s+\d+\s+',physical.split('| Process:')[-1]),'GPU0/1 occupied'
grpo=psutil.Process(3174448);assert grpo.create_time()==1791793629.73
launch_path=root/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json'
launch=json.loads(launch_path.read_bytes())
env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']='0,1'
for name in ('RAY_ADDRESS','VERL_GLOBAL_TOKEN_MEAN','DTPO_BEHAVIOR_PROBE'):env.pop(name,None)
env['PYTHONPATH']=str(out)+os.pathsep+env.get('PYTHONPATH','')
actor=Path(env['VERL_ROOT'])/'verl/workers/actor/dp_actor.py'
assert hashlib.sha256(actor.read_bytes()).hexdigest()=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
assert env['VENV_PYTHON']=='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
with (dest/'stdout.log').open('wb') as stdout,(dest/'stderr.log').open('wb') as stderr:
 p=subprocess.Popen([env['VENV_PYTHON'],'-u',str(out/'observe_native_gradient_chain_v4.py')],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
record={{'pid':p.pid,'birth':psutil.Process(p.pid).create_time(),'started_unix':time.time(),
 'physical_gpus':[0,1],'scope':'Same original checkpoint5, saved iteration6/81 positive target carriers, first global64 each; two original PG backward passes with passive module/tensor hooks; no optimizer/scheduler/DT/generation/validation/GRPO calls.',
 'evidence':'Completed native target measurements show large late parameter/logit directional gains under both retrospective denominator views. Fixed late input already produces a large native PG at checkpoint5. This motivates locating amplification boundaries, not a fresh-training conclusion.',
 'question':'At which actual native model boundary does the late-input adjoint amplification appear, after retaining the old inputs and explicitly accounting for output score scale?',
 'competing_localizations':'Amplification already at head/final normalization, amplification within decoder propagation, or amplification only at parameter projections. Boundary observations determine the next owner/value inspection.',
 'discriminating_quantity':'Actual per-B4 input/output adjoint norms through32 decoder layers and final norm, with forward RMS at the positions carrying native adjoints.',
 'falsifier':'No corresponding amplification in the purported boundary; then inspect where it actually enters rather than attributing it to that module.',
 'decision':'Localize an evidenced native backward discrepancy or valid input-conditioned sensitivity before considering any owner change. Do not modify a mathematically valid operation just to reduce norms.',
 'counterfactual_scope':'This is an old-distribution diagnostic. It neither proves nor disproves that starting with correct denominators would avoid the anomaly; historical wrong-denominator feedback remains open.',
 'limitations':'Failure rows are existing zero-derivative placeholders; early denominators/alpha have recorded approximation. Hidden adjoints are rank-local B4 quantities. No formal full historical update or repair acceptance is claimed.',
 'diagnostic_commit':{commit!r},'formal_training_changed':False,'accepted_repair':None,'deployed_repair':None,
 'protected_GRPO':{{'pid':grpo.pid,'birth':grpo.create_time()}},
 'inherited_launch':str(launch_path),'inherited_launch_SHA256':hashlib.sha256(launch_path.read_bytes()).hexdigest(),
 'sources':{{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['observe_native_gradient_chain_v4.py','observe_native_optimizer_minibatch.py','observe_native_actor_loss_gradients.py']}}}}
(dest/'job.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=90)
(HERE/'launch-native-gradient-chain-v4.stdout.txt').write_bytes(r.stdout)
(HERE/'launch-native-gradient-chain-v4.stderr.txt').write_bytes(r.stderr)
if r.returncode:
    print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
data=json.loads(r.stdout)
(HERE/'native-gradient-chain-v4-job.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:data[k] for k in ('pid','birth','physical_gpus','diagnostic_commit','sources')}))
