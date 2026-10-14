"""Run one bounded native derivative-chain localization, never training."""
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
manifest=sys.argv[1] if len(sys.argv)>1 else None
tag='native-gradient-chain-v4' if manifest is None else (sys.argv[2] if len(sys.argv)>2 else 'matched-task-target-derivatives-v11')
assert tag in ('native-gradient-chain-v4','matched-task-target-derivatives-v11','matched-action-observation-target-derivatives-v20','matched-history-ratio-chain-v23','native-DT-group-gradient-components-v26')
remote_sources=out if manifest is None else out+'/'+tag
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
sources=[HERE/'observe_native_gradient_chain_v4.py',
    HERE.parent/'rl_upstream_alignment_20260929/observe_native_optimizer_minibatch.py',
    HERE.parent/'rl_upstream_alignment_20260929/observe_native_actor_loss_gradients.py']
subprocess.run(entry.SCP+[str(p) for p in sources]+[entry.SSH[-1]+':'+remote_sources+'/'],check=True)
script=rf'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path({entry.ROOT!r});out=Path({out!r});dest=out/{tag!r};source_dir=Path({remote_sources!r})
manifest={manifest!r}
case_description=None if manifest is None else json.loads(Path(manifest).read_bytes())
if case_description is not None:assert case_description['destination']==str(dest)
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
if manifest is not None:
 env['DT_NATIVE_DIAGNOSTIC_MANIFEST']=manifest
 env['PYTHONPATH']=str(source_dir)+os.pathsep+env['PYTHONPATH']
actor=Path(env['VERL_ROOT'])/'verl/workers/actor/dp_actor.py'
assert hashlib.sha256(actor.read_bytes()).hexdigest()=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
assert env['VENV_PYTHON']=='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
with (dest/'stdout.log').open('wb') as stdout,(dest/'stderr.log').open('wb') as stderr:
 p=subprocess.Popen([env['VENV_PYTHON'],'-u',str(source_dir/'observe_native_gradient_chain_v4.py')],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
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
 'sources':{{name:hashlib.sha256((source_dir/name).read_bytes()).hexdigest() for name in ['observe_native_gradient_chain_v4.py','observe_native_optimizer_minibatch.py','observe_native_actor_loss_gradients.py']}}}}
if case_description is not None:
 common_coefficients=dest.name in ('matched-action-observation-target-derivatives-v20','matched-history-ratio-chain-v23')
 record.update(scope=case_description['scope'],question=case_description['question'],
  evidence=case_description['why_needed'],
  discriminating_quantity=('Native target PG norms at checkpoint5 for15 exact matched initial-state/action/observation factual pairs, with the same per-target loss coefficients; original global64/B4 interface,49 identical zero-derivative filler rows.' if common_coefficients else
   'Native global64 target PG norms at the same checkpoint5 on the16 exact matched task states, early versus late saved inputs and original coefficients.'),
  decision=('If amplification remains, differences between the two actual reasoning histories suffice for this finite comparison. If it disappears, the earlier68x result cannot be attributed to reasoning-history differences alone. Neither result explains why those histories were adopted or accepts a repair.' if common_coefficients else
   'If late-input amplification disappears under exact task matching, the older mixed-task contrast cannot support a general history mechanism. If it remains, task composition alone cannot explain it. Neither result is a fresh-training repair acceptance.'),
  limitations=case_description['limits'],case_manifest=manifest,
  case_manifest_SHA256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
  copied_training_function_bodies=0,new_optimizer_scheduler_DT_generation_validation_GRPO_calls=0,
  passive_boundary_readout_enabled=case_description.get('passive_boundary_readout_enabled',False))
 record.pop('competing_localizations',None)
 record.pop('falsifier',None)
 if case_description.get('native_ratio_chain_readout',False):
  record.update(
   discriminating_quantity='Native LM-head output/input and decoder-boundary adjoint norms, preclip parameter norms by layer, and the Gram matrix of actual B4 accumulation increments for the unchanged15 matched pairs/common coefficients.',
   decision='Account for the measured28.274x ratio along the actual backward chain, separating output-score magnitude, propagation and B4 directional accumulation. Do not investigate why histories shortened or infer a repair from one lower norm.',
   native_ratio_chain_readout=True)
 if 'diagnostic_advantage_views' in case_description:
  record.update(discriminating_quantity='Native parameter-gradient norms and cross inner products for the original full DT advantage, its same-positive-reward-group component and complementary groups, on the unchanged saved first global64/checkpoint80.',
   decision='Measure whether the coefficient concentration in groups that original GRPO zeros is also a concentration of the actual DT parameter gradient. This is a gradient-component accounting test, not a credit-quality or new-training experiment.',
   checkpoint=case_description['checkpoint'],diagnostic_advantage_views=case_description['diagnostic_advantage_views'])
(dest/'job.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=90)
(HERE/('launch-'+tag+'.stdout.txt')).write_bytes(r.stdout)
(HERE/('launch-'+tag+'.stderr.txt')).write_bytes(r.stderr)
if r.returncode:
    print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
data=json.loads(r.stdout)
(HERE/(tag+'-job.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:data[k] for k in ('pid','birth','physical_gpus','diagnostic_commit','sources')}))
