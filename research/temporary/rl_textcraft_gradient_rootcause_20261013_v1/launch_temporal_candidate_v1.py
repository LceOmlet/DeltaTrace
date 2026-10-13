"""Submit the native DT-only temporal regression; never resume the old run."""
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
files=[HERE/'install_temporal_capture_v2.py',HERE.parent/'rl_textcraft_restart_20261012/read_worker_state.py',
       HERE.parent/'rl_textcraft_restart_20261012/install_passive_records_native_graph.py']
subprocess.run(entry.SCP+[str(p) for p in files]+[entry.SSH[-1]+':'+out+'/'],check=True)
script=rf'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import copy,hashlib,json,os,psutil,re,shutil,subprocess,time
from pathlib import Path
root=Path({entry.ROOT!r});out=Path({out!r})
prepared=out/'formal-DT-global-token-mean-fresh-v1-prepared'
record=json.loads((prepared/'launch.json').read_bytes())
assert record['status']=='prepared_only_not_submitted_or_accepted'
assert 'behavior_verification_pending' not in record,'Rejected single-update gate must stay withdrawn'
run=root/'runs/textcraft-dt-global-token-mean-temporal-20261014-v1'
assert run.parent.resolve()==(root/'runs').resolve() and not run.exists(),'Existing run retained; do not duplicate'
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'\|\s*[23]\s+\d+\s+',physical.split('| Process:')[-1]),'Physical GPU2/3 occupied; not submitted'
expected=record['owner_SHA256'];candidate=Path(record['effective_environment']['VERL_ROOT'])
for name,sha in expected.items():assert hashlib.sha256((candidate/name).read_bytes()).hexdigest()==sha,name
baseline=root/'candidates/textcraft-checkpoint-rollout-rng-20261012-v1/verl-owner'
assert hashlib.sha256((baseline/'verl/workers/actor/dp_actor.py').read_bytes()).hexdigest()=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
assert json.loads((out/'global-token-mean-v3-capacity-retry-v4/result.json').read_bytes())['status']=='passed_B8_exact32768_graph_DT_HF_native_update'
old_run=record['run']
def remap(value):
 if isinstance(value,str):return value.replace(old_run,str(run))
 if isinstance(value,list):return [remap(v) for v in value]
 if isinstance(value,dict):return {{k:remap(v) for k,v in value.items()}}
 return value
launch=remap(record);launch['run']=str(run)
options=launch['options'];options['+ray_init._temp_dir']='/tmp/ray-tc-temporal-v1'
argv=launch['argv'];assert not any(a.startswith('+ray_init._temp_dir=') for a in argv)
argv.append('+ray_init._temp_dir=/tmp/ray-tc-temporal-v1')
fixed={{'actor_rollout_ref.model.lora_rank':8,'actor_rollout_ref.model.lora_alpha':16,
 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':4,'actor_rollout_ref.actor.ppo_mini_batch_size':64,
 'actor_rollout_ref.actor.ppo_epochs':1,'actor_rollout_ref.rollout.max_model_len':32768,
 'actor_rollout_ref.actor.ppo_max_token_len_per_gpu':32768,'env.rollout.n':8,'data.train_batch_size':32,
 'trainer.total_epochs':30,'trainer.save_freq':5,'trainer.total_training_steps':None,
 'trainer.resume_from_path':None,'trainer.val_before_train':False,'trainer.test_freq':-1,
 'trainer.max_actor_ckpt_to_keep':2,'actor_rollout_ref.rollout.enforce_eager':False,
 'actor_rollout_ref.rollout.free_cache_engine':False}}
assert {{k:options[k] for k in fixed}}==fixed
env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']='2,3'
for k in list(env):
 if k in ('RAY_ADDRESS','RAY_TMPDIR','MACA_VISIBLE_DEVICES','DTPO_BEHAVIOR_PROBE') or k.startswith(('DT_HOLD','DT_STOP_AFTER','DT_ACTOR_INCIDENT')):env.pop(k)
assert argv[0]==env['VENV_PYTHON'] and '-m' in argv and argv[argv.index('-m')+1]=='verl.trainer.main_ppo',argv[:5]
run.mkdir();(run/'before-mx-smi.txt').write_text(physical)
(run/'before-disk.json').write_text(json.dumps(dict(zip(('total','used','free'),shutil.disk_usage(root))),indent=2)+'\n')
source=remap(json.loads((prepared/'source.json').read_bytes()))
source['historical_submission_repository_commit']=source.get('submission_repository_commit')
source.update(submission_repository_commit={commit!r},prepared_only=False,
 deployment={{'status':'candidate_temporal_regression_not_complete_rootcause_acceptance'}},
 actual_import_scope='pending new-worker initial readback',passive_recording_status='pending new-worker RPC attach',
 startup_options=options,environment=launch['effective_environment'])
source['temporal_diagnostic_source']={{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'install_temporal_capture_v2.py',out/'read_worker_state.py',out/'install_passive_records_native_graph.py']}}
(run/'source.json').write_text(json.dumps(source,indent=2)+'\n')
launch.update(status='submitted_candidate_temporal_regression_not_accepted_repair',
 source=str(run/'source.json'),source_SHA256=hashlib.sha256((run/'source.json').read_bytes()).hexdigest(),
 diagnostic_commit={commit!r},normalization_patch_commit='56e69825fc45aa6cd5f3ef6a3bb48680bb22843d',
 hypothesis='Legacy B4 token denominators amplify updates as trajectories shorten and target density rises; correcting the denominator prevents the accelerating temporal anomaly.',
 competing_explanation='DTPO input/credit dynamics or native update cause the same accelerating anomaly even with correct global64 token normalization.',
 discriminating_observation='Actual cross-iteration native gradient trajectory and reward over the original anomaly phase, with real full256 batches and four optimizer-minibatch metrics.',
 falsifier='Persistent accelerating norm with reward deterioration under the corrected denominator falsifies denominator error as a sufficient root cause; do not deploy as a complete fix.',
 decision='Only temporal causal/regression evidence can accept the repair; early stability or one lower norm cannot. Captured input/state enable locating any remaining amplification.',
 original_budget_preserved=True,GRPO_rerun=False,old_DTPO_resumed=False,formal_manifests_changed=False)
with (run/'driver.log').open('wb') as stdout,(run/'driver.stderr.log').open('wb') as stderr:
 p=subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr,start_new_session=True,cwd=env['VERL_ROOT'])
launch.update(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time())
(run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
job={{k:launch[k] for k in ('pid','birth','started_unix','status','run','source_SHA256','diagnostic_commit','normalization_patch_commit','physical_gpus','hypothesis','competing_explanation','discriminating_observation','falsifier','decision','original_budget_preserved','GRPO_rerun','old_DTPO_resumed','formal_manifests_changed')}}
(out/'temporal-candidate-v1-job.json').write_text(json.dumps(job,indent=2)+'\n')
print(json.dumps(job))
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=70)
(HERE/'launch-temporal-candidate-v1.stdout.txt').write_bytes(r.stdout)
(HERE/'launch-temporal-candidate-v1.stderr.txt').write_bytes(r.stderr)
if r.returncode:print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
job=json.loads(r.stdout);(HERE/'temporal-candidate-v1-job.json').write_text(json.dumps(job,indent=2)+'\n',encoding='utf-8')
print(json.dumps(job))
