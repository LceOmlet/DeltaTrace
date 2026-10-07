set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');prep_path=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/textcraft-fresh-row-prefix-20261007-v2/prepared.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert sha(prep_path)=='31afb0856c2fbaceed4dfeade435bd8df8c053cd8d8b76e0dd3d8c300079d3f4'
prep=read(prep_path)
assert prep['status']=='prepared_only_CPU_interface_passed_not_submitted'
for key in ('source_template','run_environment','launch_plan','interface'):
 item=prep[key];assert sha(item['path'])==item['sha256'],key
plan=read(prep['launch_plan']['path']);source=read(prep['source_template']['path'])
env=dict(os.environ,**plan['environment'])
for key in ('MACA_VISIBLE_DEVICES','RAY_ADDRESS','RAY_TMPDIR','DT_PREFIX_CHECKPOINT'):
 env.pop(key,None)
assert env['CUDA_VISIBLE_DEVICES']=='2,3'
assert plan['resume_mode']=='disable' and not plan['checkpoint_restore_requested']
assert '--resume-from' not in plan['argv']
options=plan['startup_options']
assert options['trainer.resume_mode']=='disable' and 'trainer.resume_from_path' not in options
for key,value in {'actor_rollout_ref.model.lora_rank':8,'actor_rollout_ref.model.lora_alpha':16,
 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':4}.items():assert options[key]==value
for location,mapping in [(prep['entry'],source['entry_sha256']),
 (prep['selected_VERL_root'],source['verl_sha256']),(prep['DT_root'],source['dt_source_sha256'])]:
 for name,digest in mapping.items():assert sha(Path(location)/name)==digest,(location,name)
active=read(root/'active-training.json');sources=read(root/'active-source.json')
app=next(j for j in active['jobs'] if j['task']=='AppWorld')
assert app['pid']==2360541 and app['observed_process_created_unix']==1791325655.01
assert app['devices']==[4,5]
assert psutil.Process(app['pid']).create_time()==app['observed_process_created_unix']
assert sha(app['source_receipt'])=='c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789'
old=next((j for j in active['jobs'] if j['task']=='TextCraft'),None)
if old and psutil.pid_exists(old['pid']):
 prior=psutil.Process(old['pid'])
 assert prior.create_time()!=old.get('observed_process_created_unix') or prior.status()==psutil.STATUS_ZOMBIE, 'Existing TextCraft still active; no stop performed'
physical=subprocess.check_output(['mx-smi'],text=True)
for d in [2,3]:
 assert not re.search(r'^\|\s+'+str(d)+r'\s+\d+\s+',physical.split('| Process:')[-1],re.M), 'Selected physical GPU is occupied'
service_path=prep_path.parent/'service-current.json'
assert sha(service_path)=='d7ebfac1cd8ce7b5af70da687c0bf8fdf0bbe00d13d9d20545f1376fc125031e'
service=read(service_path)
assert service['live'] and psutil.Process(service['pid']).create_time()==service['pid_birth']
assert psutil.Process(service['pid']).status()!=psutil.STATUS_ZOMBIE
output=Path(plan['working_directory']);base=output.parent
assert not base.exists(), 'Inspect the existing submission rather than duplicate it'
output.mkdir(parents=True)
for name in ('active-training.json','active-source.json','formal-training.json'):
 (base/('prior-'+name)).write_bytes((root/name).read_bytes())
(base/'physical-before.txt').write_text(physical)
source.update(unix=time.time(),prepared_only=False,prepared_receipt=str(prep_path),
 prepared_receipt_sha256=sha(prep_path),launch_plan_receipt=prep['launch_plan']['path'],
 launch_plan_sha256=prep['launch_plan']['sha256'],submission_repository_commit='14ba007d957d3b21ee3172993796b23474163da4',
 submission_script_sha256='58f7cbf7edd44dd92d97813eaefd894e11a0a37675d6cf0ca7c5b3175d8c82d9',resume_mode='disable',checkpoint_restore_requested=False,
 runtime_verification_status='submitted; actual worker imports and complete update not yet observed',
 scope=source['source_scope'],service_observation=service,
 current_service_receipt=dict(path=str(service_path),sha256=sha(service_path)))
for item in [dict(path=str(prep_path),sha256=sha(prep_path)),
             prep['launch_plan'],prep['interface'],dict(path=str(service_path),sha256=sha(service_path))]:
 source['source_bindings'][item['path']]=item['sha256']
(output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
with (output/'train.log').open('wb') as log:
 child=subprocess.Popen(plan['argv'],env=env,cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(task='TextCraft',method='dt',pid=child.pid,observed_process_created_unix=psutil.Process(child.pid).create_time(),
 started_unix=time.time(),devices=[2,3],entry=prep['entry'],verl_root=prep['selected_VERL_root'],dt_root=prep['DT_root'],
 argv=plan['argv'],output=str(output),log=str(output/'train.log'),checkpoints=str(output/'checkpoints'),
 source_receipt=str(output/'source.json'),status='fresh_formal_submitted_not_yet_verified',
 lora_rank=8,lora_alpha=16,actor_microbatch=4,log_prob_micro_batch_size_per_gpu=4,
 budget=plan['budget'],checkpoint_restore_requested=False,resume_mode='disable')
(output/'job.json').write_text(json.dumps(job,indent=2)+'\n')
jobs=[job if j['task']=='TextCraft' else j for j in active['jobs']]
if old is None:jobs.append(job)
manifest=dict(active,manifest=str(base/'formal-training.json'),jobs=jobs)
if old:manifest.setdefault('retired_jobs',[]).append(dict(old,replacement_pid=child.pid,status='replaced_by_fresh_base_run_without_checkpoint'))
assert next(j for j in manifest['jobs'] if j['task']=='AppWorld')==app
for path in (base/'formal-training.json',root/'active-training.json',root/'formal-training.json'):
 path.write_text(json.dumps(manifest,indent=2)+'\n')
item=dict(task=job['task'],pid=job['pid'],entry=job['entry'],verl_root=job['verl_root'],source_receipt=job['source_receipt'],
 status=job['status'],runtime_override=None,actor_microbatch=4,lora_rank=8,lora_alpha=16)
existing=any(j['task']=='TextCraft' for j in sources['jobs'])
sources['jobs']=[item if j['task']=='TextCraft' else j for j in sources['jobs']]
if not existing:sources['jobs'].append(item)
sources.update(unix=time.time(),manifest=manifest['manifest'])
(root/'active-source.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps(dict(job=job,source_sha256=sha(output/'source.json'),prepared_sha256=sha(prep_path),AppWorld_preserved=True)),flush=True)

PY
