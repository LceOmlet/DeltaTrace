set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');base=root/'runs/appworld-fresh-official-padding-20261007-v2';output=base/'appworld-dt'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prep_path=root/'receipts/appworld-efficiency-20261007/fresh-official-padding-v1/prepared.json';prep=json.loads(prep_path.read_text())
assert not base.exists(), 'Inspect the existing submission instead of duplicating it'
assert prep['cpu_validator_returncode']==0 and prep['checkpoint_resume_mode']=='disable'
active=json.loads((root/'active-training.json').read_text());old=next(j for j in active['jobs'] if j['task']=='AppWorld');prior=json.loads(Path(old['source_receipt']).read_text())
for location,hashes in [(prep['entry'],prep['entry_sha256']),(prep['verl_root'],prep['owner_sha256']),(prep['loop_root'],prior['author_sha256'])]:
 for name,digest in hashes.items():assert sha(Path(location)/name)==digest,(location,name)
lock=json.loads((root/'receipts/environment-only-20260930/entry/verified_runtime.json').read_text())
installed=[]
for item in lock['installed_restored_files']:
 actual=sha(item['path']);assert actual==item['expected_verified_sha256'],item['path']
 installed.append(dict(path=item['path'],sha256=actual,expected_verified_sha256=item['expected_verified_sha256']))
for item in lock['native_fla_files'].values():assert sha(item['path'])==item['expected'],item['path']
restoration=root/'receipts/appworld-efficiency-20261007/restore-verified-vllm-31146-v1/restored.json'
assert json.loads(restoration.read_text())['status']=='verified_bytes_restored_new_workers_required'
physical=subprocess.check_output(['mx-smi'],text=True)
for d in [4,5]:assert not re.search(r'^\|\s+'+str(d)+r'\s+\d+\s+',physical.split('| Process:')[-1],re.M)
try:
 p=psutil.Process(old['pid']);assert p.status()==psutil.STATUS_ZOMBIE or p.create_time()!=old['observed_process_created_unix'], 'Previous AppWorld driver remains active'
except psutil.NoSuchProcess:pass
base.mkdir();output.mkdir()
for name in ('active-training.json','active-source.json'):(base/('prior-'+name)).write_bytes((root/name).read_bytes())
env=dict(os.environ);env.update(prior.get('resource_environment',{}));env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None);env.pop('RAY_TMPDIR',None);env.pop('DT_PREFIX_CHECKPOINT',None)
env.update(VERL_ROOT=prep['verl_root'],DT_ROOT=prep['dt_root'],DT_ENTRY_ROOT=prep['entry'],LOOP_ROOT=prep['loop_root'],
 DT_ENVIRONMENT_JSON=str(Path(prep['dt_root'])/'environment.json'),CUDA_VISIBLE_DEVICES='4,5',
 APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'))
paths=prior['pythonpath'].split(':');paths=[prep['entry'] if q==old['entry'] else prep['verl_root'] if q==old['verl_root'] else q for q in paths]
env['PYTHONPATH']=':'.join(paths)
argv=[env['VENV_PYTHON'],str(Path(prep['entry'])/'launch_appworld_native.py'),'--output',str(output)]
source=dict(prior,unix=time.time(),entry_sha256=prep['entry_sha256'],owner_head_sha256=prep['owner_sha256'],
 verl_root=prep['verl_root'],verl_sha256={n:sha(Path(prep['verl_root'])/n) for n in prior['verl_sha256']},
 dt_root=prep['dt_root'],loop_root=prep['loop_root'],pythonpath=env['PYTHONPATH'],prepared_receipt=str(prep_path),prepared_receipt_sha256=sha(prep_path),
 actor_padding_sha256=prep['actor_sha256'],trainer_sha256=prep['trainer_sha256'],
 padding_comparison_receipt=prep['official_padding_receipt'],padding_comparison_receipt_sha256=prep['official_padding_sha256'],
 fresh_base_model=env['MODEL_PATH'],checkpoint_restore_requested=False,resume_mode='disable',
 source_scope='Official masked_whiten once across collected action tokens; accepted original VERL padding comparison; new run from base weights without checkpoint loading')
for key in ('resume_from','completed_checkpoint_marker','resume_launcher','unfinished_rollout_restart','initialization_retry'):source.pop(key,None)
source['dt_source_sha256']={str(p.relative_to(Path(prep['dt_root']))):sha(p) for p in Path(prep['dt_root']).rglob('*.py') if '.git' not in p.parts and '__pycache__' not in p.parts}
source['resource_environment'].update({k:env[k] for k in ['DT_ENVIRONMENT_JSON','VERL_TRIM_SHARED_PADDING','CUDA_VISIBLE_DEVICES']})
source.update(submission_repository_commit='c1a079f002ceaac5f13a6145bea3adf7600c35f0',
 submission_script_sha256=sha(root/'receipts/appworld-efficiency-20261007/submit-fresh-official-padding-v2.sh'),
 submission_script_scope='Exact staged submission script; upstream, prepared owner and imported file hashes identify runtime independently',
 installed_verified_files=installed,installed_restoration_receipt=str(restoration),installed_restoration_receipt_sha256=sha(restoration))
source.pop('provenance_correction',None)
(output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
with (output/'train.log').open('wb') as log:child=subprocess.Popen(argv,env=env,cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(task='AppWorld',method='dt',pid=child.pid,observed_process_created_unix=psutil.Process(child.pid).create_time(),started_unix=time.time(),
 devices=[4,5],entry=prep['entry'],verl_root=prep['verl_root'],loop_root=prep['loop_root'],dt_root=prep['dt_root'],argv=argv,
 output=str(output),log=str(output/'train.log'),checkpoints=str(output/'checkpoints'),source_receipt=str(output/'source.json'),
 status='fresh_formal_submitted_not_yet_verified',lora_rank=8,lora_alpha=16,actor_microbatch=4,log_prob_micro_batch_size_per_gpu=4,
 budget=old['budget'],checkpoint_restore_requested=False,resume_mode='disable')
(output/'job.json').write_text(json.dumps(job,indent=2)+'\n')
other=[]
for j in active['jobs']:
 if j['task']=='AppWorld':continue
 try:
  p=psutil.Process(j['pid']);same=p.status()!=psutil.STATUS_ZOMBIE and p.create_time()==j.get('observed_process_created_unix')
 except psutil.NoSuchProcess:same=False
 if not same:j=dict(j,status='terminal_observed_no_restart_requested',terminal_observed_unix=time.time())
 other.append(j)
manifest=dict(active,manifest=str(base/'formal-training.json'),jobs=other+[job]);manifest.setdefault('retired_jobs',[]).append(dict(old,replacement_pid=child.pid,status='replaced_by_fresh_base_run_without_checkpoint'))
for path in [base/'formal-training.json',root/'active-training.json',root/'formal-training.json']:path.write_text(json.dumps(manifest,indent=2)+'\n')
sources=json.loads((root/'active-source.json').read_text());sources.update(unix=time.time(),manifest=manifest['manifest'])
sources['jobs']=[dict(task=j['task'],pid=j['pid'],entry=j['entry'],verl_root=j['verl_root'],source_receipt=j.get('source_receipt'),status=j['status'],runtime_override=None) for j in manifest['jobs']]
(root/'active-source.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps(dict(job=job,source_sha256=sha(output/'source.json'),prepared_sha256=sha(prep_path),physical_before=physical)),flush=True)
PY
