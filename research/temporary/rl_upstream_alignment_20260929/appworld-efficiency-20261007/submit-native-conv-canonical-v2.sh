set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');base=root/'runs/appworld-fresh-native-conv-canonical-20261007-v2';output=base/'appworld-dt'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prep_path=root/'receipts/appworld-efficiency-20261007/native-conv-canonical-owner-v2/prepared.json';prep=json.loads(prep_path.read_text())
assert sha(prep_path)=='f5232ac1dc7cc13a6142902488fdb28243f86dd7de91774bb91c3279fe5c0ec3'
# Root binds the existing completed 32k receipt; this is provenance only,
# not another numeric tolerance or a new capacity test. Never run unbound.
capacity_path=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/owner-b8-dispatch-20260930/native-prefix-reuse-native-conv-capacity-offset84-20261007-c1a079f-current-base/result.json')
capacity_sha256='ee782fd6e6a37f6b36c05792693a4935857a8b0e4d0f368aebfde52106df5214'
assert '@' not in str(capacity_path) and '@' not in capacity_sha256, 'Bind the existing capacity receipt before submission'
assert sha(capacity_path)==capacity_sha256
assert not base.exists(), 'Inspect the existing submission instead of duplicating it'
assert prep['cpu_validator_returncode']==0 and prep['checkpoint_resume_mode']=='disable'
active=json.loads((root/'active-training.json').read_text());old=next(j for j in active['jobs'] if j['task']=='AppWorld');prior=json.loads(Path(old['source_receipt']).read_text())
assert sha(old['source_receipt'])==prep['base_source']['sha256'], 'Prepared against this exact formal source'
assert Path(old['source_receipt'])==Path(prep['base_source']['path'])
assert prep['checkpoint_restore_requested'] is False
for item in [prep['candidate_environment'],prep['cpu_actual_imports'],prep['canonical_HF_owner']]:assert sha(item['path'])==item['sha256'],item['path']
assert sha(prep['isolated_owners']['receipt'])==prep['isolated_owners']['sha256']
isolated=prep['isolated_owners']['details']
for path,digest in isolated['candidate_sources'].items():assert sha(path)==digest,path
for item in [isolated['hf_candidate'],isolated['sitecustomize']]:assert sha(item['path'])==item['sha256'],item['path']
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
env=dict(os.environ);env.update(prior.get('resource_environment',{}));env.update(prep['resource_environment']);env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None);env.pop('RAY_TMPDIR',None);env.pop('DT_PREFIX_CHECKPOINT',None)
env.update(VERL_ROOT=prep['verl_root'],DT_ROOT=prep['dt_root'],DT_ENTRY_ROOT=prep['entry'],LOOP_ROOT=prep['loop_root'],
 DT_ENVIRONMENT_JSON=prep['candidate_environment']['path'],CUDA_VISIBLE_DEVICES='4,5',
 APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'))
env['PYTHONPATH']=prep['pythonpath']
for key in ['DT_PREFIX_NATIVE_CONV_INITIAL_STATES','DT_CONV_ISOLATED_IMPORT_ROOT']:env.pop(key,None)
argv=[env['VENV_PYTHON'],str(Path(prep['entry'])/'launch_appworld_native.py'),'--output',str(output)]
source=dict(prior,unix=time.time(),entry_sha256=prep['entry_sha256'],owner_head_sha256=prep['owner_sha256'],
 verl_root=prep['verl_root'],verl_sha256={n:sha(Path(prep['verl_root'])/n) for n in prior['verl_sha256']},
 dt_root=prep['dt_root'],loop_root=prep['loop_root'],pythonpath=env['PYTHONPATH'],prepared_receipt=str(prep_path),prepared_receipt_sha256=sha(prep_path),
 actor_padding_sha256=prep['actor_sha256'],trainer_sha256=prep['trainer_sha256'],
 padding_comparison_receipt=prep['official_padding_receipt'],padding_comparison_receipt_sha256=prep['official_padding_sha256'],
 fresh_base_model=env['MODEL_PATH'],checkpoint_restore_requested=False,resume_mode='disable',
 source_scope='Official masked_whiten once across collected action tokens; accepted original VERL padding comparison; cached-convolution owner initial_states option through the original canonical installed HF module; no eager namespace loader in task services; new run from base weights without checkpoint loading',
 native_conv_preparation=dict(path=str(prep_path),sha256=sha(prep_path)),
 isolated_owners=prep['isolated_owners'],candidate_environment=prep['candidate_environment'],canonical_HF_owner=prep['canonical_HF_owner'],
 native_conv_capacity_receipt=dict(path=str(capacity_path),sha256=capacity_sha256,scope='Existing completed capacity receipt bound by root; no new capacity/numeric criterion here'))
for key in ('resume_from','completed_checkpoint_marker','resume_launcher','unfinished_rollout_restart','initialization_retry'):source.pop(key,None)
# Keep the verified relative-key inventory; pathlib.rglob omits linked dirs.
source['dt_source_sha256']={name:sha(Path(prep['dt_root'])/name) for name in prior['dt_source_sha256']}
changed_dt={name for name,digest in source['dt_source_sha256'].items() if digest!=prior['dt_source_sha256'][name]}
expected_dt={str(Path(path).relative_to(Path(prep['dt_root']))):digest for path,digest in isolated['candidate_sources'].items()}
assert not changed_dt, changed_dt  # Same already tested DT bytes; only HF import route changes
for name,digest in expected_dt.items():assert source['dt_source_sha256'][name]==digest,name
source['resource_environment']=dict(prep['resource_environment'])
source['resource_environment'].update({k:env[k] for k in ['DT_ROOT','DT_ENVIRONMENT_JSON','VERL_TRIM_SHARED_PADDING','CUDA_VISIBLE_DEVICES']})
source.update(submission_repository_commit='ae35ecc',
 submission_script_sha256=sha(root/'receipts/appworld-efficiency-20261007/submit-native-conv-canonical-v2.sh'),
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
