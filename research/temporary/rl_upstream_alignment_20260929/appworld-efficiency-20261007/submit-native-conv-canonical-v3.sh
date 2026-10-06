set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time

root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
base=root/'runs/appworld-fresh-native-conv-canonical-20261007-v3'
output=base/'appworld-dt'
receipt_dir=root/'receipts/appworld-efficiency-20261007/native-conv-canonical-owner-v3'
script_path=root/'receipts/appworld-efficiency-20261007/submit-native-conv-canonical-v3.sh'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior_path=root/'runs/appworld-fresh-native-conv-canonical-20261007-v2/appworld-dt/source.json'
prior_sha='3cd90b2db649cd477bc21398e7677dc8ea5534d5230296fc5f63c04037cd3427'
prep_path=root/'receipts/appworld-efficiency-20261007/native-conv-canonical-owner-v2/prepared.json'
prep_sha='f5232ac1dc7cc13a6142902488fdb28243f86dd7de91774bb91c3279fe5c0ec3'
assert sha(prior_path)==prior_sha, 'Frozen v2 source identity changed'
assert sha(prep_path)==prep_sha, 'Frozen preparation identity changed'
prior=json.loads(prior_path.read_text());prep=json.loads(prep_path.read_text())
active=json.loads((root/'active-training.json').read_text())
old=next(j for j in active['jobs'] if j['task']=='AppWorld')
assert old['source_receipt']==str(prior_path), 'Active AppWorld source changed'
assert not base.exists(), 'Inspect the existing submission instead of duplicating it'
assert prior['resume_mode']=='disable' and not prior['checkpoint_restore_requested']
assert prep['checkpoint_resume_mode']=='disable' and not prep['checkpoint_restore_requested']
assert prior['lora_rank']==8 and prior['lora_alpha']==16
assert old['actor_microbatch']==4 and old['devices']==[4,5]
assert prep['entry']==old['entry'] and prep['verl_root']==old['verl_root']
assert prep['dt_root']==prior['dt_root'] and prep['loop_root']==prior['loop_root']
assert prep['pythonpath']==prior['pythonpath']
inventories=[]
for label,location,key in [('entry',prep['entry'],'entry_sha256'),('VERL',prep['verl_root'],'verl_sha256'),('LOOP',prep['loop_root'],'author_sha256'),('DT',prep['dt_root'],'dt_source_sha256')]:
 hashes=prior[key]
 for name,digest in hashes.items():assert sha(Path(location)/name)==digest,(label,name)
 inventories.append({'owner':label,'root':location,'verified_files':len(hashes),'mismatches':[]})
for item in [prior['candidate_environment'],prior['canonical_HF_owner']]:assert sha(item['path'])==item['sha256'],item['path']
for item in prior['installed_verified_files']:assert sha(item['path'])==item['sha256'],item['path']
lock=json.loads((root/'receipts/environment-only-20260930/entry/verified_runtime.json').read_text())
for item in lock['native_fla_files'].values():assert sha(item['path'])==item['expected'],item['path']
launcher=Path(prep['entry'])/'launch_appworld_native.py'
launcher_sha='1223c007e2918a61858a41e836aad53bd199ea43cd45513c6860b50c05739532'
assert sha(launcher)==launcher_sha
try:
 p=psutil.Process(old['pid'])
 assert p.status()==psutil.STATUS_ZOMBIE or p.create_time()!=old['observed_process_created_unix'], 'Previous AppWorld driver remains active'
except psutil.NoSuchProcess:pass
for p in psutil.process_iter(['pid','cmdline','status']):
 if p.info['status']==psutil.STATUS_ZOMBIE:continue
 assert not any(str(old['output']) in arg for arg in (p.info['cmdline'] or [])), ('Previous AppWorld command remains active',p.info['pid'])
physical=subprocess.check_output(['mx-smi'],text=True)
for d in [4,5]:assert not re.search(r'^\|\s+'+str(d)+r'\s+\d+\s+',physical.split('| Process:')[-1],re.M), 'Requested physical GPU remains occupied'

# Reuse the recorded provisioning environment and the original frozen submit's
# resource overrides. Only output/run identity is new; owner code is untouched.
env=dict(os.environ);env.update(prior.get('resource_environment',{}));env.update(prep['resource_environment'])
for key in ['MACA_VISIBLE_DEVICES','RAY_ADDRESS','RAY_TMPDIR','DT_PREFIX_CHECKPOINT','DT_PREFIX_NATIVE_CONV_INITIAL_STATES','DT_CONV_ISOLATED_IMPORT_ROOT']:env.pop(key,None)
env.update(VERL_ROOT=prep['verl_root'],DT_ROOT=prep['dt_root'],DT_ENTRY_ROOT=prep['entry'],LOOP_ROOT=prep['loop_root'],
 DT_ENVIRONMENT_JSON=prep['candidate_environment']['path'],CUDA_VISIBLE_DEVICES='4,5',
 APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'))
env['PYTHONPATH']=prior['pythonpath']
assert env['MODEL_PATH']==prior['fresh_base_model']
argv=[env['VENV_PYTHON'],str(launcher),'--output',str(output)]
assert '--resume-from' not in argv
base.mkdir();output.mkdir();receipt_dir.mkdir(exist_ok=True)
for name in ['active-training.json','active-source.json']:(base/('prior-'+name)).write_bytes((root/name).read_bytes())
source=dict(prior,unix=time.time(),prior_driver_pid=old['pid'],prior_source_receipt=str(prior_path),prior_source_sha256=prior_sha,
 submission_repository_commit='4c859dfd29c1303729253417f4aa088bbf5552c6',submission_script_sha256=sha(script_path),
 submission_script_scope='Identity-only v3 fresh submission through unchanged frozen owner launcher; source3cd90b2d/preparedf523 remain code and configuration owners',
 checkpoint_restore_requested=False,resume_mode='disable')
for key in ['resume_from','completed_checkpoint_marker','resume_launcher','unfinished_rollout_restart','initialization_retry']:source.pop(key,None)
(output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
with (output/'train.log').open('wb') as log:
 child=subprocess.Popen(argv,env=env,cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
manifest=dict(active,manifest=str(base/'formal-training.json'),jobs=other+[job])
manifest.setdefault('retired_jobs',[]).append(dict(old,replacement_pid=child.pid,status='replaced_by_fresh_base_run_without_checkpoint'))
for path in [base/'formal-training.json',root/'active-training.json',root/'formal-training.json']:path.write_text(json.dumps(manifest,indent=2)+'\n')
sources=json.loads((root/'active-source.json').read_text());sources.update(unix=time.time(),manifest=manifest['manifest'])
sources['jobs']=[dict(task=j['task'],pid=j['pid'],entry=j['entry'],verl_root=j['verl_root'],source_receipt=j.get('source_receipt'),status=j['status'],runtime_override=None) for j in manifest['jobs']]
(root/'active-source.json').write_text(json.dumps(sources,indent=2)+'\n')
submission={'job':job,'source_sha256':sha(output/'source.json'),'frozen_parent':{'path':str(prior_path),'sha256':prior_sha},
 'prepared':{'path':str(prep_path),'sha256':prep_sha},'launcher':{'path':str(launcher),'sha256':launcher_sha},
 'provisioning_env':{'path':str(root/'receipts/environment-only-20260930/entry/metax-entry.env.sh'),'sha256':sha(root/'receipts/environment-only-20260930/entry/metax-entry.env.sh')},
 'submission_script':{'path':str(script_path),'sha256':sha(script_path)},'owner_inventories':inventories,
 'effective_environment':{k:env[k] for k in ['VENV_PYTHON','MODEL_PATH','VERL_ROOT','DT_ROOT','DT_ENTRY_ROOT','LOOP_ROOT','DT_ENVIRONMENT_JSON','CUDA_VISIBLE_DEVICES','APPWORLD_ROOT','PYTHONPATH','VERL_TRIM_SHARED_PADDING','VERL_RELEASE_UNUSED_HOST_CACHE']},
 'physical_before':physical,'checkpoint_restore_requested':False,'resume_mode':'disable',
 'scope':'Fresh formal process submission only; no prepare, parameter change, owner edit, storage/row-cut candidate selection, old checkpoint operation or profiler attach. Startup is not yet training health evidence.'}
(receipt_dir/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
print(json.dumps(submission),flush=True)
PY
