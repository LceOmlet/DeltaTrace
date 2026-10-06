set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time

root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
receipt_dir=root/'receipts/appworld-efficiency-20261007/row-prefix-owner-fresh-v1'
script_path=receipt_dir/'submit-fresh.sh'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs_path=receipt_dir/'inputs.json'
assert sha(inputs_path)=='7559b0905df5b59804cf151529252143eb655dca0ed71036e16c968071b93480'
inputs=json.loads(inputs_path.read_bytes())
assert sha(receipt_dir/'prepare_cpu.py')=='4b69366c397021219557f2a3fd3aea143eda2f6774c620b5b345da194ca8909e'
prepared_path=receipt_dir/'prepared.json'
prepared=json.loads(prepared_path.read_bytes())
assert prepared['status']=='prepared_only_not_stopped_not_submitted'
assert prepared['inputs_sha256']==sha(inputs_path)
for key in ['source_template','run_environment','options']:
 assert sha(prepared[key]['path'])==prepared[key]['sha256'],key
prior_path=Path(inputs['formal_source']['path'])
assert sha(prior_path)==inputs['formal_source']['sha256']
prior=json.loads(prior_path.read_bytes())
source=json.loads(Path(prepared['source_template']['path']).read_bytes())
env=json.loads(Path(prepared['run_environment']['path']).read_bytes())
assert source['resume_mode']=='disable' and not source['checkpoint_restore_requested']
assert source['fresh_base_model']==env['MODEL_PATH']==prior['fresh_base_model']
assert source['lora_rank']==8 and source['lora_alpha']==16
assert env['CUDA_VISIBLE_DEVICES']=='4,5' and 'MACA_VISIBLE_DEVICES' not in env
for path,expected in inputs['baseline_source_bindings'].items():assert sha(path)==expected,path
for path,expected in inputs['selected_source_bindings'].items():assert sha(path)==expected,path
inventories=[]
for label,location,key in [('entry',env['DT_ENTRY_ROOT'],'entry_sha256'),('VERL',env['VERL_ROOT'],'verl_sha256'),('LOOP',env['LOOP_ROOT'],'author_sha256'),('DT',env['DT_ROOT'],'dt_source_sha256')]:
 for name,digest in source[key].items():assert sha(Path(location)/name)==digest,(label,name)
 inventories.append(dict(owner=label,root=location,verified_files=len(source[key]),mismatches=[]))
for item in [source['candidate_environment'],source['canonical_HF_owner'],*source['installed_verified_files']]:assert sha(item['path'])==item['sha256'],item['path']
for key in ['production_prepared','production_cpu_imports','combined_capacity_result','provisioning_environment']:
 item=inputs[key];assert sha(item['path'])==item['sha256'],key
lock=json.loads((root/'receipts/environment-only-20260930/entry/verified_runtime.json').read_bytes())
for item in lock['native_fla_files'].values():assert sha(item['path'])==item['expected'],item['path']
active=json.loads((root/'active-training.json').read_bytes())
old=next(j for j in active['jobs'] if j['task']=='AppWorld')
assert old['pid']==1953903 and old['observed_process_created_unix']==1791321793.72
assert old['source_receipt']==str(prior_path) and old['devices']==[4,5]
assert old['actor_microbatch']==4 and old['budget']==inputs['budget']
try:
 p=psutil.Process(old['pid'])
 assert p.status()==psutil.STATUS_ZOMBIE or p.create_time()!=old['observed_process_created_unix'],'Previous AppWorld driver remains active'
except psutil.NoSuchProcess:pass
for p in psutil.process_iter(['pid','cmdline','status']):
 if p.info['status']==psutil.STATUS_ZOMBIE:continue
 assert not any(str(old['output']) in arg for arg in (p.info['cmdline'] or [])),('Previous AppWorld command remains active',p.info['pid'])
physical=subprocess.check_output(['mx-smi'],text=True)
for d in [4,5]:assert not re.search(r'^\|\s+'+str(d)+r'\s+\d+\s+',physical.split('| Process:')[-1],re.M),'Requested physical GPU remains occupied'
output=Path(inputs['new_output']);base=output.parent
assert not base.exists(),'Inspect the existing submission instead of duplicating it'
launcher=Path(env['DT_ENTRY_ROOT'])/'launch_appworld_native.py'
assert sha(launcher)==inputs['launcher_sha256']
options=json.loads(Path(prepared['options']['path']).read_bytes())['options']
for key,value in inputs['fixed_options'].items():assert options[key]==value,key
assert 'trainer.resume_from_path' not in options
argv=[env['VENV_PYTHON'],str(launcher),'--output',str(output)]
assert '--resume-from' not in argv
base.mkdir();output.mkdir()
for name in ['active-training.json','active-source.json']:(base/('prior-'+name)).write_bytes((root/name).read_bytes())
source.update(unix=time.time(),submission_script_sha256=sha(script_path),
 submission_script_scope='Fresh fixed AppWorld submission through unchanged original launcher and configuration; only recorded DT owner/entry/numerical-environment and output identities change',
 formal_preparation=dict(path=str(prepared_path),sha256=sha(prepared_path)))
(output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
with (output/'train.log').open('wb') as log:
 child=subprocess.Popen(argv,env=env,cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(task='AppWorld',method='dt',pid=child.pid,observed_process_created_unix=psutil.Process(child.pid).create_time(),started_unix=time.time(),
 devices=[4,5],entry=env['DT_ENTRY_ROOT'],verl_root=env['VERL_ROOT'],loop_root=env['LOOP_ROOT'],dt_root=env['DT_ROOT'],argv=argv,
 output=str(output),log=str(output/'train.log'),checkpoints=str(output/'checkpoints'),source_receipt=str(output/'source.json'),
 status='fresh_formal_submitted_not_yet_verified',lora_rank=8,lora_alpha=16,actor_microbatch=4,log_prob_micro_batch_size_per_gpu=4,
 budget=old['budget'],checkpoint_restore_requested=False,resume_mode='disable')
(output/'job.json').write_text(json.dumps(job,indent=2)+'\n')
# Preserve all other task entries verbatim; this launcher only replaces AppWorld.
manifest=dict(active,manifest=str(base/'formal-training.json'),jobs=[j for j in active['jobs'] if j['task']!='AppWorld']+[job])
manifest.setdefault('retired_jobs',[]).append(dict(old,replacement_pid=child.pid,status='replaced_by_fresh_base_run_without_checkpoint'))
for path in [base/'formal-training.json',root/'active-training.json',root/'formal-training.json']:path.write_text(json.dumps(manifest,indent=2)+'\n')
sources=json.loads((root/'active-source.json').read_bytes());sources.update(unix=time.time(),manifest=manifest['manifest'])
sources['jobs']=[dict(task=j['task'],pid=j['pid'],entry=j['entry'],verl_root=j['verl_root'],source_receipt=j.get('source_receipt'),status=j['status'],runtime_override=None) for j in manifest['jobs']]
(root/'active-source.json').write_text(json.dumps(sources,indent=2)+'\n')
submission=dict(job=job,source_sha256=sha(output/'source.json'),prepared=dict(path=str(prepared_path),sha256=sha(prepared_path)),
 frozen_parent=inputs['formal_source'],production_owner=inputs['production_prepared'],CPU_imports=inputs['production_cpu_imports'],
 launcher=dict(path=str(launcher),sha256=sha(launcher)),submission_script=dict(path=str(script_path),sha256=sha(script_path)),
 owner_inventories=inventories,baseline_bindings_verified=1550,selected_bindings_verified=1550,
 launcher_option_changes=prepared['launcher_option_changes'],physical_before=physical,
 effective_environment={k:env[k] for k in ['VENV_PYTHON','MODEL_PATH','VERL_ROOT','DT_ROOT','DT_ENTRY_ROOT','LOOP_ROOT','DT_ENVIRONMENT_JSON','CUDA_VISIBLE_DEVICES','APPWORLD_ROOT','PYTHONPATH','VERL_TRIM_SHARED_PADDING','VERL_RELEASE_UNUSED_HOST_CACHE']},
 checkpoint_restore_requested=False,resume_mode='disable',
 scope='Fresh process submission only; no prepare, checkpoint operation, numerical correction, parameter change or profiler attach; model/runtime imports and training health need actual new-worker evidence')
(receipt_dir/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
print(json.dumps(submission),flush=True)
PY
