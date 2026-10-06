set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,hashlib,json,os,psutil,sys,time

root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest_path=root/'active-training.json'
manifest=json.loads(manifest_path.read_bytes())
jobs=[j for j in manifest['jobs'] if j['task']=='AppWorld']
assert len(jobs)==1
job=jobs[0]
prepared_path=root/'candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/prepared.json'
assert sha(prepared_path)=='6c65c8c17f80575cbb643505e06cd8e6c8150d90ccdb23c0d49d4320aace2103'
prepared=json.loads(prepared_path.read_bytes())
assert job['entry']==prepared['entry'] and job['dt_root']==prepared['dt_root'], 'Current active AppWorld is not the prepared row-prefix owner'
assert job['resume_mode']=='disable' and not job['checkpoint_restore_requested']
driver=psutil.Process(job['pid'])
assert driver.create_time()==job['observed_process_created_unix']
source_path=Path(job['source_receipt']);source_sha=sha(source_path)
source=json.loads(source_path.read_bytes())
assert source['dt_root']==job['dt_root'] and source['verl_root']==job['verl_root']
assert source['candidate_environment']['path']==prepared['environment']['path']
assert source['candidate_environment']['sha256']==prepared['environment']['sha256']
assert source['trainer_sha256']=='d35ddd26b7497b153cec22f22e92eb1a08ef70385428dcbdcc37e879d9f16a4f'
entry,dt,verl=map(Path,(job['entry'],job['dt_root'],job['verl_root']))
expected={
 'verl.workers.actor.dp_actor':(verl/'verl/workers/actor/dp_actor.py',source['verl_sha256']['verl/workers/actor/dp_actor.py']),
 'verl.workers.fsdp_workers':(verl/'verl/workers/fsdp_workers.py',source['verl_sha256']['verl/workers/fsdp_workers.py']),
 'verl.utils.torch_functional':(verl/'verl/utils/torch_functional.py',source['verl_sha256']['verl/utils/torch_functional.py']),
 'deltatrace_rollout':(entry/'deltatrace_rollout.py',source['entry_sha256']['deltatrace_rollout.py']),
 'native_prefix_leases':(entry/'native_prefix_leases.py',source['entry_sha256']['native_prefix_leases.py']),
 'reward_readout':(entry/'reward_readout.py',source['entry_sha256']['reward_readout.py']),
 **{name:(dt/('clean/qwen35/'+name+'.py'),source['dt_source_sha256']['clean/qwen35/'+name+'.py'])
    for name in ('qwen35_dense_finite_runner','qwen35_answer_finite','qwen35_native_prefix_artifacts',
                 'vendor_fa_finite_bf16_d256','qwen35_gdn_finite','finite_fla_gpu')},
 'transformers.models.qwen3_5.modeling_qwen3_5':(Path(source['canonical_HF_owner']['path']),source['canonical_HF_owner']['sha256'])}
expected={name:dict(path=str(path),resolved_path=str(path.resolve()),sha256=digest) for name,(path,digest) in expected.items()}
for value in expected.values():assert sha(value['path'])==value['sha256'],value['path']
trainer=verl/'verl/trainer/ppo/ray_trainer.py'
assert sha(trainer)==source['trainer_sha256']
trainer_bytes=trainer.read_bytes();trainer_ast=ast.parse(trainer_bytes)
whiten_calls=[dict(line=n.lineno,call=ast.unparse(n)) for n in ast.walk(trainer_ast)
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='masked_whiten']
trainer_lines=trainer_bytes.decode().splitlines()
trainer_read=dict(path=str(trainer),sha256=sha(trainer),masked_whiten_calls=whiten_calls,
 source_excerpt=[dict(line=i+1,text=line) for i,line in enumerate(trainer_lines) if 367<=i<378],
 scope='Read existing trainer source only; does not prove this fresh run has executed whitening')
for path in reversed([job['entry'],job['verl_root']]):sys.path.insert(0,path)
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
import ray
ray.init(address='127.0.0.1:'+port,log_to_driver=False)

def inspect_imports(worker):
 import inspect,os,psutil,hashlib,time,sys
 from pathlib import Path
 owner=next(w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False))
 files={}
 for name,binding in expected.items():
  module=sys.modules.get(name)
  if module is None:
   files[name]=dict(status='not_imported',expected=binding)
   continue
  path=Path(inspect.getsourcefile(module));digest=hashlib.sha256(path.read_bytes()).hexdigest()
  files[name]=dict(status='imported',path=str(path),resolved_path=str(path.resolve()),sha256=digest,
   expected=binding,matches=path.resolve()==Path(binding['resolved_path']) and digest==binding['sha256'])
 installed={}
 for name in ('vllm_metax.device_allocator.cumem','vllm.v1.worker.gpu_worker'):
  module=sys.modules.get(name)
  if module is None:installed[name]=dict(status='not_imported');continue
  path=Path(inspect.getsourcefile(module))
  installed[name]=dict(status='imported',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 env_path=os.environ.get('DT_ENVIRONMENT_JSON')
 environment=dict(path=env_path,status='not_set')
 if env_path:
  import json
  path=Path(env_path);data=json.loads(path.read_bytes())['qwen35']
  environment=dict(path=env_path,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
   owner_kwargs={k:data.get(k) for k in ('individual_prefixes','boundary_row_storage')},
   finite_library=data['finite_library'],finite_library_sha256=data['finite_library_sha256'])
 cfg=owner.config
 return dict(pid=os.getpid(),pid_birth=psutil.Process().create_time(),rank=owner.rank,observed_unix=time.time(),
  imported_files=files,installed_imports_present=installed,
  shared_padding=os.environ.get('VERL_TRIM_SHARED_PADDING'),
  lora_rank=cfg.model.lora_rank,lora_alpha=cfg.model.lora_alpha,
  actor_microbatch=cfg.actor.ppo_micro_batch_size_per_gpu,ppo_epochs=cfg.actor.ppo_epochs,
  ppo_minibatch=cfg.actor.ppo_mini_batch_size,entropy_coeff=cfg.actor.entropy_coeff,clip_ratio_c=cfg.actor.clip_ratio_c,
  rollout_max_model_len=cfg.rollout.max_model_len,dt_environment=environment,extra_model_calls=0,
  scope='Only sys.modules, existing file bytes, environment and actor configuration are read; lazy DT modules are not imported or invoked')

try:
 actors=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in a['name']]
 assert len(actors)==len(job['devices'])
 refs=[ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=inspect_imports) for a in actors]
 workers=ray.get(refs,timeout=25)
 current=next(j for j in json.loads(manifest_path.read_bytes())['jobs'] if j['task']=='AppWorld')
 assert current['pid']==job['pid'] and current['observed_process_created_unix']==job['observed_process_created_unix']
 assert psutil.Process(job['pid']).create_time()==job['observed_process_created_unix']
 assert sha(source_path)==source_sha
 result=dict(driver_pid=driver.pid,driver_created_unix=driver.create_time(),job_output=job['output'],
  source=dict(path=str(source_path),sha256=source_sha),entry=job['entry'],dt_root=job['dt_root'],
  trainer_source=trainer_read,workers=workers,observed_unix=time.time(),
  scope='Read-only existing original-worker RPC; no forward/backward/update/checkpoint/runtime override, no DT lazy import; source evidence is distinct from execution evidence')
 out=root/'receipts/appworld-efficiency-20261007/row-prefix-owner-fresh-v1'/('actual-worker-imports-'+str(time.time_ns())+'.json')
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(receipt=str(out),sha256=sha(out),**result)),flush=True)
finally:ray.shutdown()
PY
