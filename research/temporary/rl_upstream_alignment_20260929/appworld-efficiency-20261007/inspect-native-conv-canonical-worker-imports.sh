set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,hashlib,time,psutil,sys,os
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='AppWorld')
driver=psutil.Process(job['pid']);assert driver.create_time()==job['observed_process_created_unix']
for path in reversed([job['entry'],job['verl_root']]):sys.path.insert(0,path)
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server');port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
import ray
ray.init(address='127.0.0.1:'+port,log_to_driver=False)
def inspect_imports(worker):
 import inspect,os,hashlib,time,sys
 from pathlib import Path
 owner=next(w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False))
 paths={}
 for name in ['verl.workers.actor.dp_actor','verl.workers.fsdp_workers','verl.utils.torch_functional','transformers.models.qwen3_5.modeling_qwen3_5','deltatrace_rollout','qwen35_dense_finite_runner','qwen35_gdn_finite']:
  module=sys.modules.get(name)
  if module is None:continue
  path=Path(inspect.getsourcefile(module));paths[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 installed={}
 for name in ['vllm_metax.device_allocator.cumem','vllm.v1.worker.gpu_worker']:
  if name in sys.modules:
   path=Path(inspect.getsourcefile(sys.modules[name]));installed[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 return {'installed_imports_present':installed,'pid':os.getpid(),'rank':owner.rank,'observed_unix':time.time(),'imported_files':paths,'shared_padding':os.environ.get('VERL_TRIM_SHARED_PADDING'),'lora_rank':owner.config.model.lora_rank,'lora_alpha':owner.config.model.lora_alpha,'actor_microbatch':owner.config.actor.ppo_micro_batch_size_per_gpu,'ppo_epochs':owner.config.actor.ppo_epochs,'entropy_coeff':owner.config.actor.entropy_coeff,'clip_ratio_c':owner.config.actor.clip_ratio_c,'dt_environment_json':os.environ.get('DT_ENVIRONMENT_JSON'),'native_conv_loader_enabled':os.environ.get('DT_PREFIX_NATIVE_CONV_INITIAL_STATES'),'extra_model_calls':0}
try:
 actors=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in a['name']];assert len(actors)==2
 refs=[ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=inspect_imports) for a in actors]
 workers=ray.get(refs,timeout=25)
 out=root/'receipts/appworld-efficiency-20261007/native-conv-canonical-owner-v2/actual-worker-imports.json'
 x={'driver_pid':driver.pid,'driver_created_unix':driver.create_time(),'workers':workers,'observed_unix':time.time(),'scope':'Read-only original-worker RPC; no forward/backward, update, checkpoint load or runtime override'}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True)
finally:ray.shutdown()
PY
