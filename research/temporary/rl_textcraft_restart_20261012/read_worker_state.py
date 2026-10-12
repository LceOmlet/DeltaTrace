"""Read actual imported owners and initial CPU optimizer state through official RPC."""
from pathlib import Path
import hashlib,inspect,json,os,sys,time,psutil,subprocess
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-native-graph-deployment-20261012-v1'
def read_worker(worker):
 import sys,hashlib,json,os,psutil
 from pathlib import Path
 owners=[w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False)];assert len(owners)==1
 w=owners[0];modules={}
 for name in ['verl.workers.actor.dp_actor','verl.trainer.ppo.core_algos','verl.workers.fsdp_workers','verl.utils.checkpoint.fsdp_checkpoint_manager','verl.workers.sharding_manager.fsdp_vllm','reward_readout','qwen35_gdn_finite','qwen35_dense_finite_runner','qwen35_decoder_finite','deltatrace_rollout','vllm','vllm.v1.worker.gpu_worker','vllm_metax.device_allocator.cumem']:
  m=sys.modules.get(name);p=Path(m.__file__) if m is not None else None
  modules[name]=dict(loaded=p is not None,path=str(p) if p else None,sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p else None)
 opt=getattr(w,'actor_optimizer',None);schedule=getattr(w,'actor_lr_scheduler',None)
 c=w.config
 engine=w.rollout.inference_engine
 vc=engine.llm_engine.vllm_config
 result=dict(pid=os.getpid(),birth=psutil.Process().create_time(),rank=w.rank,unix=time.time(),modules=modules,
  optimizer_states=len(opt.state) if opt is not None else None,scheduler_last_epoch=schedule.last_epoch if schedule is not None else None,
  configuration=dict(lora_rank=c.model.lora_rank,lora_alpha=c.model.lora_alpha,microbatch=c.actor.ppo_micro_batch_size_per_gpu,minibatch_per_rank=c.actor.ppo_mini_batch_size,ppo_epochs=c.actor.ppo_epochs,entropy=c.actor.entropy_coeff,dual_clip=c.actor.clip_ratio_c,context=c.rollout.max_model_len,max_num_seqs=c.rollout.max_num_seqs,enforce_eager=vc.model_config.enforce_eager,async_scheduling=vc.scheduler_config.async_scheduling,prefix_cache=vc.cache_config.enable_prefix_caching,cudagraph_mode=str(vc.compilation_config.cudagraph_mode)),
  passive_credit_directory=getattr(sys.modules['reward_readout'].DirectActionTargetReadout,'_diagnostic_directory',None) if 'reward_readout' in sys.modules else None,model_calls=0,optimizer_calls=0)
 return result
if __name__=='__main__':
 import ray
 method=sys.argv[1]
 manifest=json.loads((ROOT/'formal-training.json').read_bytes())
 job=next(j for j in manifest['jobs'] if j.get('task')=='TextCraft' and j.get('method')==method)
 assert 'formal-native-graph-fresh-20261012-v1' in job['output']
 pid=job['pid'];birth=job['observed_process_created_unix']
 p=psutil.Process(pid);assert p.create_time()==birth
 gcs=next(x for x in p.children(recursive=True) if x.name()=='gcs_server')
 port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
 host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
 ray.init(address=host+':'+port,log_to_driver=False)
 names=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']];assert len(names)==2
 results=ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=read_worker) for a in names])
 data=dict(unix=time.time(),method=method,pid=pid,birth=birth,status='actual_owner_read_completed',scope=__doc__,workers=results)
 (OUT/(method+'-initial-worker-state.json')).write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
 ray.shutdown()
