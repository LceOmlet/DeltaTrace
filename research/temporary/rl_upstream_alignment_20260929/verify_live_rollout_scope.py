"""Bounded replay at an existing RPC boundary, preserving the training worker.

Use four saved factual TextCraft prompts per rank, current LoRA8/16 and the
original vLLM 64-token/top5 hybrid comparison. No environment rollout, optimizer
update, model instance, GPU reservation or changed formal sampling parameter.
The unchanged vLLM comparison is a token/top-k check, not scalar allclose.
"""
from stage_environment_entry import remote, ROOT, ENTRY


remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,psutil,ray,time
root=Path('@ROOT@');out=root/'receipts/owner-b8-dispatch-20260930/rollout-scope'
job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='TextCraft')
driver=psutil.Process(job['pid']);assert driver.create_time()==job['observed_process_created_unix']
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))

def replay(worker):
 import ast,hashlib,importlib.util,json,os,sys,time,torch,warnings
 import numpy as np
 from pathlib import Path
 from verl import DataProto
 record=dict(pid=os.getpid(),unix=time.time(),cases=[],sources={})
 def owner_function(path,name,class_name=None):
  record['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  nodes=ast.parse(path.read_text()).body
  if class_name:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==class_name).body
  node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
  node.decorator_list=[]
  tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
  namespace=dict(torch=torch,warnings=warnings)
  exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
  return namespace[name]
 convert=owner_function(root/'receipts/upstream-alignment-20260929/vllm015-conftest.py','_final_steps_generate_w_logprobs','VllmRunner')
 compare=owner_function(root/'receipts/upstream-alignment-20260929/vllm015-test-utils.py','check_logprobs_close')
 spec=importlib.util.spec_from_file_location('verl.workers._scope_verifier',out/'fsdp_workers.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 candidate=module.ActorRolloutRefWorker
 record['sources'][str(out/'fsdp_workers.py')]=hashlib.sha256((out/'fsdp_workers.py').read_bytes()).hexdigest()
 for role,w in worker.worker_dict.items():
  if not hasattr(w,'_deltatrace_producer'):continue
  record.update(role=role,rank=w.rank,lora=[dict(rank=x.r,alpha=x.lora_alpha) for x in w.actor.actor_module.peft_config.values()])
  saved=w._deltatrace_producer.readout.last_report['minimum_log_ratio_batch']['samples']
  prompts=[s['selected_input_ids'][:s['source_start']] for s in saved]
  assert len(prompts)==4
  record['prompt_lengths']=list(map(len,prompts))
  (out/f'prompts-{os.getpid()}.json').write_text(json.dumps(prompts)+'\n')
  engine=w.rollout.inference_engine;original_generate=engine.generate
  manager=w.rollout_sharding_manager
  original_gen_rng=manager.gen_random_states.clone() if manager.gen_random_states is not None else None
  original_torch_rng=manager.torch_random_states.clone()
  had_scope=hasattr(w,'_owner_rollout_context_open');old_scope=getattr(w,'_owner_rollout_context_open',False)
  assert not old_scope
  captured=[]
  def capture(*args,**kwargs):
   tick=time.perf_counter();result=original_generate(*args,**kwargs)
   captured.append((result,time.perf_counter()-tick));return result
  engine.generate=capture
  def run(label,fn):
   raw=np.empty(4,dtype=object);raw[:]=prompts
   options=np.empty(4,dtype=object);options[:]=[dict(temperature=0.,max_tokens=64,logprobs=5,n=1,seed=1234)]*4
   batch=DataProto.from_dict(tensors=dict(input_ids=torch.ones(4,1,dtype=torch.long),
    attention_mask=torch.ones(4,1,dtype=torch.long),position_ids=torch.zeros(4,1,dtype=torch.long)),
    non_tensors=dict(raw_prompt_ids=raw,owner_sampling_kwargs=options))
   tick=time.perf_counter();fn(batch);wall=time.perf_counter()-tick
   outputs,engine_seconds=captured[-1]
   ids=sorted(engine.llm_engine.list_loras())
   row=dict(label=label,rpc_seconds=wall,engine_seconds=engine_seconds,lora_ids=ids,
    cached_tokens=sum(o.num_cached_tokens for o in outputs),prompt_tokens=sum(len(o.prompt_token_ids) for o in outputs),
    generated_tokens=sum(len(s.token_ids) for o in outputs for s in o.outputs))
   record['cases'].append(row)
   print('[Bounded rollout scope replay] '+json.dumps(dict(pid=os.getpid(),**row)),flush=True)
   return convert(outputs)
  try:
   with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
    original=run('original_per_call_context',w.generate_sequences)
    original_warm=run('original_per_call_context_repeat',w.generate_sequences)
    candidate.begin_rollout_context(w)
    try:
     cold=run('whole_rollout_first_call',lambda batch:candidate.generate_sequences(w,batch))
     cached=run('whole_rollout_second_call',lambda batch:candidate.generate_sequences(w,batch))
    finally:candidate.end_rollout_context(w)
    for label,value in [('original_repeat',original_warm),('scope_first',cold),('scope_cached',cached)]:
     compare(outputs_0_lst=original,outputs_1_lst=value,name_0='original_vllm',name_1=label)
    record['status']='passed_original_vllm_hybrid_comparison'
    torch.save(dict(original=original,original_repeat=original_warm,scope_first=cold,scope_cached=cached),out/f'outputs-{os.getpid()}.pt')
  except Exception as error:
   record.update(status='failed',error=repr(error))
  finally:
   if getattr(w,'_owner_rollout_context_open',False):candidate.end_rollout_context(w)
   engine.generate=original_generate
   manager.gen_random_states=original_gen_rng;manager.torch_random_states=original_torch_rng
   if had_scope:w._owner_rollout_context_open=old_scope
   elif hasattr(w,'_owner_rollout_context_open'):del w._owner_rollout_context_open
   record['restored']=engine.generate==original_generate and not getattr(w,'_owner_rollout_context_open',False)
   record['finished_unix']=time.time()
   (out/f'replay-{os.getpid()}.json').write_text(json.dumps(record,indent=2)+'\n')
 return record

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 actors=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in a['name']]
 assert len(actors)==2
 refs=[ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(replay) for a in actors]
 print(json.dumps(dict(status='queued bounded replay at original RPC boundary',driver_pid=driver.pid)),flush=True)
 result=dict(driver_pid=driver.pid,driver_created_unix=driver.create_time(),workers=ray.get(refs))
 (out/'replay-complete.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)
finally:ray.shutdown()
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY))
