"""One B4 replay per existing TextCraft rank, using original DT/worker APIs.

Saved factual response IDs and returns go through the existing readout. Only
metadata and CUDA events wrap installed FSDP functions; no collective, dtype,
parameter, numerical rule, optimizer or formal sampling option is replaced.
"""
import argparse
from stage_environment_entry import remote, ROOT, ENTRY

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--compare-partitions',action='store_true',
    help='Two B4 calls/rank using duplicated real cases, comparing native partitioning with the deployed contiguous split.')
args=parser.parse_args()

remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,psutil,ray,time
root=Path('@ROOT@')
job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='TextCraft')
driver=psutil.Process(job['pid']);assert driver.create_time()==job['observed_process_created_unix']
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(x.split('=',1)[1] for x in gcs.cmdline() if x.startswith('--gcs_server_port='))
out=root/'receipts/owner-b8-dispatch-20260930'/f'native-dt-gathers-{int(time.time())}'
out.mkdir()

def profile(worker, prepared=None):
 import functools,hashlib,inspect,json,os,time,torch
 import torch.distributed as dist
 from pathlib import Path
 from torch.distributed.fsdp._fully_shard import _fsdp_collectives as ops, _fsdp_param_group as pg
 from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState
 from owner_trajectory_batch import native_credit_response_batch
 record=dict(pid=os.getpid(),unix=time.time(),scope='One saved B4 replay; original worker/readout/FSDP calls with stream events, no optimizer update')
 for role,w in worker.worker_dict.items():
  if not hasattr(w,'_deltatrace_producer'):continue
  producer=w._deltatrace_producer;readout=producer.readout;old_report=readout.last_report
  saved=(old_report['minimum_log_ratio_batch'] if prepared is None else prepared['saved'])
  samples=saved['samples'];assert len(samples) in (4,8)
  rows=[]
  for s in samples:
   rows.append(dict(prompt_ids=s['selected_input_ids'][:s['source_start']],
    response_ids=s['selected_input_ids'][s['source_start']:s['source_end']],
    active_masks=True,env_step=s['trace']['source_step'],traj_uid=s['traj_uid'],
    data_source='textcraft',rewards=s['observed_return'],episode_rewards=s['observed_return'],
    # The original reward manager reads this even with normalize_by_length=False.
    # The saved replay has no full-episode length; None is unused metadata, not
    # a fabricated episode length or a changed reward.
    episode_lengths=None))
  data=native_credit_response_batch(readout.tokenizer,rows)
  data.batch['dt_complete_return']=torch.tensor([s['observed_return'] for s in samples],dtype=torch.float64)
  data.meta_info.update(eos_token_id=saved['eos_token_id'],pad_token_id=readout.tokenizer.pad_token_id)
  # Confirm the existing target builder recovers the exact saved endpoints.
  from agent_system.multi_turn_rollout.utils import to_list_of_dict
  original_rows=to_list_of_dict(data)
  counter=dict(nonzero_reward_events=0,policy_tokens=0,actual_row_lengths=[])
  _,requests=readout._prepare_episode(original_rows,counter,data.batch['dt_complete_return'].tolist())
  assert len(requests)==len(samples)
  for s,r in zip(samples,requests):
   assert torch.cat([r[n] for n in ('prompt','actions','query','target')]).tolist()==s['selected_input_ids']
  assert readout.alphabet.label_ids(readout.tokenizer)==saved['outcome_token_ids']
  if prepared is None:
   return dict(rank=w.rank,pid=os.getpid(),saved=saved,prepared=True)
  record.update(rank=w.rank,role=role,lengths=[len(s['selected_input_ids']) for s in samples],
    label=prepared.get('label','profile'),row_indices=prepared.get('row_indices'),
    response_lengths=[s['source_end']-s['source_start'] for s in samples],
    actual_lora=[dict(rank=p.r,alpha=p.lora_alpha) for p in w.actor.actor_module.peft_config.values()],
    actor_microbatch=w.actor.config.ppo_micro_batch_size_per_gpu,dt_minibatch=readout.minibatch_size)
  (out/f"inputs-{record['label']}-{os.getpid()}.json").write_text(json.dumps(saved)+'\n')
  originals=dict(inputs=ops._get_param_all_gather_inputs,
   collective=dist.all_gather_into_tensor,copy_out=pg.foreach_all_gather_copy_out,
   unshard=pg.FSDPParamGroup.unshard)
  model=producer.runner.model
  original_replay=model.replay_finite_layer;original_prefix=model.forward_root
  original_attribute=producer.runner.attribute
  record['sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
   {Path(inspect.getsourcefile(fn)) for fn in [*originals.values(),original_replay,original_attribute]}}
  context=dict(stage='worker_boundary',group=None);events=[];gathers=[];dt_info=[]
  def event_call(label,fn,*args,**kwargs):
   begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   begin.record();tick=time.perf_counter();result=fn(*args,**kwargs);end.record()
   events.append((dict(kind=label,stage=context['stage'],group=context['group'],host_seconds=time.perf_counter()-tick),begin,end))
   return result
  def unshard(group,*args,**kwargs):
   previous=context['group'];context['group']=group._module_fqn
   try:return originals['unshard'](group,*args,**kwargs)
   finally:context['group']=previous
  def inputs(params):
   # Read stored tensor metadata only; do not access all_gather_inputs twice.
   cpu_bytes=0;source_dtypes=set();source_numels=0;offloaded=0
   for p in params:
    t=p._sharded_param_data if p.sharded_state==ShardedState.SHARDED else p._sharded_post_forward_param_data
    source_dtypes.add(str(t.dtype));source_numels+=t.numel()
    if p.offload_to_cpu and t.device.type=='cpu':
     cpu_bytes+=t.numel()*t.element_size();offloaded+=1
   values=event_call('parameter_input_copy_and_cast',originals['inputs'],params)
   gathers.append(dict(stage=context['stage'],group=context['group'],parameters=len(params),
    offloaded_parameters=offloaded,cpu_source_bytes=cpu_bytes,source_numels=source_numels,
    source_dtypes=sorted(source_dtypes),gather_input_bytes=sum(t.numel()*t.element_size() for ts in values for t in ts)))
   return values
  def collective(*args,**kwargs):return event_call('all_gather_collective',originals['collective'],*args,**kwargs)
  def copy_out(*args,**kwargs):return event_call('all_gather_copy_out',originals['copy_out'],*args,**kwargs)
  layer_indices={id(layer):i for i,layer in enumerate(model.model.language_model.layers)}
  def replay(layer,fn):
   previous=context['stage'];context['stage']='replay_'+str(layer_indices[id(layer)])
   try:return original_replay(layer,fn)
   finally:context['stage']=previous
  def prefix(*args,**kwargs):
   previous=context['stage'];context['stage']='prefix'
   try:return original_prefix(*args,**kwargs)
   finally:context['stage']=previous
  def attribute(*args,**kwargs):
   context['stage']='root_or_finite_boundary'
   result=original_attribute(*args,**kwargs)
   dt_info.append({k:result[1][k] for k in ('calls','complete_attribution_seconds_with_diagnostics') if k in result[1]})
   return result
  try:
   ops._get_param_all_gather_inputs=inputs;dist.all_gather_into_tensor=collective
   pg.foreach_all_gather_copy_out=copy_out;pg.FSDPParamGroup.unshard=unshard
   model.replay_finite_layer=replay;model.forward_root=prefix;producer.runner.attribute=attribute
   with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
    tick=time.perf_counter();result=w.compute_dt_token_advantages(data)
    torch.cuda.synchronize();record['worker_seconds']=time.perf_counter()-tick
   record['finite_outputs']={k:bool(torch.isfinite(v).all()) for k,v in result.batch.items()}
   output_path=out/f"outputs-{record['label']}-rank{w.rank}.pt"
   torch.save({k:v.cpu() for k,v in result.batch.items()},output_path)
   record['output_path']=str(output_path)
   record.update(status='profile_completed',gathers=gathers,dt_info=dt_info,
    events=[dict(row,stream_seconds=begin.elapsed_time(end)/1000) for row,begin,end in events])
  except Exception as error:
   record.update(status='profile_failed',error=repr(error))
  finally:
   ops._get_param_all_gather_inputs=originals['inputs'];dist.all_gather_into_tensor=originals['collective']
   pg.foreach_all_gather_copy_out=originals['copy_out'];pg.FSDPParamGroup.unshard=originals['unshard']
   model.replay_finite_layer=original_replay;model.forward_root=original_prefix
   producer.runner.attribute=original_attribute;readout.last_report=old_report
   record.update(restored=True,finished_unix=time.time())
   (out/f"{record['label']}-rank{w.rank}.json").write_text(json.dumps(record,indent=2)+'\n')
 return record

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 actors=[x for x in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in x['name']];assert len(actors)==2
 handles=[ray.get_actor(x['name'],namespace=x['namespace']) for x in actors]
 # Both CPU preparations must succeed before either rank enters a collective.
 prepared=ray.get([actor.execute_with_func_generator.remote(profile) for actor in handles])
 assert all(p['prepared'] for p in prepared)
 modes={'profile':prepared}
 if @COMPARE@:
  import sys
  sys.path.insert(0,job['verl_root'])
  from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
  assert prepared[0]['saved']['outcome_token_ids']==prepared[1]['saved']['outcome_token_ids']
  # Explicit bounded fixture: 8 factual cases repeated once, not 16 new tasks.
  samples=sorted([s for p in prepared for s in p['saved']['samples']]*2,key=lambda s:len(s['selected_input_ids']))
  partitions=get_seqlen_balanced_partitions([len(s['selected_input_ids']) for s in samples],k_partitions=2,equal_size=True)
  modes={}
  for label,groups in [('contiguous',[list(range(8)),list(range(8,16))]),('owner_balanced',partitions)]:
   modes[label]=[dict(p,label=label,row_indices=indices,saved=dict(p['saved'],samples=[samples[i] for i in indices])) for p,indices in zip(prepared,groups)]
 all_records={}
 for label,fixtures in modes.items():
  refs=[actor.execute_with_func_generator.remote(profile,fixture) for actor,fixture in zip(handles,fixtures)]
  print(json.dumps(dict(receipt=str(out),mode=label,status='saved B4 calls queued at original RPC boundary',driver_pid=driver.pid)),flush=True)
  records=ray.get(refs);all_records[label]=records
  (out/f'completed-{label}.json').write_text(json.dumps(dict(driver_pid=driver.pid,workers=records),indent=2)+'\n')
  for r in records:print(json.dumps({k:r[k] for k in ['pid','rank','status','restored','lengths','worker_seconds','error'] if k in r}),flush=True)
  assert all(r['status']=='profile_completed' and r['restored'] for r in records),records
 (out/'completed.json').write_text(json.dumps(dict(driver_pid=driver.pid,modes=all_records),indent=2)+'\n')
finally:ray.shutdown()
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY).replace('@COMPARE@',repr(args.compare_partitions)))
