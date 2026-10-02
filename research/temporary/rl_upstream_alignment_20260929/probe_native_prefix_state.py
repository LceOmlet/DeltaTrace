"""Bounded readout of an existing native prefix's FLA state.

Observe one layer of one real DT prefix forward on each rank. Reuse the
installed FLA state function twice (full prefix and one shorter prefix), never
another model forward. No cache, attribution, kernel or training result is
replaced. Restore the original runner binding after the observed call.
"""
from pathlib import Path
import hashlib
import subprocess

from stage_environment_entry import ENTRY, ROOT, REPO, remote


script = r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,os,pathlib,psutil,sys,time,ray
root=pathlib.Path('@ROOT@')
job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='AppWorld')
for p in reversed([job['entry'],job['verl_root']]):sys.path.insert(0,p)
os.environ['PYTHONPATH']=':'.join([job['entry'],job['verl_root'],os.environ.get('PYTHONPATH','')])
driver=psutil.Process(job['pid'])
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
out=root/'receipts/owner-b8-dispatch-20260930'/f'formal-native-prefix-state-{int(time.time())}'
out.mkdir()

def install(worker):
 import hashlib,importlib,inspect,json,os,time
 from pathlib import Path
 import torch
 from accelerated.native_capture_events import LocalCaptureEvents
 owner=next(w for w in worker.worker_dict.values() if hasattr(w,'_deltatrace_producer'))
 runner=owner._deltatrace_producer.runner;model=runner.model
 original=runner.attribute;original_forward=model.forward_root
 had_attribute='attribute' in vars(runner);old_attribute=vars(runner).get('attribute')
 had_forward='forward_root' in vars(model);old_forward=vars(model).get('forward_root')
 chunk=importlib.import_module('fla.ops.gated_delta_rule.chunk')
 state_owner=chunk.chunk_gated_delta_rule_fwd_h
 stage=chunk.chunk_gated_delta_rule_fwd
 record=dict(rank=owner.rank,pid=os.getpid(),installed_unix=time.time(),restored=False,
  scope='One native prefix, one GDN layer, two original state-operator calls; no additional model forward or changed owner result',
  original_source=inspect.getsourcefile(original),
  original_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(original)).read_bytes()).hexdigest(),
  state_owner_source=inspect.getsourcefile(state_owner),
  state_owner_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(state_owner)).read_bytes()).hexdigest(),
  actor_microbatch=owner.actor.config.ppo_micro_batch_size_per_gpu,
  skipped_short_calls=0)
 path=out/f'rank{owner.rank}.json'
 def save():path.write_text(json.dumps(record,indent=2)+'\n')
 def restore():
  if had_forward:model.forward_root=old_forward
  elif 'forward_root' in vars(model):delattr(model,'forward_root')
  if had_attribute:runner.attribute=old_attribute
  elif 'attribute' in vars(runner):delattr(runner,'attribute')
  record.update(restored=True,restored_unix=time.time())
 def forward(*args,**kwargs):
  ids=kwargs.get('input_ids',args[0] if args else None)
  if ids is None or not 4096<=ids.shape[-1]<=16384 or kwargs.get('past_key_values') is not None:
   record['prefix_not_observed']='Outside bounded native-prefix observation'
   return original_forward(*args,**kwargs)
  captured={}
  def event(frame,kind,value):
   if kind!='return' or captured or value is None:return
   f=frame.f_locals
   if f['initial_state'] is not None or not f['output_final_state'] or f['cu_seqlens'] is not None:return
   for name in ('k','w','u','g','final_state'):captured[name]=f[name].detach()
  with LocalCaptureEvents([stage.__code__],event,returns_only=True):
   result=original_forward(*args,**kwargs)
  if not captured:
   record['prefix_not_observed']='Native stage did not expose the expected final state'
   return result
  try:
   free,total=torch.cuda.mem_get_info()
   if free<4*1024**3:
    record['probe_not_run']={'free_bytes':free,'reason':'Bounded optional readout has insufficient physical headroom'}
    return result
   k,w,u,g=(captured[n] for n in ('k','w','u','g'));reference=captured['final_state']
   full=int(k.shape[1]);half=(full//128)*64
   record.update(prefix_shape=list(k.shape),native_final_state_dtype=str(reference.dtype),
    intermediate_input_dtype=str(k.dtype),physical_free_bytes=free,readouts=[])
   for length in (full,half):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();tick=time.perf_counter()
    # Existing owner performs the recurrence and emits its FP32 final state.
    h,new_value,state=state_owner(k=k[:,:length].contiguous(),
     w=w[:,:length].contiguous(),u=u[:,:length].contiguous(),
     g=g[:,:length].contiguous(),initial_state=None,output_final_state=True)
    b.record();b.synchronize()
    row=dict(tokens=length,host_and_wait_seconds=time.perf_counter()-tick,
     stream_seconds=a.elapsed_time(b)/1000,state_dtype=str(state.dtype),
     state_bytes=state.numel()*state.element_size(),
     intermediate_state_bytes=h.numel()*h.element_size())
    if length==full:
     row.update(equal_to_native_final_state=bool(torch.equal(state,reference)),
      max_absolute_difference=float((state-reference).abs().max()))
    record['readouts'].append(row)
    del h,new_value,state
  except Exception as error:
   # Optional diagnostic failure is not a different training failure policy.
   record['probe_error']=repr(error)
  finally:captured.clear()
  return result
 def observed(*args,**kwargs):
  paired=args[0] if args else kwargs['paired_ids']
  if paired.shape[-1]<8192:
   record['skipped_short_calls']+=1
   return original(*args,**kwargs)
  model.forward_root=forward
  try:return original(*args,**kwargs)
  finally:
   restore();record['paired_shape']=list(paired.shape)
   try:save()
   except Exception as error:print('[native prefix state diagnostic unavailable] '+repr(error),flush=True)
 save();runner.attribute=observed
 return dict(rank=owner.rank,pid=os.getpid(),receipt=str(path),installed=True)

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 actors=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in a['name']]
 refs=[ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=install) for a in actors]
 submission=dict(driver_pid=driver.pid,driver_created_unix=driver.create_time(),
  entry=job['entry'],verl_root=job['verl_root'],submitted_unix=time.time(),
  diagnostic_commit='@COMMIT@',diagnostic_source_sha256='@SHA@',receipt=str(out),
  status='Queued at original worker RPC boundary; not yet observed')
 (out/'submitted.json').write_text(json.dumps(submission,indent=2)+'\n')
 print(json.dumps(submission),flush=True)
 installed=ray.get(refs)
 (out/'installed.json').write_text(json.dumps(dict(submission,workers=installed),indent=2)+'\n')
 print(json.dumps(dict(submission,workers=installed,status='One-shot native state readout installed')),flush=True)
finally:ray.shutdown()
PY
'''

if __name__ == '__main__':
    remote(script.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY)
           .replace('@COMMIT@', subprocess.check_output(
               ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
           .replace('@SHA@', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
