"""Collect one real cached GDN convolution and its existing finite seed.

No extra forward/backward, replacement output, kernel or propagation rule.
The temporary observer calls the original module function and restores it at
the original attribute boundary. Tensor copies are diagnostic overhead, so
this invocation must not be used as a speed measurement.
"""
from pathlib import Path
import ast
import hashlib
import subprocess

from stage_environment_entry import ENTRY, REPO, ROOT, SSH


SCRIPT = r'''set -e
source @ENTRY@/metax-entry.env.sh
CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,psutil,sys,time,ray
root=Path('@ROOT@')
job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='AppWorld')
assert job['pid']==634197 and job['observed_process_created_unix']==1791303503.79
driver=psutil.Process(job['pid']);assert driver.create_time()==job['observed_process_created_unix']
for p in reversed([job['entry'],job['verl_root']]):sys.path.insert(0,p)
os.environ['PYTHONPATH']=':'.join([job['entry'],job['verl_root'],os.environ.get('PYTHONPATH','')])
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
out=root/'receipts/gdn-cached-conv-interface-20261007'/('original-call-'+str(int(time.time())))
out.mkdir(parents=True,exist_ok=False)

def install(worker):
 import functools,inspect,json,os,time
 from pathlib import Path
 import torch
 owner=next(w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False))
 def attach(producer):
  runner=producer.runner
  original_attribute=runner.attribute
  had_instance='attribute' in vars(runner);old_instance=vars(runner).get('attribute')
  module=runner.model.model.language_model.layers[0].linear_attn
  record=dict(rank=owner.rank,pid=os.getpid(),created_unix=psutil.Process().create_time(),
   installed_unix=time.time(),restored=False,status='Awaiting one original cached-conv DT call',
   diagnostic_source_sha256='@SHA@',repository_commit='@COMMIT@',
   maximum_single_input_bytes=384*1024*1024,maximum_total_snapshot_bytes=3*1024*1024*1024,
   tensors={},tensor_bytes=0,native_calls=0,finite_calls=0,skipped_oversized_calls=0,
   calls={},extra_model_forwards=0,extra_backwards=0,numerical_replacements=0,
   original_attribute_source=inspect.getsourcefile(original_attribute),
   original_convolution_source=inspect.getsourcefile(inspect.unwrap(module.causal_conv1d_fn)),
   scope='Passive original-call tensor copies; not timing/accuracy acceptance')
  record['owner_source_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
   for p in (record['original_attribute_source'],record['original_convolution_source'])}
  path=out/f'rank{owner.rank}.json';payload={}
  def save():path.write_text(json.dumps(record,indent=2)+'\n')
  def retain(name,tensor):
   if tensor is None:return
   try:
    n=tensor.numel()*tensor.element_size()
    if record['tensor_bytes']+n>record['maximum_total_snapshot_bytes']:
     record['snapshot_budget_exceeded']=True;return
    payload[name]=tensor.detach().to(device='cpu',copy=True)
    record['tensors'][name]=dict(shape=list(tensor.shape),stride=list(tensor.stride()),
     dtype=str(tensor.dtype),device=str(tensor.device),bytes=n)
    record['tensor_bytes']+=n
   except Exception as error:record.setdefault('observation_errors',[]).append(repr(error))
  def restore_attribute():
   if had_instance:runner.attribute=old_instance
   else:delattr(runner,'attribute')
   record.update(restored=True,restored_unix=time.time())
  def observed(*args,**kwargs):
   old_conv=module.causal_conv1d_fn;signature=inspect.signature(old_conv)
   paired=args[0] if args else kwargs['paired_ids']
   record['actual_paired_shape']=list(paired.shape)
   hooks=[];native_shape=None;pending_native={}
   def pending_retain(name,tensor):
    try:
     n=tensor.numel()*tensor.element_size()
     if n>record['maximum_single_input_bytes']:return
     pending_native['cpu'][name]=tensor.detach().to(device='cpu',copy=True)
     pending_native['metadata'][name]=dict(shape=list(tensor.shape),stride=list(tensor.stride()),
       dtype=str(tensor.dtype),device=str(tensor.device),bytes=n)
    except Exception as error:record.setdefault('observation_errors',[]).append(repr(error))
   def before(_module,args,kwargs):
    nonlocal native_shape
    value=args[0] if args else kwargs['hidden_states']
    native_shape=tuple(value.shape)
    pending_native.clear();pending_native.update(cpu={},metadata={},cached=False)
    try:
     if not record['native_calls']:
      cache=kwargs.get('cache_params') if 'cache_params' in kwargs else (args[1] if len(args)>1 else None)
      if cache is not None and cache.has_previous_state(_module.layer_idx):
       pending_native['cached']=True
       # Read the actual prior state before native update_conv_state mutates or replaces it.
       pending_retain('native_conv_state',cache.layers[_module.layer_idx].conv_states)
    except Exception as error:record.setdefault('observation_errors',[]).append(repr(error))
    return None
   def after_projection(_module,args,output):
    try:
     if pending_native.get('cached') and not record['native_calls']:
      # Keep the actual Linear output; its native transpose defines the second cat input.
      pending_retain('native_projection_output',output)
    except Exception as error:record.setdefault('observation_errors',[]).append(repr(error))
    return None
   @functools.wraps(old_conv)
   def conv(*args,**kwargs):
    result=old_conv(*args,**kwargs)
    try:
     arguments=signature.bind(*args,**kwargs);arguments.apply_defaults();a=arguments.arguments
     x=a['x'];activation=a['activation'];tensor=result[0] if isinstance(result,tuple) else result
     n=x.numel()*x.element_size()
     if a['seq_idx'] is not None or a['initial_states'] is not None:return result
     if activation in ('silu','swish') and not record['native_calls']:
      context=x.shape[-1]-native_shape[1]
      if context<=0:return result
      small_bytes=sum(v.numel()*v.element_size() for v in (a['weight'],a['bias']) if v is not None)
      if n>record['maximum_single_input_bytes'] or 7*n+2*small_bytes>record['maximum_total_snapshot_bytes']:
       record['skipped_oversized_calls']+=1;return result
      record['native_calls']=1
      record['calls']['native']=dict(input_shape=list(native_shape),left_context=context,
        activation=activation,width=a['weight'].shape[-1],seq_idx_none=True)
      retain('native_x',x);retain('native_output',tensor)
      retain('native_weight',a['weight']);retain('native_bias',a['bias'])
      for name in ('native_conv_state','native_projection_output'):
       if name in pending_native.get('cpu',{}):
        metadata=pending_native['metadata'][name]
        if record['tensor_bytes']+metadata['bytes']<=record['maximum_total_snapshot_bytes']:
         payload[name]=pending_native['cpu'][name]
         record['tensors'][name]=metadata;record['tensor_bytes']+=metadata['bytes']
      record['calls']['native']['actual_preconcat_inputs']=dict(
       conv_state='native_conv_state' if 'native_conv_state' in payload else None,
       projection_output='native_projection_output' if 'native_projection_output' in payload else None,
       projection_to_mixed_qkv='Original transpose(1,2); no output reconstruction',
       source='Original GDN pre-hook and original in_proj_qkv forward-hook',
       observer_hook_returns_none=True,extra_projection_forwards=0)
      pending_native.clear()
     elif activation is None and record['native_calls'] and not record['finite_calls'] and x.requires_grad:
      record['finite_calls']=1
      record['calls']['finite']=dict(activation=None,width=a['weight'].shape[-1],seq_idx_none=True,
        initial_states_none=True)
      retain('finite_x',x);retain('finite_pre',tensor)
      retain('finite_weight',a['weight']);retain('finite_bias',a['bias'])
      def seed(value):retain('finite_seed',value);return None
      def gradient(value):retain('finite_dx',value);return None
      hooks.append(tensor.register_hook(seed));hooks.append(x.register_hook(gradient))
    except Exception as error:record.setdefault('observation_errors',[]).append(repr(error))
    return result
   pre=module.register_forward_pre_hook(before,with_kwargs=True)
   projection_hook=module.in_proj_qkv.register_forward_hook(after_projection)
   module.causal_conv1d_fn=conv
   try:
    result=original_attribute(*args,**kwargs)
   except BaseException as error:
    record['owner_error']=repr(error);raise
   finally:
    module.causal_conv1d_fn=old_conv;pre.remove();projection_hook.remove();pending_native.clear()
    for handle in hooks:handle.remove()
    if record['finite_calls'] or 'owner_error' in record:
     restore_attribute()
     record['convolution_identity_restored']=module.causal_conv1d_fn is old_conv
     record['status']='Original call observed' if record['finite_calls'] else 'Original owner raised; observer restored'
     try:
      target=out/f'rank{owner.rank}.pt'
      with target.open('xb') as stream:torch.save(dict(record=record,tensors=payload),stream)
      with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
      record['tensor_file']=dict(path=str(target),bytes=target.stat().st_size,
       sha256=digest)
     except Exception as error:record.setdefault('observation_errors',[]).append(repr(error))
     payload.clear()
    else:
     payload.clear();record['tensors']={};record['tensor_bytes']=0;record['native_calls']=0
    try:save()
    except Exception as error:print('[conv observer unavailable] '+repr(error),flush=True)
   return result
  save();runner.attribute=observed
  return dict(rank=owner.rank,pid=os.getpid(),receipt=str(path),installed=True)

 if hasattr(owner,'_deltatrace_producer'):
  return attach(owner._deltatrace_producer)
 # The pinned worker initializes its producer only at the first actual DT RPC.
 # Observe that original constructor once; never initialize it from this RPC.
 from deltatrace_rollout import DeltaTraceRolloutProducer
 cls=DeltaTraceRolloutProducer;original_init=cls.__init__
 path=out/f'rank{owner.rank}.json'
 pending=dict(rank=owner.rank,pid=os.getpid(),created_unix=psutil.Process().create_time(),
  installed_unix=time.time(),status='Awaiting original lazy producer initialization',
  diagnostic_source_sha256='@SHA@',extra_model_forwards=0,extra_backwards=0,
  numerical_replacements=0,constructor_identity_restored=False)
 def save_pending():path.write_text(json.dumps(pending,indent=2)+'\n')
 @functools.wraps(original_init)
 def initialized(producer,*args,**kwargs):
  try:result=original_init(producer,*args,**kwargs)
  finally:
   cls.__init__=original_init
   pending['constructor_identity_restored']=True
  try:
   attach(producer)
  except Exception as error:
   pending.update(status='Optional observation unavailable; original constructor retained',
    observation_error=repr(error));save_pending()
  return result
 save_pending();cls.__init__=initialized
 return dict(rank=owner.rank,pid=os.getpid(),receipt=str(path),installed=True,
  status='One-shot observer armed at original lazy producer constructor; no producer initialized')

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 actors=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in a['name']]
 assert len(actors)==2, 'Use only this original two-worker Ray session'
 refs=[ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=install) for a in actors]
 submission=dict(driver_pid=driver.pid,driver_created_unix=driver.create_time(),
  submitted_unix=time.time(),entry=job['entry'],verl_root=job['verl_root'],
  source_receipt=job['source_receipt'],diagnostic_commit='@COMMIT@',
  diagnostic_source_sha256='@SHA@',receipt=str(out),status='Queued at original worker RPC boundary')
 (out/'submitted.json').write_text(json.dumps(submission,indent=2)+'\n')
 print(json.dumps(submission),flush=True)
 installed=ray.get(refs)
 (out/'installed.json').write_text(json.dumps(dict(submission,workers=installed),indent=2)+'\n')
 print(json.dumps(dict(submission,workers=installed,status='One-shot observer installed')),flush=True)
finally:ray.shutdown()
PY
'''


if __name__ == '__main__':
    source = SCRIPT.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY)
    source = source.replace('@COMMIT@',subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    source = source.replace('@SHA@',hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    code = source.split("<<'PY'\n",1)[1].rsplit('\nPY\n',1)[0]
    ast.parse(code)
    output = Path(__file__).parent/'gdn-cached-conv-interface-20261007'
    (output/'passive-capture.sh').write_text(source,encoding='utf-8')
    result = subprocess.run(SSH+['bash','-s'],input=source.encode(),capture_output=True)
    (output/'passive-capture.stdout.txt').write_bytes(result.stdout+result.stderr)
    print(result.stdout.decode(errors='replace'));print(result.stderr.decode(errors='replace'))
    result.check_returncode()
