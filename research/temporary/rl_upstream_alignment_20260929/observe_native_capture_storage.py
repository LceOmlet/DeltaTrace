"""Read capture storage metadata from one existing formal DT call per rank.

The owner classes still capture and compute normally. This observer adds no
tensor copy, model call, kernel, CUDA event, barrier or change to a returned
attribution. Restore the original runner/backend even if the owner raises.
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
out=root/'receipts/owner-b8-dispatch-20260930'/f'formal-dt-capture-storage-{int(time.time())}'
out.mkdir()

def install(worker):
 import hashlib,inspect,json,os,time
 from pathlib import Path
 from types import SimpleNamespace
 import torch
 owner=next(w for w in worker.worker_dict.values() if hasattr(w,'_deltatrace_producer'))
 runner=owner._deltatrace_producer.runner
 original=runner.attribute;backend=runner.capture_backend
 had_instance='attribute' in vars(runner);old_instance=vars(runner).get('attribute')
 layers=runner.model.model.language_model.layers
 layer_ids={id(layer):i for i,layer in enumerate(layers)}
 mixer_ids={id(layer.self_attn if layer.block_type=='full_attention' else layer.linear_attn):i for i,layer in enumerate(layers)}
 record=dict(rank=owner.rank,pid=os.getpid(),installed_unix=time.time(),restored=False,
  original_source=inspect.getsourcefile(original),
  original_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(original)).read_bytes()).hexdigest(),
  original_actor_microbatch=owner.actor.config.ppo_micro_batch_size_per_gpu,
  scope='One original >=8192-token DT call storage metadata only; no extra tensors or model/numerical work',
  captures=[],skipped_short_calls=0)
 path=out/f'rank{owner.rank}.json'
 def save():path.write_text(json.dumps(record,indent=2)+'\n')
 def capture_metadata(capture,index,kind):
  tick=time.perf_counter()
  rows=[];storages={}
  try:
   for group in ('values','endpoints'):
    for name,value in getattr(capture,group,{}).items():
     if not isinstance(value,torch.Tensor):continue
     storage=value.untyped_storage()
     identity=(str(value.device),storage.data_ptr(),storage.nbytes())
     number=storages.setdefault(identity,len(storages))
     rows.append(dict(group=group,name=name,shape=list(value.shape),stride=list(value.stride()),
      dtype=str(value.dtype),device=str(value.device),logical_bytes=value.numel()*value.element_size(),
      storage_bytes=storage.nbytes(),storage_identity=number))
   record['captures'].append(dict(layer=index,kind=kind,tensors=rows,
    unique_storage_bytes=sum(key[2] for key in storages),
    unique_storage_count=len(storages),input_shape=list(getattr(capture,'input_shape',()) or ()),
    calls=dict(capture.calls),metadata_seconds=time.perf_counter()-tick))
  except Exception as error:record['metadata_error']=repr(error)
 class Decoder(backend.NativeDecoderCapture):
  def __exit__(self,*args):
   result=super().__exit__(*args)
   if args[0] is None:capture_metadata(self,layer_ids[id(self.layer)],'decoder')
   return result
 class Attention(backend.NativeDenseAttentionCapture):
  def __exit__(self,*args):
   result=super().__exit__(*args)
   if args[0] is None:capture_metadata(self,mixer_ids[id(self.module)],'attention')
   return result
 class GDN(backend.NativeGDNCapture):
  def __exit__(self,*args):
   result=super().__exit__(*args)
   if args[0] is None:capture_metadata(self,mixer_ids[id(self.module)],'GDN')
   return result
 def restore():
  runner.capture_backend=backend
  if had_instance:runner.attribute=old_instance
  else:delattr(runner,'attribute')
  record.update(restored=True,restored_unix=time.time())
 def observed(*args,**kwargs):
  paired=args[0] if args else kwargs['paired_ids']
  if paired.shape[-1]<8192:
   record['skipped_short_calls']+=1
   return original(*args,**kwargs)
  runner.capture_backend=SimpleNamespace(NativeDecoderCapture=Decoder,
   NativeDenseAttentionCapture=Attention,NativeGDNCapture=GDN)
  try:result=original(*args,**kwargs)
  except BaseException as error:
   record['owner_error']=repr(error)
   raise
  finally:
   restore()
   record.update(paired_shape=list(paired.shape),observed_unix=time.time())
   try:save()
   except Exception as error:print('[DT storage observation unavailable] '+repr(error),flush=True)
  info=result[1]
  record.update(native_shared_prefix_length=info.get('native_shared_prefix_length'),
   original_full_seconds=info.get('complete_attribution_seconds_with_diagnostics'),
   original_stage_timers=info.get('calls'))
  try:save()
  except Exception as error:print('[DT storage observation unavailable] '+repr(error),flush=True)
  return result
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
 print(json.dumps(dict(submission,workers=installed,status='One-shot storage metadata observer installed')),flush=True)
finally:ray.shutdown()
PY
'''

if __name__ == '__main__':
    remote(script.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY)
           .replace('@COMMIT@', subprocess.check_output(
               ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
           .replace('@SHA@', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
