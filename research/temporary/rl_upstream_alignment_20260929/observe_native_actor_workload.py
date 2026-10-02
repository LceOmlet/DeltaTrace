"""Record eight actual native actor B4 input layouts without replacing math.

Only the original mask is copied to CPU for input-slot accounting. Do not call
another model, reconstruct tokens, change batching or impose a numerical gate.
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
job=next(j for j in json.loads((root/'active-training.json').read_bytes())['jobs'] if j['task']=='AppWorld')
for path in reversed([job['entry'],job['verl_root']]):sys.path.insert(0,path)
os.environ['PYTHONPATH']=':'.join([job['entry'],job['verl_root'],os.environ.get('PYTHONPATH','')])
driver=psutil.Process(job['pid'])
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
out=root/'receipts/owner-b8-dispatch-20260930'/f'formal-native-actor-workload-{int(time.time())}'
out.mkdir()

def install(worker):
 import hashlib,inspect,json,os,time
 from pathlib import Path
 import torch
 owner=next(w for w in worker.worker_dict.values() if hasattr(w,'_deltatrace_producer'))
 actor=owner.actor;original=actor._forward_micro_batch
 had_binding='_forward_micro_batch' in vars(actor);old_binding=vars(actor).get('_forward_micro_batch')
 path=out/f'rank{owner.rank}.json'
 source=Path(inspect.getsourcefile(original))
 record=dict(pid=os.getpid(),rank=owner.rank,installed_unix=time.time(),restored=False,
  source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
  local_microbatch=actor.config.ppo_micro_batch_size_per_gpu,
  local_ppo_minibatch=actor.config.ppo_mini_batch_size,ppo_epochs=actor.config.ppo_epochs,
  entropy_coeff=actor.config.entropy_coeff,samples=[],
  scope='Eight original calculate_entropy=True actor forwards; mask input-slot accounting only, no GPU timing, model/reward/optimizer/batch changes')
 def restore():
  if had_binding:actor._forward_micro_batch=old_binding
  elif '_forward_micro_batch' in vars(actor):delattr(actor,'_forward_micro_batch')
  record.update(restored=True,restored_unix=time.time())
 def observed(*args,**kwargs):
  entropy=kwargs.get('calculate_entropy',args[2] if len(args)>2 else False)
  if not entropy:return original(*args,**kwargs)
  tick=time.perf_counter()
  try:
   batch=kwargs.get('micro_batch',args[0] if args else None)
   mask=batch['attention_mask'].detach().cpu().bool()
   width=mask.shape[-1];columns=torch.arange(width)
   last=(mask*columns).amax(-1)
   shared_right=width-int(last.max())-1
   first=mask.long().argmax(-1)
   record['samples'].append(dict(observed_unix=time.time(),input_shape=list(batch['input_ids'].shape),
    response_shape=list(batch['responses'].shape),active_tokens_per_row=mask.sum(-1).tolist(),
    last_active_column_per_row=last.tolist(),first_active_column_per_row=first.tolist(),
    shared_right_padding_columns=shared_right,shared_left_padding_columns=int(first.min()),
    total_input_slots=mask.numel(),shared_right_padding_slots=mask.shape[0]*shared_right,
    actor_training=actor.actor_module.training,metadata_seconds=time.perf_counter()-tick))
   if len(record['samples'])==8:restore()
   path.write_text(json.dumps(record,indent=2)+'\n')
  except Exception as error:
   restore();record['observation_error']=repr(error)
   try:path.write_text(json.dumps(record,indent=2)+'\n')
   except Exception:pass
  return original(*args,**kwargs)
 actor._forward_micro_batch=observed
 path.write_text(json.dumps(record,indent=2)+'\n')
 return dict(rank=owner.rank,pid=os.getpid(),receipt=str(path),installed=True)

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 actors=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in a['name']]
 refs=[ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=install) for a in actors]
 receipt=dict(driver_pid=driver.pid,driver_created_unix=driver.create_time(),entry=job['entry'],
  verl_root=job['verl_root'],submitted_unix=time.time(),diagnostic_commit='@COMMIT@',
  diagnostic_source_sha256='@SHA@',receipt=str(out),status='Queued at native worker RPC boundary; no actor sample claimed')
 (out/'submitted.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
 installed=ray.get(refs)
 (out/'installed.json').write_text(json.dumps(dict(receipt,workers=installed),indent=2)+'\n')
 print(json.dumps(dict(receipt,workers=installed,status='Native actor layout observer installed')),flush=True)
finally:ray.shutdown()
PY
'''

if __name__ == '__main__':
    remote(script.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY)
           .replace('@COMMIT@', subprocess.check_output(
               ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
           .replace('@SHA@', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
