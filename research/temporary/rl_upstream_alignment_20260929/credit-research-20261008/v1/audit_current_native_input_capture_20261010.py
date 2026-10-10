"""Read the real passive input snapshots on CPU; no model or training replay."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
audit=r'''
import hashlib,io,json,psutil,resource,time
from pathlib import Path
import torch
root=Path(ROOT);out=root/'receipts/textcraft-native-actor-incidents-20261010-v1'
assert psutil.Process(982372).create_time()==1791553809.84
started=time.perf_counter();rows=[]
def finite(value):
 if isinstance(value,torch.Tensor):return dict(tensors=1,elements=value.numel(),nonfinite=int((~torch.isfinite(value)).sum()))
 parts=[finite(x) for x in (value.values() if isinstance(value,dict) else value)] if isinstance(value,(dict,list,tuple)) else []
 return {key:sum(p[key] for p in parts) for key in ['tensors','elements','nonfinite']}
for pid,birth in [(987808,1791553850.00),(989860,1791553867.51)]:
 worker=psutil.Process(pid);assert worker.create_time()==birth
 folder=next(out.glob('rank*-pid'+str(pid)))
 path=folder/'pending-update.pt'
 events=[json.loads(line) for line in (folder/'native-events.jsonl').read_text().splitlines()] if (folder/'native-events.jsonl').exists() else []
 preclip=[json.loads(line) for line in (folder/'preclip-events.jsonl').read_text().splitlines()] if (folder/'preclip-events.jsonl').exists() else []
 item=dict(pid=pid,birth=birth,phase=worker.name(),snapshot_exists=path.exists(),native_events=events,preclip_events=preclip)
 if path.exists():
  blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
  data=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
  assert data['pid']==pid and data['birth']==birth and data['source_sha256']=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
  batch=data['input_batch'];width=batch['responses'].shape[-1]
  mask=batch['loss_mask'][:,-width:].bool() if 'loss_mask' in batch else batch['attention_mask'][:,-width:].bool()
  fields={}
  for key in ['old_log_probs','ref_log_prob','advantages']:
   if key not in batch:continue
   value=batch[key];active=value[mask]
   fields[key]=dict(shape=list(value.shape),dtype=str(value.dtype),all_nonfinite=int((~torch.isfinite(value)).sum()),
    active_nonfinite=int((~torch.isfinite(active)).sum()),active_min=float(active.min()) if active.numel() else None,
    active_max=float(active.max()) if active.numel() else None)
  cfg=data['effective_config']
  item.update(path=str(path),bytes=len(blob),sha256=digest,rank=data['rank'],update_index=data['update_index'],
    batch_rows=batch.batch_size[0],input_shape=list(batch['input_ids'].shape),
    active_policy_tokens=int(mask.sum()),fields=fields,
    trainable_local_parameters=finite(data['trainable_local_parameters']),
    optimizer_local_state=finite(data['optimizer_local_state']),
    source_path=data['source_path'],source_sha256=data['source_sha256'],
    ppo_micro_batch_size_per_gpu=cfg['actor']['ppo_micro_batch_size_per_gpu'],
    actor_ppo_mini_batch_size=cfg['actor']['ppo_mini_batch_size'],
    temperature=data['input_meta_info'].get('temperature'),
    meta_global_steps=data['input_meta_info'].get('global_steps'),
    non_tensor_keys=list(data['input_non_tensor_batch']),
    uid_sha256=hashlib.sha256(repr(data['input_non_tensor_batch'].get('uid')).encode()).hexdigest())
  del data,batch,blob
 rows.append(item)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(unix=time.time(),seconds=time.perf_counter()-started,workers=rows,
 cpu_auditor_PSS_bytes=psutil.Process().memory_full_info().pss,
 cpu_auditor_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 host_available_bytes=psutil.virtual_memory().available,CUDA_initialized=False,
 model_calls=0,DT_calls=0,rollouts=0,optimizer_calls=0,production_changes=0,
 scope='Saved native inputs/parameters/moments are finite or not; no backward or root-cause proof.')))
'''.replace('ROOT',repr(transport.ROOT),1)
outer=r'''
import json,os,subprocess
from pathlib import Path
source=json.loads(Path(SOURCE).read_bytes())
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None)
env['CUDA_VISIBLE_DEVICES']='-1';env['PYTHONPATH']=source['pythonpath']
r=subprocess.run([env['VENV_PYTHON'],'-c',AUDIT],env=env,capture_output=True,text=True,timeout=45)
if r.returncode:raise RuntimeError(r.stderr)
print(r.stdout)
'''.replace('SOURCE',repr(transport.ROOT+'/runs/textcraft-formal-stable-20261009-v1/source.json'),1).replace('AUDIT',repr(audit),1)
command='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+outer+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=55)
r.check_returncode();d=json.loads(r.stdout)
path=HERE/'direct-credit-records-20261009-v1'/('native-input-capture-audit-'+str(int(d['unix']))+'.json')
path.write_bytes(r.stdout)
print(json.dumps(dict(saved=str(path),**d)))
