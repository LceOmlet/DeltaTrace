"""Observe one native prepared DT request group per rank, then restore owner.

Only lengths, factual identities and exact prefix-extension booleans are saved.
No model call, cache transition, event, barrier or numerical result is added.
"""
from pathlib import Path
import argparse
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
label='requests' if @SAVE_REQUESTS@ else 'layout'
out=root/'receipts/owner-b8-dispatch-20260930'/f'formal-dt-prefix-{label}-{int(time.time())}'
out.mkdir()

def install(worker):
 import hashlib,inspect,json,os,time
 from pathlib import Path
 import torch
 from reward_readout import EventRatioReadout
 original=EventRatioReadout._prepare_episode
 producer=next(w._deltatrace_producer for w in worker.worker_dict.values() if hasattr(w,'_deltatrace_producer'))
 owner=next(w for w in worker.worker_dict.values() if hasattr(w,'_deltatrace_producer'))
 record=dict(pid=os.getpid(),rank=owner.rank,installed_unix=time.time(),
  owner_method_source=inspect.getsourcefile(original),
  owner_method_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(original)).read_bytes()).hexdigest(),
  scope='One actual pending request group metadata only; original vectors/requests/results returned unchanged, no new model/cache calculation',
  original_actor_microbatch=owner.actor.config.ppo_micro_batch_size_per_gpu,
  original_dt_minibatch=producer.readout_options.get('minibatch_size'),restored=False)
 path=out/f'rank{owner.rank}.json'
 path.write_text(json.dumps(record,indent=2)+'\n')
 def observed(readout,*args,**kwargs):
  try:result=original(readout,*args,**kwargs)
  finally:EventRatioReadout._prepare_episode=original
  tick=time.perf_counter()
  try:
   requests=result[1];grouped={};metadata=[];edges=[]
   for req in requests:
    uid=req['traj_uid'];digest=hashlib.sha256(uid.encode()).hexdigest()
    grouped.setdefault(uid,[]).append(req)
    metadata.append(dict(uid_sha256=digest,source_step=req['source_step'],
     source_start=req['start'],source_end=req['end'],context_tokens=req['context_tokens'],
     query_tokens=req['query_tokens'],observed_return=req['observed_return']))
   for uid,items in grouped.items():
    ordered=sorted(items,key=lambda x:x['source_step'])
    for old,new in zip(ordered,ordered[1:]):
     before=old['prompt'];after=new['prompt'];n=min(before.numel(),after.numel())
     equal=bool(torch.equal(before[:n],after[:n]))
     edges.append(dict(uid_sha256=hashlib.sha256(uid.encode()).hexdigest(),
      source_steps=[old['source_step'],new['source_step']],prefix_tokens=n,
      exact_shared_prefix=equal,history_grew=after.numel()>=before.numel()))
   record.update(observed_unix=time.time(),task=readout.alphabet.task,
    native_test_denominator=readout.alphabet.values,requests=metadata,
    exact_prefix_edges=edges,requests_count=len(requests),trajectory_count=len(grouped),
    metadata_seconds=time.perf_counter()-tick)
   if @SAVE_REQUESTS@:
    blob=out/f'actual-requests-rank{owner.rank}.pt'
    torch.save(dict(rows=args[0],complete_returns=args[2],requests=[{name:req[name] for name in (
      'prompt','actions','query','target','case','start','end','source_step',
      'traj_uid','observed_return','context_tokens','query_tokens','row_index')}
      for req in requests],eos_token_id=readout.tokenizer.eos_token_id,
      outcome_token_ids=readout.alphabet.label_ids(readout.tokenizer),
      task=readout.alphabet.task,max_length=readout.max_length,
      minibatch_size=readout.minibatch_size),blob)
    record['request_artifacts']=dict(path=str(blob),bytes=blob.stat().st_size,
      sha256=hashlib.sha256(blob.read_bytes()).hexdigest(),
      scope='Exact original prepared request tensors for an isolated same-workload comparison; no model or environment rerun')
    record['capture_and_save_seconds']=time.perf_counter()-tick
  except Exception as error:
   record['observation_error']=repr(error)
  finally:
   EventRatioReadout._prepare_episode=original
   record.update(restored=True,restored_unix=time.time())
   try:path.write_text(json.dumps(record,indent=2)+'\n')
   except Exception as error:print('[DT prefix observation unavailable] '+repr(error),flush=True)
  return result
 EventRatioReadout._prepare_episode=observed
 return dict(rank=owner.rank,pid=os.getpid(),receipt=str(path),installed=True)

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 actors=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in a['name']]
 refs=[ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=install) for a in actors]
 submission=dict(driver_pid=driver.pid,driver_created_unix=driver.create_time(),
  entry=job['entry'],verl_root=job['verl_root'],submitted_unix=time.time(),
  diagnostic_commit='@COMMIT@',diagnostic_source_sha256='@SHA@',receipt=str(out),
  status='Queued at original worker RPC boundary; no observation yet')
 (out/'submitted.json').write_text(json.dumps(submission,indent=2)+'\n')
 print(json.dumps(submission),flush=True)
 installed=ray.get(refs)
 (out/'installed.json').write_text(json.dumps(dict(submission,workers=installed),indent=2)+'\n')
 print(json.dumps(dict(submission,workers=installed,status='One-shot metadata observer installed')),flush=True)
finally:ray.shutdown()
PY
'''

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-requests',action='store_true',
                        help='Persist exact original CPU request tensors once; do not change training results.')
    options=parser.parse_args()
    remote(script.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY)
           .replace('@SAVE_REQUESTS@',str(options.save_requests))
           .replace('@COMMIT@', subprocess.check_output(
               ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
           .replace('@SHA@', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
