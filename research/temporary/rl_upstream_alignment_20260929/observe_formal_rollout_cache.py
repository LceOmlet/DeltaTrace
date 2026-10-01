"""Observe eight existing AppWorld generate calls, without extra generation.

Read original RequestOutput metadata and elapsed time. Sampling, generated
objects, engine state transitions and training remain owned by VERL/vLLM.
Install at the original sequential worker RPC boundary; restore automatically.
"""
from pathlib import Path
import hashlib
import subprocess
from stage_environment_entry import remote, ROOT, ENTRY, REPO

revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,psutil,ray,time
root=Path('@ROOT@');job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='AppWorld')
driver=psutil.Process(job['pid']);assert driver.create_time()==job['observed_process_created_unix']
assert job['entry']==str(root/'candidates/appworld-rollout-scope-20261001/entry')
children=[]
for p in driver.children(recursive=True):
 try:children.append((p,p.name()))
 except psutil.NoSuchProcess:pass
gcs=next(p for p,name in children if name=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
out=root/'receipts/owner-b8-dispatch-20260930/appworld-rollout-scope/formal-cache'
if out.exists():
 # Both positional registrations failed Ray's client-side signature check
 # before any RPC. Its actual actor metadata exposes **kwargs only.
 assert not (out/'installed.json').exists() and not list(out.glob('rank*.json'))
 failure=out/('client-signature-failure-keywords.json' if (out/'client-signature-failure.json').exists()
              else 'client-signature-failure.json')
 assert not failure.exists()
 previous=json.loads((out/'submitted.json').read_text())
 previous['failure']='Ray client metadata exposes **kwargs only; positional arguments rejected before RPC submission'
 failure.write_text(json.dumps(previous,indent=2)+'\n')
else:out.mkdir()
def install(worker):
 import hashlib,inspect,json,os,time
 from pathlib import Path
 receipt=out;records=[]
 for role,w in worker.worker_dict.items():
  if not hasattr(w,'rollout'):continue
  engine=w.rollout.inference_engine
  original=engine.generate;had_instance='generate' in vars(engine);old_instance=vars(engine).get('generate')
  record=dict(pid=os.getpid(),rank=w.rank,role=role,installed_unix=time.time(),calls=[],restored=False,
   original_generate_source=inspect.getsourcefile(original),scope_open_at_install=getattr(w,'_owner_rollout_context_open',False))
  path=receipt/f'rank{w.rank}.json'
  def write():path.write_text(json.dumps(record,indent=2)+'\n')
  def restore():
   if had_instance:engine.generate=old_instance
   else:delattr(engine,'generate')
   record['restored']=True;record['restored_unix']=time.time()
  def observed(*args,**kwargs):
   started=time.perf_counter()
   try:outputs=original(*args,**kwargs)
   except BaseException:
    restore();write();raise
   elapsed=time.perf_counter()-started
   try:
    cached=[getattr(o,'num_cached_tokens',None) for o in outputs]
    row=dict(unix=time.time(),engine_seconds=elapsed,requests=len(outputs),
     prompt_tokens=sum(len(o.prompt_token_ids or []) for o in outputs),
     generated_tokens=sum(len(s.token_ids) for o in outputs for s in o.outputs),
     cached_tokens=sum(cached) if all(n is not None for n in cached) else None,
     cache_metadata_missing=sum(n is None for n in cached),
     lora_ids=sorted(engine.llm_engine.list_loras()),
     owner_scope_open=getattr(w,'_owner_rollout_context_open',False))
    record['calls'].append(row)
    if len(record['calls'])>=8:restore()
    write()
   except Exception as error:
    # Optional observation failure must not replace a valid owner output.
    if not record['restored']:restore()
    record['observation_error']=repr(error)
    try:write()
    except OSError:pass
   return outputs
  engine.generate=observed;write();records.append(dict(pid=record['pid'],rank=w.rank,path=str(path)))
 assert len(records)==1
 return records[0]
ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 actors=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in a['name']]
 assert len(actors)==2
 record=dict(driver_pid=job['pid'],driver_created_unix=driver.create_time(),submitted_unix=time.time(),
  observer_pid=psutil.Process().pid,actors=actors,source_commit=@REVISION@,script_sha256=@SCRIPT_SHA@,
  scope='Eight actual generate calls/rank only; no replay, new trajectory, model or parameter change.')
 (out/'submitted.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
 handles=[ray.get_actor(a['name'],namespace=a['namespace']) for a in actors]
 record['actual_ray_method_signatures']=[repr(a._ray_method_signatures['execute_with_func_generator']) for a in handles]
 refs=[a.execute_with_func_generator.remote(func=install) for a in handles]
 record['installed']=ray.get(refs);record['installed_unix']=time.time()
 (out/'installed.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
finally:ray.shutdown()
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY).replace('@REVISION@',repr(revision)).replace('@SCRIPT_SHA@',repr(script_sha)))
