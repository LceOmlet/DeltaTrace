"""Use vLLM's bounded profiler on one existing formal batch per rank.

No replay, new engine, generation loop, scheduler or numerical change. The
public collective_rpc injects the original ProfilerConfig/TorchProfilerWrapper
into the existing worker's profiler slot, then removes it after that call.
"""
from pathlib import Path
import argparse
import hashlib
import subprocess
from stage_environment_entry import ROOT, ENTRY, REPO, remote

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receipt', default=ROOT+'/receipts/owner-b8-dispatch-20260930/appworld-rollout-scope/native-profiler',
    help='Separate observation directory containing the current-PID preflight; never overwrite a previous probe.')
args = parser.parse_args()

revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,psutil,ray,time
root=Path('@ROOT@');out=Path(@RECEIPT@)
read=lambda p:json.loads(p.read_text())
preflight=read(out/'preflight.json')
job=next(j for j in read(root/'active-training.json')['jobs'] if j['task']=='AppWorld')
driver=psutil.Process(job['pid'])
assert driver.pid==preflight['driver_pid'] and driver.create_time()==preflight['driver_created_unix']
assert not (out/'installed.json').exists()
for group in preflight['ranks']:
 for row in group:
  assert len(row['native'])==1
  native=row['native'][0]
  assert native['profiler_is_none'] and set(native['supported_activities'])=={'ProfilerActivity.CPU','ProfilerActivity.CUDA'}
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))

def begin_native(wrapper,directory):
 import hashlib,inspect,os
 from pathlib import Path
 from vllm.config import ProfilerConfig
 from vllm.profiler.wrapper import TorchProfilerWrapper
 # collective_rpc passes WorkerWrapperBase, whose __getattr__ delegates reads
 # but not assignments. The actual owner's profiler slot is on .worker.
 w=wrapper.worker
 assert w.profiler is None
 cfg=ProfilerConfig(profiler='torch',torch_profiler_dir=directory,
     torch_profiler_with_stack=False,torch_profiler_record_shapes=False,
     torch_profiler_with_memory=False,torch_profiler_with_flops=False,
     ignore_frontend=True,max_iterations=16)
 Path(directory).mkdir(exist_ok=True)
 w.profiler=TorchProfilerWrapper(cfg,worker_name=f'formal-{os.getpid()}',
     local_rank=w.local_rank,activities=['CPU','CUDA'])
 w.profile(True)
 sources=[inspect.getsourcefile(ProfilerConfig),inspect.getsourcefile(TorchProfilerWrapper),
          inspect.getsourcefile(w.profile),inspect.getsourcefile(w.annotate_profile)]
 return dict(pid=os.getpid(),config=repr(cfg),running=w.profiler._running,
     sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})

def finish_native(wrapper):
 w=wrapper.worker
 if w.profiler is None:return dict(restored=True,profiler_was_none=True)
 profiler=w.profiler
 report=dict(running_before_cleanup=profiler._running,
     worker_iterations=profiler._active_iteration_count,
     profiler_iterations=profiler._profiling_for_iters,
     original_max_iterations=profiler._max_iters)
 try:w.profile(False)
 finally:w.profiler=None
 report.update(restored=w.profiler is None,running_after_stop=profiler._running)
 return report

def install(worker):
 import inspect,json,os,time
 from pathlib import Path
 records=[]
 for role,w in worker.worker_dict.items():
  if not hasattr(w,'rollout'):continue
  engine=w.rollout.inference_engine;original=engine.generate
  had_instance='generate' in vars(engine);old_instance=vars(engine).get('generate')
  directory=out/f'rank{w.rank}';directory.mkdir(exist_ok=True)
  record=dict(pid=os.getpid(),rank=w.rank,installed_unix=time.time(),restored=False,
      skipped_calls=0,original_generate_source=inspect.getsourcefile(original))
  path=out/f'rank{w.rank}.json'
  def save():path.write_text(json.dumps(record,indent=2)+'\n')
  def restore():
   if had_instance:engine.generate=old_instance
   else:delattr(engine,'generate')
   record['restored']=True;record['restored_unix']=time.time()
  def observed(*args,**kwargs):
   prompts=kwargs.get('prompts',args[0] if args else ())
   # Observe a queued batch, not an isolated singleton. This only selects
   # which existing call to profile; all calls and inputs remain unchanged.
   if len(prompts)<16:
    record['skipped_calls']+=1
    if record['skipped_calls']>=4:restore();save()
    return original(*args,**kwargs)
   try:
    record['begin']=engine.collective_rpc(begin_native,args=(str(directory),))
   except Exception as error:
    record['observation_error']=repr(error)
   started=time.perf_counter()
   try:
    outputs=original(*args,**kwargs)
    record.update(engine_seconds=time.perf_counter()-started,requests=len(outputs),
        prompt_tokens=sum(len(o.prompt_token_ids or []) for o in outputs),
        generated_tokens=sum(len(s.token_ids) for o in outputs for s in o.outputs),
        cached_tokens=[o.num_cached_tokens for o in outputs])
    return outputs
   finally:
    try:record['finish']=engine.collective_rpc(finish_native)
    except Exception as error:record['cleanup_error']=repr(error)
    restore()
    try:save()
    except OSError:pass
  engine.generate=observed;save()
  records.append(dict(pid=os.getpid(),rank=w.rank,record=str(path)))
 assert len(records)==1
 return records[0]

ray.init(address='127.0.0.1:'+port,log_to_driver=False)
try:
 actors=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in a['name']]
 assert len(actors)==2
 record=dict(driver_pid=driver.pid,driver_created_unix=driver.create_time(),submitted_unix=time.time(),
     source_commit=@REVISION@,script_sha256=@SCRIPT_SHA@,
     scope='One existing batch of at least 16 requests per rank; original profiler stops after 16 worker iterations. No additional generation or sampler change.')
 (out/'submitted.json').write_text(json.dumps(record,indent=2)+'\n')
 record['workers']=ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=install) for a in actors])
 record['installed_unix']=time.time()
 (out/'installed.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record,indent=2),flush=True)
finally:ray.shutdown()
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY).replace('@RECEIPT@',repr(args.receipt))
   .replace('@REVISION@',repr(revision)).replace('@SCRIPT_SHA@',repr(script_sha)))
