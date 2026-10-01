"""Observe three existing DT or vLLM returns at an original worker RPC boundary.

No extra model call, event, barrier, profiler or numerical transformation is
added. The returned object is passed through unchanged; the observation hook
removes itself before its third call. Measurement I/O cannot fail training.
"""
import argparse
from stage_environment_entry import remote, ROOT, ENTRY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['SkyRL-SQL', 'AppWorld', 'TextCraft'], required=True)
    parser.add_argument('--kind', choices=['dt', 'generation'], default='dt')
    args = parser.parse_args()
    remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,psutil,time,ray
root=Path('@ROOT@')
job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='@TASK@')
driver=psutil.Process(job['pid'])
assert driver.create_time()==job['observed_process_created_unix']
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
out=root/'receipts/owner-b8-dispatch-20260930'/f'native-@KIND@-times-{job["task"].lower()}-{int(time.time())}'
out.mkdir()

def arm(worker):
 import functools,hashlib,inspect,json,os,time
 from pathlib import Path
 records=[]
 for role,w in worker.worker_dict.items():
  if '@KIND@'=='dt':
   if not hasattr(w,'_deltatrace_producer'):continue
   runner=w._deltatrace_producer.runner;method='attribute'
  else:
   if not hasattr(w,'rollout'):continue
   runner=w.rollout.inference_engine;method='generate'
  original=getattr(runner,method)
  if getattr(original,'_native_times_capture',False):
   records.append(dict(pid=os.getpid(),role=role,status='already armed'));continue
  source=inspect.getsourcefile(original)
  origin=dict(file=source,sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
  # One actor role in the current WorkerDict; each closure binds its own runner.
  def make_capture(runner,method,original,origin,role):
   remaining=3
   @functools.wraps(original)
   def capture(*args,**kwargs):
    nonlocal remaining
    remaining-=1
    if remaining==0:setattr(runner,method,original)
    started=time.perf_counter()
    result=original(*args,**kwargs)
    elapsed=time.perf_counter()-started
    try:
     if '@KIND@'=='dt':
      info=result[1]
      selected={key:info[key] for key in (
       'calls','complete_attribution_seconds_with_diagnostics',
       'native_shared_prefix_length','fa_coefficient_starts','gdn_fla_coefficient_start',
       'actual_head_input_shapes','actual_output_bytes','controller_diagnostic_scheduling') if key in info}
      selected['paired_input_shape']=list(args[0].shape)
     else:
      import dataclasses
      selected=dict(engine_generate_seconds=elapsed,requests=[dict(
       prompt_tokens=len(item.prompt_token_ids),cached_tokens=item.num_cached_tokens,
       generated_tokens=sum(len(x.token_ids) for x in item.outputs),
       metrics=dataclasses.asdict(item.metrics) if dataclasses.is_dataclass(item.metrics) else None)
       for item in result])
     record=dict(pid=os.getpid(),role=role,unix=time.time(),source=origin,
      kind='@KIND@',remaining=remaining,
      restored=getattr(runner,method)==original,info=selected)
     target=out/f'{os.getpid()}-{3-remaining}.json'
     target.write_text(json.dumps(record,indent=2)+'\n')
     print('[Native @KIND@ timing receipt] '+str(target),flush=True)
    except Exception as error:
     print('[Native @KIND@ timing unavailable] '+repr(error),flush=True)
    return result
   capture._native_times_capture=True
   return capture
  setattr(runner,method,make_capture(runner,method,original,origin,role))
  records.append(dict(pid=os.getpid(),rank=w.rank,role=role,source=origin,status='armed for three existing returns'))
 return records

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 names=[n for n in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in n['name']]
 assert len(names)==2,names
 refs=[ray.get_actor(n['name'],namespace=n['namespace']).execute_with_func_generator.remote(arm) for n in names]
 print(json.dumps(dict(task=job['task'],driver_pid=driver.pid,receipt_dir=str(out),status='queued at original RPC boundary')),flush=True)
 record=dict(task=job['task'],driver_pid=driver.pid,driver_created_unix=driver.create_time(),
  receipt_dir=str(out),workers=ray.get(refs),scope='Original runner return timing only; unchanged input/output and execution; no new CUDA events or synchronization.')
 (out/'armed.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record),flush=True)
finally:ray.shutdown()
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY).replace('@TASK@', args.task).replace('@KIND@', args.kind))


if __name__ == '__main__':
    main()
