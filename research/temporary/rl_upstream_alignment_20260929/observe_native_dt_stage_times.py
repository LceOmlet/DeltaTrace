"""Retain the existing DT stage timers from three real formal calls per rank.

This adds no model replay, CUDA event, synchronization, profiler or numerical
calculation. The original runner already returns its stage timings; the normal
reward report omits them. A temporary pass-through records that result and
restores the original callable after three calls of at least 8k context.
"""
from pathlib import Path
import hashlib
import subprocess

from stage_environment_entry import ENTRY, ROOT, REPO, remote


remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,os,psutil,sys,time,ray
root=Path('@ROOT@')
job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='AppWorld')
for path in reversed([job['entry'],job['verl_root']]):sys.path.insert(0,path)
os.environ['PYTHONPATH']=':'.join([job['entry'],job['verl_root'],os.environ.get('PYTHONPATH','')])
driver=psutil.Process(job['pid'])
assert driver.create_time()==job['observed_process_created_unix']
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
out=root/'receipts/owner-b8-dispatch-20260930'/f'formal-dt-stage-times-{int(time.time())}'
out.mkdir()

def install(worker):
 import hashlib,inspect,json,os,time
 from pathlib import Path
 rows=[]
 for role,w in worker.worker_dict.items():
  if not hasattr(w,'_deltatrace_producer'):continue
  producer=w._deltatrace_producer;runner=producer.runner
  original=runner.attribute
  had_instance='attribute' in vars(runner);old_instance=vars(runner).get('attribute')
  record=dict(pid=os.getpid(),rank=w.rank,role=role,installed_unix=time.time(),
   original_source=inspect.getsourcefile(original),original_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(original)).read_bytes()).hexdigest(),
   original_microbatch=w.actor.config.ppo_micro_batch_size_per_gpu,
   original_lora=[dict(rank=p.r,alpha=p.lora_alpha) for p in w.actor.actor_module.peft_config.values()],
   scope='Existing formal DT runner returned timings only; no model replay, new event/profiler/barrier or numerical change',
   calls=[],restored=False,skipped_short_calls=0)
  path=out/f'rank{w.rank}.json'
  def save():path.write_text(json.dumps(record,indent=2)+'\n')
  def restore():
   if had_instance:runner.attribute=old_instance
   else:delattr(runner,'attribute')
   record['restored']=True;record['restored_unix']=time.time()
  def observed(*args,**kwargs):
   try:
    result=original(*args,**kwargs)
   except BaseException as error:
    record['error']=repr(error);restore()
    try:save()
    except Exception:pass
    raise
   try:
    paired=args[0] if args else kwargs['paired_ids']
    if paired.shape[-1]>=8192:
     info=result[1]
     keys=('complete_attribution_seconds_with_diagnostics','native_shared_prefix_length',
      'gdn_fla_coefficient_start','actual_head_input_shapes','actual_output_bytes',
      'peak_allocated','peak_reserved','root_peak_allocated','calls','controller_diagnostic_scheduling')
     record['calls'].append(dict(unix=time.time(),paired_shape=list(paired.shape),
      **{k:info[k] for k in keys if k in info}))
     if len(record['calls'])==3:restore()
     save()
    else:record['skipped_short_calls']+=1
   except Exception as error:
    # Optional receipt failures cannot turn a valid owner result into a
    # training failure. Stop observation and preserve the original result.
    record['observation_error']=repr(error);restore()
    try:save()
    except Exception:pass
    print('[DT timing observation unavailable] '+repr(error),flush=True)
   return result
  save();runner.attribute=observed
  rows.append(dict(pid=record['pid'],rank=w.rank,path=str(path),installed=True))
 return rows

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 actors=[n for n in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in n['name']]
 assert len(actors)==2
 refs=[ray.get_actor(n['name'],namespace=n['namespace']).execute_with_func_generator.remote(func=install) for n in actors]
 submission=dict(driver_pid=driver.pid,driver_created_unix=driver.create_time(),
  entry=job['entry'],verl_root=job['verl_root'],submitted_unix=time.time(),receipt=str(out),
  diagnostic_commit='@COMMIT@',diagnostic_sha256='@SCRIPT_SHA@',status='queued at original worker RPC boundary')
 (out/'submitted.json').write_text(json.dumps(submission,indent=2)+'\n')
 print(json.dumps(submission),flush=True)
 installed=ray.get(refs)
 (out/'installed.json').write_text(json.dumps(dict(submission,workers=installed),indent=2)+'\n')
 print(json.dumps(dict(receipt=str(out),workers=installed,status='timing observation installed')),flush=True)
finally:ray.shutdown()
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY)
 .replace('@COMMIT@',subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
 .replace('@SCRIPT_SHA@',hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
