"""Read native PyTorch allocator counters at the existing VERL worker RPC seam."""
import argparse
import subprocess
import time

from stage_environment_entry import SSH, ROOT, ENTRY, AUDIT


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', choices=['AppWorld', 'TextCraft'], required=True)
parser.add_argument('--release-unused-cache', action='store_true',
                    help='Call the installed native _host_emptyCache once; no persistent hook.')
args = parser.parse_args()

script = r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,psutil,sys,time,ray
root=Path('@ROOT@')
job=next(j for j in json.loads((root/'active-training.json').read_text())['jobs'] if j['task']=='@TASK@')
driver=psutil.Process(job['pid'])
assert driver.create_time()==job['observed_process_created_unix']
for path in reversed([job['entry'],job['verl_root']]):sys.path.insert(0,path)
os.environ['PYTHONPATH']=':'.join([job['entry'],job['verl_root'],os.environ.get('PYTHONPATH','')])
gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
directory=root/'receipts/host-memory-20261002'/f"native-allocator-@TASK@-{int(time.time())}"
directory.mkdir(parents=True,exist_ok=False)

def observe(worker):
 import os,time,torch,json,hashlib
 from pathlib import Path
 result=dict(pid=os.getpid(),unix=time.time(),torch_version=str(torch.__version__),
  cuda_initialized=torch.cuda.is_initialized(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
  scope='Native allocator read only; no synchronization, allocation, cache clearing, reset, model call, or training setting mutation')
 if @RELEASE@:
  result['scope']='One native unused-host-cache release at the existing RPC seam; no persistent hook, model call, reset, or training configuration mutation'
 result['native_host_memory_bindings']=[name for name in dir(torch._C) if 'host' in name.lower() and any(key in name.lower() for key in ('cache','alloc','memory'))]
 try:
  result['pinned_host_stats']=dict(torch.cuda.memory.host_memory_stats())
  result['device_allocator_stats']=dict(torch.cuda.memory_stats())
  if @RELEASE@:
   start=time.perf_counter()
   torch._C._host_emptyCache()
   result['native_cache_release_seconds']=time.perf_counter()-start
   result['pinned_host_stats_after_release']=dict(torch.cuda.memory.host_memory_stats())
 except Exception as error:
  result['native_stats_error']=repr(error)
 source=Path(torch.cuda.memory.__file__)
 result['native_memory_source']=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
 result['smaps_rollup']=Path('/proc/self/smaps_rollup').read_text()
 result['roles']=list(worker.worker_dict)
 out=directory/f"worker-{os.getpid()}.json"
 out.write_text(json.dumps(result,indent=2)+'\n')
 return result

# Only read through the existing RPC; no polling, sampler, or interruption.
ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
 actors=[n for n in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in n['name']]
 assert len(actors)==2,actors
 refs=[ray.get_actor(n['name'],namespace=n['namespace']).execute_with_func_generator.remote(func=observe) for n in actors]
 submitted=dict(task=job['task'],driver_pid=driver.pid,driver_created_unix=driver.create_time(),
  submitted_unix=time.time(),directory=str(directory),actors=actors,release_unused_cache=@RELEASE@,status='queued native allocator callback')
 (directory/'submitted.json').write_text(json.dumps(submitted,indent=2)+'\n')
 print(json.dumps(submitted),flush=True)
 ready,pending=ray.wait(refs,num_returns=len(refs),timeout=15)
 print(json.dumps(dict(ready=len(ready),pending=len(pending),workers=ray.get(ready)),indent=2),flush=True)
finally:ray.shutdown()
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY).replace('@TASK@', args.task).replace('@RELEASE@', repr(args.release_unused_cache))
result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True, timeout=55)
receipt = AUDIT / f'native-allocator-{args.task.lower()}-{"release" if args.release_unused_cache else "submitted"}-20261002-{int(time.time())}.txt'
receipt.write_bytes(result.stdout + result.stderr)
print(result.stdout.decode(errors='replace'))
print(result.stderr.decode(errors='replace'))
print('Local receipt:', receipt)
raise SystemExit(result.returncode)
