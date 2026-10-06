"""One read-only inspection of the owned row-cut B8 probe; no attachment or RPC."""
import hashlib, importlib.util, json, pathlib, subprocess, time
HERE=pathlib.Path(__file__).resolve().parent
AUDIT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('stage_environment_entry', AUDIT/'stage_environment_entry.py');stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
OUT=stage.ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1/real-b8-v3'
SCRIPT=r'''import datetime,hashlib,json,pathlib,re,subprocess,time
import psutil
out=pathlib.Path(@OUT@);now=time.time()
def source(p):
 raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size=len(raw))
def process_info(p):
 try:
  values=p.as_dict(attrs=['pid','ppid','create_time','status','name','cpu_times','memory_info'])
  for name in ('cpu_times','memory_info'):
   if values.get(name) is not None:values[name]=values[name]._asdict()
  roll=pathlib.Path('/proc')/str(p.pid)/'smaps_rollup'
  values['smaps_rollup_kib']={}
  if roll.exists():
   for line in roll.read_text().splitlines():
    if ':' in line:
     k,v=line.split(':',1)
     if k in ('Rss','Pss','Pss_Anon','Pss_File','Pss_Shmem','Anonymous','Shared_Clean','Shared_Dirty','Private_Clean','Private_Dirty'):
      values['smaps_rollup_kib'][k]=int(v.split()[0])
  return values
 except (psutil.NoSuchProcess,psutil.AccessDenied,OSError) as e:return dict(pid=p.pid,error=type(e).__name__)
job=json.loads((out/'job.json').read_bytes())
expected_pid=2138789;expected_birth=1791323526.94
assert job['pid']==expected_pid and abs(job['pid_birth']-expected_birth)<0.02
result=dict(observed_unix=now,observed_utc=datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),scope='One read-only owned probe log/process/PSS/physical-memory inspection; no model/GPU computation, tracer/RPC, checkpoint or runtime change',out=str(out),job=job,job_file=source(out/'job.json'),prepared_file=source(out/'prepared.json'))
if psutil.pid_exists(expected_pid):
 driver=psutil.Process(expected_pid)
 result['driver_identity_matches']=abs(driver.create_time()-expected_birth)<0.02
 result['processes']=[process_info(p) for p in [driver,*driver.children(recursive=True)]] if result['driver_identity_matches'] else [process_info(driver)]
else:result.update(driver_identity_matches=False,driver_missing=True,processes=[])
ranks={}
for rank in (0,1):
 p=out/f'rank{rank}.json'
 if not p.exists():ranks[str(rank)]=dict(exists=False);continue
 raw=json.loads(p.read_bytes())
 ranks[str(rank)]=dict(exists=True,file=source(p),values={k:raw[k] for k in ('rank','pid','phase','observed_unix','imported_sources','variant','total_wall_seconds','context_lengths','sources','reports','vectors','actual_finite','native_gdn0','native_fa3') if k in raw})
result['ranks']=ranks
log=out/'probe.log'
if log.exists():
 raw=log.read_bytes();tail=raw[-524288:].decode(errors='replace');lines=tail.splitlines();events=[];errors=[]
 for i,line in enumerate(lines):
  j=line.find('{')
  if j>=0:
   try:obj,end=json.JSONDecoder().raw_decode(line[j:])
   except ValueError:obj=None
   if isinstance(obj,dict) and 'phase' in obj:events.append(obj)
  if re.search(r'Traceback|AssertionError|RuntimeError|OutOfMemory|out of memory|CUDA error|MemoryError|ImportError|ModuleNotFoundError',line):
   errors.append(dict(line_in_tail=i,context=lines[max(0,i-3):min(len(lines),i+7)]))
 result['probe_log']=dict(path=str(log),bytes=len(raw),tail_bytes_scanned=min(len(raw),524288),phase_events=events[-40:],error_contexts=errors[-12:],tail_lines=lines[-25:])
else:result['probe_log']=dict(exists=False)
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=20)
result['physical_gpu']=dict(returncode=physical.returncode,stdout=physical.stdout,stderr=physical.stderr)
cgroup={}
for name in ('memory.usage_in_bytes','memory.stat'):
 p=pathlib.Path('/sys/fs/cgroup/memory')/name
 cgroup[str(p)]=p.read_text() if p.exists() else None
result['cgroup']=cgroup
print(json.dumps(result))
'''
script=SCRIPT.replace('@OUT@',repr(OUT))
command="/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python - <<'PY'\n"+script+"\nPY\n"
completed=subprocess.run(stage.SSH+['bash','-s'],input=command.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
completed.check_returncode();result=json.loads(completed.stdout)
result['collector_source']=dict(path=str(pathlib.Path(__file__).resolve()),sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
result['collector_remote_script_sha256']=hashlib.sha256(script.encode()).hexdigest()
path=HERE/f"real-b8-v3-phase-{int(result['observed_unix'])}.json";path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),observed_unix=result['observed_unix'],driver_identity_matches=result['driver_identity_matches'],ranks={k:v.get('values',{}).get('phase') for k,v in result['ranks'].items()},phase_events=result['probe_log'].get('phase_events',[]),errors=result['probe_log'].get('error_contexts',[]),process_count=len(result['processes']))))
