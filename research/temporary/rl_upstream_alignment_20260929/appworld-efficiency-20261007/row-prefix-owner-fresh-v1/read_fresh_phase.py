"""Read only the identity-bound fresh AppWorld logs and physical resources."""
import hashlib,importlib.util,json,pathlib,subprocess
HERE=pathlib.Path(__file__).resolve().parent
AUDIT=HERE.parents[1]
spec=importlib.util.spec_from_file_location("stage",AUDIT/"stage_environment_entry.py")
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
SCRIPT=r"""
from pathlib import Path
import collections,hashlib,json,psutil,re,subprocess,time
root=Path(@ROOT@);job=next(j for j in json.loads((root/'active-training.json').read_bytes())['jobs'] if j['task']=='AppWorld')
assert job['pid']==2360541 and job['observed_process_created_unix']==1791325655.01
p=psutil.Process(job['pid']);assert p.create_time()==job['observed_process_created_unix']
source=Path(job['source_receipt']);assert hashlib.sha256(source.read_bytes()).hexdigest()=='c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789'
workers=[]
for c in [p,*p.children(recursive=True)]:
 try:
  if c.pid!=p.pid and not any(s in c.name() for s in ('TaskRunner','WorkerDict','gcs_server')):continue
  fields={}
  for line in (Path('/proc')/str(c.pid)/'smaps_rollup').read_text().splitlines():
   if ':' in line:
    k,v=line.split(':',1)
    if k in ('Pss','Pss_Anon','Pss_Shmem','Pss_File'):fields[k]=int(v.split()[0])
  workers.append(dict(pid=c.pid,birth=c.create_time(),name=c.name(),status=c.status(),smaps_rollup_kib=fields))
 except (psutil.Error,OSError):pass
files=[Path(job['log'])]
ids={str(x['pid']) for x in workers}
for session in Path('/tmp/ray').glob('*_'+str(job['pid'])):
 for path in (session/'logs').glob('worker-*'):
  if path.suffix in ('.out','.err') and path.stem.rsplit('-',1)[-1] in ids:files.append(path)
logs=[]
for f in files:
 if not f.is_file():continue
 raw=f.read_bytes();tail=raw[-524288:];lines=tail.decode(errors='replace').splitlines()
 selected=[line[:3500] for line in lines if re.search(r'DT EOS plan|DT EOS minibatch|DeltaTrace readout|native_host_cache|loop_transport|Rollout collection|timing_s/|grad_norm|Traceback|RuntimeError:|AssertionError:|OutOfMemory|out of memory|init_model|load.*weights|Loading|Completion|Starting|gcs_server|worker',line)]
 logs.append(dict(path=str(f),bytes=len(raw),sha256_of_observed_bytes=hashlib.sha256(raw).hexdigest(),mtime=f.stat().st_mtime,last_selected_lines=selected[-12:],tail_lines=[line[:3500] for line in lines[-8:]]))
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=20)
cgroup={}
for name in ('memory.usage_in_bytes','memory.stat'):
 f=Path('/sys/fs/cgroup/memory')/name;cgroup[name]=f.read_text() if f.exists() else None
print(json.dumps(dict(observed_unix=time.time(),job=job,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),processes=workers,logs=logs,physical_gpu=dict(returncode=physical.returncode,stdout=physical.stdout,stderr=physical.stderr),cgroup=cgroup,scope='Read-only identity-bound fresh formal logs/process PSS/physical resources; no model/RPC/profiler/checkpoint or runtime mutation')))
"""
r=subprocess.run(stage.SSH+['bash','-s'],input=("/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python - <<'PY'\n"+SCRIPT.replace('@ROOT@',repr(stage.ROOT))+"\nPY\n").encode(),capture_output=True,timeout=65)
if r.returncode:print(r.stderr.decode(errors='replace'));r.check_returncode()
x=json.loads(r.stdout);x['collector_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
out=HERE/('phase-'+str(int(x['observed_unix']))+'.json');out.write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(dict(receipt=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),observed_unix=x['observed_unix'],processes=x['processes'],logs=[dict(path=v['path'],tail_lines=v['tail_lines']) for v in x['logs']])))
