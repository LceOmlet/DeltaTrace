"""Read identity-bound original logs and physical resources, without worker RPC."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('existing_stage',HERE.parent/'stage_environment_entry.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
SCRIPT=r'''
from pathlib import Path
import hashlib,json,psutil,re,subprocess,time
root=Path(@ROOT@);jobs=json.loads((root/'active-training.json').read_bytes())['jobs'];out=[]
for job in jobs:
 if job['task'] not in ('TextCraft','AppWorld'):continue
 p=psutil.Process(job['pid']);assert p.create_time()==job['observed_process_created_unix']
 source=Path(job['source_receipt']);observed=dict(job=job,driver_status=p.status(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),processes=[],logs=[])
 files={job['log']}
 for c in [p,*p.children(recursive=True)]:
  try:
   if c.pid!=p.pid and not any(s in c.name() for s in ('TaskRunner','WorkerDict')):continue
   smaps={}
   for line in (Path('/proc')/str(c.pid)/'smaps_rollup').read_text().splitlines():
    if ':' in line:
     key,value=line.split(':',1)
     if key in ('Pss','Pss_Anon','Pss_Shmem','Pss_File'):smaps[key]=int(value.split()[0])
   observed['processes'].append(dict(pid=c.pid,birth=c.create_time(),name=c.name(),status=c.status(),PSS_kib=smaps))
   files.update(f.path for f in c.open_files() if '/worker-' in f.path and f.path.endswith(('.out','.err')))
  except (psutil.Error,OSError):pass
 for name in sorted(files):
  f=Path(name)
  if not f.is_file():continue
  with f.open('rb') as stream:
   stream.seek(max(0,f.stat().st_size-131072));tail=stream.read()
  lines=tail.decode(errors='replace').splitlines()
  selected=[line[:2500] for line in lines if re.search(r'DT EOS plan|DT EOS minibatch|loop_transport|timing_s/|grad_norm|Traceback|RuntimeError:|AssertionError:|OutOfMemory|out of memory|Loading weights|Training Progress|Processed prompts|Rollout collection|Total training steps|Starting|host_cache',line)]
  observed['logs'].append(dict(path=name,bytes=f.stat().st_size,mtime=f.stat().st_mtime,selected=selected[-8:],tail=lines[-3:]))
 out.append(observed)
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=20)
cgroup=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
print(json.dumps(dict(observed_unix=time.time(),jobs=out,physical_gpu=physical.stdout,cgroup_usage_bytes=int(cgroup.read_text()),scope='Original process/log/PSS/physical GPU reads only; no RPC, profiler, model, checkpoint or source mutation')))
'''
if __name__=='__main__':
 script=stage.ROOT+'/../deltatrace_qwen35_20260912/env/bin/python'
 command=script+" - <<'PY'\n"+SCRIPT.replace('@ROOT@',repr(stage.ROOT))+"\nPY\n"
 r=subprocess.run(stage.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=65)
 if r.returncode:print(r.stderr.decode(errors='replace'));r.check_returncode()
 x=json.loads(r.stdout);x['collector_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 folder=HERE/'fresh-row-prefix-v2'/'deployment'
 output=folder/('phase-'+str(int(x['observed_unix']))+'.json');output.write_text(json.dumps(x,indent=2)+'\n')
 print(json.dumps(dict(receipt=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),jobs=[dict(task=j['job']['task'],pid=j['job']['pid'],processes=j['processes'],logs=[dict(path=f['path'],selected=f['selected']) for f in j['logs']]) for j in x['jobs']])))
