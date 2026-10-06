"""Read the actual isolated Adam process, phases, resources and artifacts."""
import json
import subprocess

from stage_environment_entry import SSH, SCP
from stage_textcraft_native_adam import OUT, LOCAL

SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import hashlib,json,pathlib,psutil,subprocess,time
out=pathlib.Path('__OUT__'); job=json.loads((out/'job.json').read_bytes())
alive=False; processes=[]
try:
 parent=psutil.Process(job['pid']); alive=abs(parent.create_time()-job['pid_birth'])<.05
 if alive:
  for proc in [parent]+parent.children(recursive=True):
   try:
    mem=proc.memory_full_info()
    processes.append(dict(pid=proc.pid,birth=proc.create_time(),name=proc.name(),status=proc.status(),rss_bytes=mem.rss,pss_bytes=getattr(mem,'pss',None)))
   except (psutil.NoSuchProcess,psutil.AccessDenied): pass
except psutil.NoSuchProcess: pass
phases=[]
for p in [out/'driver-phase.json',*sorted(out.glob('rank*-phase.json'))]:
 if p.exists(): phases.append(dict(path=str(p),value=json.loads(p.read_bytes())))
completed=(out/'completed.json').exists()
log=out/'diagnostic.log'
artifacts=[dict(name=p.name,bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()]
result=dict(observed_unix=time.time(),pid=job['pid'],pid_birth=job['pid_birth'],same_process_alive=alive,
 completed=completed,processes=processes,phases=phases,artifacts=artifacts,
 host_available_bytes=psutil.virtual_memory().available,
 physical_mx_smi=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout,
 log_tail=log.read_text(errors='replace').splitlines()[-32:] if log.exists() else [])
print(json.dumps(result))
PY
'''


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(SSH + ['bash', '-s'], input=SCRIPT.replace('__OUT__', OUT).encode(), capture_output=True)
    result.check_returncode()
    value = json.loads(result.stdout)
    path = LOCAL / f"observation-{int(value['observed_unix'])}.json"
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), alive=value['same_process_alive'], completed=value['completed'],
        host_available_bytes=value['host_available_bytes'],
        phases=[dict(file=p['path'].rsplit('/',1)[-1],phase=p['value']['phase'],branch=p['value'].get('branch')) for p in value['phases']],
        log_tail=value['log_tail'][-8:]), indent=2))
    if value['completed'] or not value['same_process_alive']:
        suffixes = [suffix for suffix in ('.json', '.yaml', '.log', '.txt')
                    if any(p['name'].endswith(suffix) for p in value['artifacts'])]
        subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/*{suffix}' for suffix in suffixes]
                       + [str(LOCAL)+'/'], capture_output=True, check=True)
