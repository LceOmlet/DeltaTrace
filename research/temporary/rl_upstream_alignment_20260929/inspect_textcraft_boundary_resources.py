"""Read current resources and the recorded environment provider, without setup."""
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ROOT, SSH


SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import hashlib,json,pathlib,psutil,subprocess,time
root=pathlib.Path('__ROOT__')
base=root/'receipts/textcraft-native-minibatch-20261006-v4'
job=json.loads((base/'job.json').read_bytes())
pid=job['reused_environment_pid']; birth=job['reused_environment_pid_birth']
p=psutil.Process(pid)
assert abs(p.create_time()-birth)<.05
env=p.environ()
usage=pathlib.Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
files={}
for name in ['active-training.json','active-source.json']:
 f=root/name
 files[name]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size)
mem=p.memory_full_info()
report=dict(observed_unix=time.time(),environment_provider=dict(pid=pid,birth=p.create_time(),
 status=p.status(),rss_bytes=mem.rss,pss_bytes=getattr(mem,'pss',None),
 python=env.get('VENV_PYTHON'),dt_root=env.get('DT_ROOT'),verl_root=env.get('VERL_ROOT')),
 authoritative_files=files,host_available_bytes=psutil.virtual_memory().available,
 cgroup_usage_bytes=int(usage.read_text()) if usage.is_file() else None,
 physical_mx_smi=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout)
print(json.dumps(report))
PY
'''


if __name__=='__main__':
    completed=subprocess.run(SSH+['bash','-s'],input=SCRIPT.replace('__ROOT__',ROOT).encode(),capture_output=True)
    completed.check_returncode()
    data=json.loads(completed.stdout)
    out=AUDIT/'textcraft-degradation-20261005/conditional-boundaries-20261006/v1'
    out.mkdir(parents=True,exist_ok=True)
    path=out/f"resources-{int(data['observed_unix'])}.json"
    path.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),host_available_bytes=data['host_available_bytes'],
        environment_provider=data['environment_provider'],physical=data['physical_mx_smi']),indent=2))
