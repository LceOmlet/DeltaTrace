"""Fetch completed, quiescent source-bound diagnostic files with SHA256."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
remote = transport.ROOT+'/receipts/direct-target-memory-capacity-20261008-v1'
code = r'''
from pathlib import Path
import hashlib,json,psutil
out=Path(@OUT@);launch=json.loads((out/'launch.json').read_bytes())
assert (out/'results/completed.json').exists(),'Do not fetch an incomplete diagnostic as completed'
assert not (psutil.pid_exists(launch['pid']) and psutil.Process(launch['pid']).create_time()==launch['birth']),'Wait for original cleanup before freezing log identities'
files=[]
for p in sorted(out.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts:continue
 b=p.read_bytes();files.append(dict(path=str(p),relative=p.relative_to(out).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
print(json.dumps(files))
'''.replace('@OUT@',repr(remote))
shell = 'set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r = subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True)
if r.returncode:
    print(r.stderr.decode(errors='replace'))
r.check_returncode()
files = json.loads(r.stdout)
for item in files:
    if item['bytes'] > 20_000_000:
        item['local_copy']=False
        continue
    destination = HERE/'actual-results'/Path(*PurePosixPath(item['relative']).parts)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest()!=item['sha256']:
        subprocess.run(transport.SCP+[transport.SSH[-1]+':'+item['path'],str(destination)],capture_output=True,check=True)
    assert hashlib.sha256(destination.read_bytes()).hexdigest()==item['sha256']
    item['local_copy']=str(destination)
(HERE/'actual-results/transport.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps(dict(files=len(files),bytes=sum(item['bytes'] for item in files),destination=str(HERE/'actual-results'))))
