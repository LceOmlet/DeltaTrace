"""Fetch completed layer diagnostic with remote/local SHA256 verification."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
remote = transport.ROOT + '/receipts/direct-target-extreme-operator-20261007-v1'
destination = HERE / 'actual-results'
code = '''from pathlib import Path
import hashlib,json
root=Path(@REMOTE@)
assert (root/'results/completed.json').is_file()
rows=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts:
  data=p.read_bytes();rows.append(dict(path=str(p),relative=p.relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(rows))
'''.replace('@REMOTE@', repr(remote))
shell = 'set -eu\nsource ' + transport.ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
response = subprocess.run(transport.SSH + ['bash', '-s'], input=shell.encode(), capture_output=True, check=True)
rows = json.loads(response.stdout)
for row in rows:
    target = destination / Path(*PurePosixPath(row['relative']).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
        subprocess.run(transport.SCP + [transport.SSH[-1] + ':' + row['path'], str(target)], capture_output=True, check=True)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256']
    row['local_path'] = str(target)
(destination / 'transport.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(dict(files=len(rows), bytes=sum(row['bytes'] for row in rows), output=str(destination / 'transport.json'))))
