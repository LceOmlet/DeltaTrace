"""Copy completed diagnostic artifacts and verify their source SHA256."""
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('transport', AUDIT/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['3','5','6'],default='3')
args=parser.parse_args()
REMOTE=transport.ROOT+'/receipts/direct-target-native-mlp-memory-20261007-v'+args.version
DESTINATION=HERE/('actual-results' if args.version=='3' else 'actual-results-v'+args.version)
code='''from pathlib import Path
import hashlib,json
root=Path(@REMOTE@)
assert (root/'results/completed.json').exists(),'Diagnostic not complete; do not fetch changing phase logs'
rows=[]
for path in sorted(root.rglob('*')):
 if not path.is_file() or path.parent.name=='failed-inputs' and path.suffix=='.pt':continue
 data=path.read_bytes();rows.append(dict(path=str(path),relative=path.relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(rows))
'''.replace('@REMOTE@',repr(REMOTE))
shell='set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
result=subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True)
if result.returncode:
    print(result.stderr.decode(errors='replace'))
result.check_returncode()
rows=json.loads(result.stdout)
for row in rows:
    path=DESTINATION/Path(*PurePosixPath(row['relative']).parts)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        result=subprocess.run(transport.SCP+[transport.SSH[-1]+':'+row['path'],str(path)],capture_output=True)
        result.check_returncode()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    row['local_path']=str(path)
(DESTINATION/'transport.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(files=len(rows),bytes=sum(row['bytes'] for row in rows),output=str(DESTINATION/'transport.json'))))
