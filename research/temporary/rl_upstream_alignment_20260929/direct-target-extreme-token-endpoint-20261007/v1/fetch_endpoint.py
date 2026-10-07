"""Copy a completed original endpoint diagnostic and verify its hashes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('transport',AUDIT/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['textcraft','appworld'],required=True)
args=parser.parse_args()
remote=transport.ROOT+'/receipts/direct-target-extreme-token-endpoint-20261007-v1/'+args.case
code='''from pathlib import Path
import hashlib,json
root=Path(@ROOT@)
assert (root/'completed.json').exists(),'Endpoint diagnostic not completed'
print(json.dumps([dict(path=str(p),name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.iterdir()) if p.is_file()]))
'''.replace('@ROOT@',repr(remote))
shell='set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
result=subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True)
if result.returncode:print(result.stderr.decode(errors='replace'))
result.check_returncode();rows=json.loads(result.stdout)
out=HERE/args.case;out.mkdir(exist_ok=True)
for row in rows:
    target=out/row['name']
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:
        subprocess.run(transport.SCP+[transport.SSH[-1]+':'+row['path'],str(target)],check=True,capture_output=True)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
    row['local_path']=str(target)
(out/'transport.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(case=args.case,files=len(rows),bytes=sum(x['bytes'] for x in rows),transport=str(out/'transport.json'))))
