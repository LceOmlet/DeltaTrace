"""Preserve the actual finite-FA build source identity, without GPU calls."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import SSH, ROOT

script = r'''python3 - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('ROOT/candidates/appworld-row-cuts-finite-20261007-v1')
files=[]
for f in sorted(p.iterdir()):
    if f.is_file():
        raw=f.read_bytes() if f.suffix in ('.json','.cu','.cpp','.sh') or f.name=='libfinite_row_query_starts.so' else None
        row=dict(path=str(f),bytes=f.stat().st_size)
        if raw is not None: row['sha256']=hashlib.sha256(raw).hexdigest()
        if raw is not None and f.suffix in ('.json','.cu','.cpp','.sh') and len(raw)<300000: row['text']=raw.decode()
        files.append(row)
print(json.dumps(dict(scope='Immediate actual production build directory only; no recursive scan or GPU',files=files)))
PY
'''.replace('ROOT', ROOT)
(HERE/'finite-FA-source-command.sh').write_text(script, encoding='utf-8', newline='\n')
r = subprocess.run(SSH+['bash','-s'], input=script.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
(HERE/'finite-FA-source.stderr').write_bytes(r.stderr)
if r.returncode:
    raise SystemExit(r.returncode)
data=json.loads(r.stdout)
(HERE/'finite-FA-source.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(path=x['path'],bytes=x['bytes'],sha256=x.get('sha256')) for x in data['files']]))
