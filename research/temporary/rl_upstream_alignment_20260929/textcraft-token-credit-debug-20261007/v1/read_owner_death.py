"""Read the original Ray owner-death interval, without altering processes."""
from pathlib import Path
import json,subprocess,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH
code=r'''
from pathlib import Path
import hashlib,json,time
base=Path('/tmp/ray/session_2026-10-07_11-39-00_016748_110053/logs')
paths=[base/'raylet.out',base/'raylet.err',base/'gcs_server.out']+list(base.glob('*113691*'))
record=dict(observed_unix=time.time(),scope='Original owner-death lines only; no process action',files=[])
for p in paths:
 if not p.is_file():continue
 rows=[]
 for number,line in enumerate(p.open(errors='replace'),1):
  time_window=any(s in line for s in ['2026-10-07 17:39:','2026-10-07 17:40:'])
  owner=('113691' in line or 'f267bf740750cb67e80c1bfac3fdcc46c7eca863cf7df85e23c27496' in line)
  if time_window or owner:rows.append(dict(line=number,text=line.rstrip()))
 record['files'].append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),lines=rows[-250:]))
out=Path('@ROOT@')/'receipts/textcraft-token-credit-debug-20261007-v1'
p=out/f"owner-death-{int(time.time())}.json";p.write_text(json.dumps(record,indent=2)+'\n')
record['remote_receipt']=str(p);print(json.dumps(record))
'''.replace('@ROOT@',ROOT)
shell='set -e\nsource '+ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+"\nPY\n"
r=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,check=True)
d=json.loads(r.stdout.decode().splitlines()[-1]);p=HERE/f"owner-death-{int(d['observed_unix'])}.json"
p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(receipt=str(p),files=d['files']),ensure_ascii=False))
