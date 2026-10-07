"""Let only the failed original driver receive Ray's terminal exception/cleanup.

All original actors must already be absent. This never resumes a worker or
submits a replacement job. No checkpoint operation occurs.
"""
from pathlib import Path
import json,subprocess,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH
code=r'''
from pathlib import Path
import hashlib,json,psutil,time
root=Path('@ROOT@');out=root/'receipts/textcraft-token-credit-debug-20261007-v1'
source=root/'runs/direct-action-target-20261007-v3/textcraft/textcraft-dt/source.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='5013ebc878b972a5f52817f7c7de7ae55ddae7f1d61609e943e1ab55bf59b993'
driver=psutil.Process(110053);assert driver.create_time()==1791344324.6
assert driver.status()==psutil.STATUS_STOPPED
for pid in (113691,115709,117229):assert not psutil.pid_exists(pid)
record=dict(observed_unix=time.time(),pid=driver.pid,birth=driver.create_time(),
 operation='Resume ONLY original failed driver to receive existing Ray exception and native cleanup',
 no_actor_resumed=True,no_job_submitted=True,no_checkpoint_operation=True)
driver.resume();record['resume_sent']=True
path=out/f"failed-driver-release-{int(time.time())}.json"
path.write_text(json.dumps(record,indent=2)+'\n');record['remote_receipt']=str(path)
print(json.dumps(record))
'''.replace('@ROOT@',ROOT)
shell='set -e\nsource '+ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+"\nPY\n"
r=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,check=True)
d=json.loads(r.stdout.decode().splitlines()[-1]);p=HERE/'failed-driver-release.json'
p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');print(json.dumps(d))
