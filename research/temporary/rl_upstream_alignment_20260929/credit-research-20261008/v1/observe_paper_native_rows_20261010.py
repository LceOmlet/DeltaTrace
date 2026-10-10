"""Read the launched paper-forward handle and preserve its original files."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH

OUT = HERE/'paper-native-rows-20261010-v1'
body = r'''
import hashlib,json,psutil,subprocess,time
from pathlib import Path
out=Path(REMOTE);launch=json.loads((out/'launch.json').read_bytes())
try:
 p=psutil.Process(launch['pid'])
 alive=abs(p.create_time()-launch['birth'])<.01 and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
 process=dict(pid=p.pid,birth=p.create_time(),status=p.status(),memory=dict(p.memory_full_info()._asdict()))
except psutil.NoSuchProcess:alive=False;process=None
files={}
for path in sorted(out.glob('*.json')):
 data=path.read_bytes();files[path.name]=dict(text=data.decode(),sha256=hashlib.sha256(data).hexdigest())
record=dict(unix=time.time(),alive=alive,process=process,files=files,
 physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout,
 host=dict(psutil.virtual_memory()._asdict()),disk=dict(psutil.disk_usage(out)._asdict()))
p=out/'driver.log'
if p.exists():
 data=p.read_bytes();record['driver_log_tail']=data[-9000:].decode(errors='replace')
print(json.dumps(record))
'''
body = 'REMOTE='+repr(ROOT+'/receipts/paper-native-rows-20261010-v1')+'\n'+body
shell = 'source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
run = subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=45)
run.check_returncode()
d = json.loads(run.stdout)
path = OUT/f"observation-{int(d['unix'])}.json"
path.write_bytes((json.dumps(d,indent=2)+'\n').encode())
for name,content in d['files'].items():
    data=content['text'].encode()
    assert hashlib.sha256(data).hexdigest()==content['sha256']
    (OUT/name).write_bytes(data)
result=json.loads(d['files'].get('result.json',{}).get('text','{}'))
print(json.dumps(dict(observation=str(path),unix=d['unix'],alive=d['alive'],
    phase=result.get('phase'),native_forward_calls=result.get('native_forward_calls'),
    first_unequal=result.get('operator_observation',{}).get('first_unequal_by_row'),
    paired_score_differences=result.get('paired_score_differences'),
    maximum_score_difference_vs_previous=result.get('maximum_score_difference_vs_previous'),
    resources=result.get('resources'),physical=d['physical'],
    log_tail=d.get('driver_log_tail','')[-4500:]),indent=2))
