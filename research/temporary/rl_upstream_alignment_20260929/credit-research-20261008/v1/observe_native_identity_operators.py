"""Read the one launched diagnostic handle and its original owner receipts."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import SSH,ROOT,ENTRY


def main():
    version=sys.argv[1] if len(sys.argv)>1 else 'v2'
    assert version in ('v1','v2')
    out=HERE/('native-identity-operators-appworld-'+version)
    body=r'''import json,psutil,subprocess,time
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes())
try:
 driver=psutil.Process(launch['pid'])
 alive=abs(driver.create_time()-launch['birth'])<0.01 and driver.is_running() and driver.status()!=psutil.STATUS_ZOMBIE
except psutil.NoSuchProcess:alive=False
record=dict(unix=time.time(),driver_alive=alive,driver_identity=dict(pid=launch['pid'],birth=launch['birth']),files={},processes=[])
for path in sorted((out/'results').glob('*.json')):
 record['files'][str(path.relative_to(out))]=json.loads(path.read_bytes())
for process in psutil.process_iter(['pid','create_time','cmdline']):
 try:
  cmd=' '.join(process.info.get('cmdline') or [])
  if str(out) not in cmd or process.pid==__import__('os').getpid():continue
  mem=process.memory_full_info()
  record['processes'].append(dict(pid=process.pid,birth=process.create_time(),pss=mem.pss,rss=mem.rss,status=process.status()))
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
record['physical']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
record['host']=dict(psutil.virtual_memory()._asdict())
record['disk']=dict(psutil.disk_usage(out)._asdict())
p=out/'driver.log'
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-14000));record['driver_log_tail']=f.read().decode(errors='replace')
print(json.dumps(record))
'''
    target=ROOT+'/receipts/credit-native-identity-operators-appworld-20261009-'+version
    body='OUT='+repr(target)+'\n'+body
    command='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
    result=subprocess.run(SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=45)
    (out/'last-observation.stderr').write_bytes(result.stderr)
    result.check_returncode()
    observation=json.loads(result.stdout)
    path=out/f"observation-{int(observation['unix'])}.json"
    path.write_bytes((json.dumps(observation,indent=2)+'\n').encode())
    for name,content in observation['files'].items():
        destination=out/Path(name).name
        destination.write_bytes((json.dumps(content,indent=2)+'\n').encode())
    print(json.dumps(dict(path=str(path),unix=observation['unix'],driver_alive=observation['driver_alive'],
        phases={k:v.get('phase') for k,v in observation['files'].items() if isinstance(v,dict) and k.endswith(('rank0.json','rank1.json'))},
        receipts=list(observation['files']),physical=observation['physical'],
        log_tail=observation['driver_log_tail'][-3500:]),indent=2))


if __name__=='__main__':main()
