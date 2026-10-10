"""Read the original workers and the one already submitted observer handle."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code=r'''
import json,psutil,time
from pathlib import Path
root=Path(ROOT);out=root/'receipts/textcraft-native-actor-incidents-20261010-v1'
formal=psutil.Process(982372);assert formal.create_time()==1791553809.84
source=str(out/'capture_current_actor_incident_20261010.py')
processes=[]
for p in psutil.process_iter(['pid','create_time','cmdline','name']):
 try:
  if source in (p.info['cmdline'] or []):processes.append(p.info)
 except psutil.Error:pass
result=json.loads((out/'installation-results.json').read_bytes()) if (out/'installation-results.json').exists() else None
workers=[]
for pid,birth in [(987808,1791553850.00),(989860,1791553867.51)]:
 p=psutil.Process(pid);assert p.create_time()==birth;m=p.memory_full_info()
 folders=list(out.glob('*-pid'+str(pid)));artifacts=[]
 for folder in folders:
  for path in folder.iterdir():
   if path.suffix in ['.json','.jsonl']:
    value=path.read_text(errors='replace')
    artifacts.append(dict(path=str(path),text=value))
   else:artifacts.append(dict(path=str(path),bytes=path.stat().st_size))
 workers.append(dict(pid=pid,birth=birth,phase=p.name(),PSS_bytes=m.pss,RSS_bytes=m.rss,artifacts=artifacts))
print(json.dumps(dict(unix=time.time(),formal_pid=formal.pid,formal_birth=formal.create_time(),
 query_processes=processes,installation=result,query_log=(out/'query.log').read_text(errors='replace')[-6000:] if (out/'query.log').exists() else None,
 workers=workers,host_available_bytes=psutil.virtual_memory().available,production_changes=0)))
'''.replace('ROOT',repr(transport.ROOT),1)
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=45)
r.check_returncode();d=json.loads(r.stdout)
out=HERE/'direct-credit-records-20261009-v1'/('native-actor-observer-observation-'+str(int(d['unix']))+'.json')
out.write_bytes(r.stdout)
print(json.dumps(dict(saved=str(out),unix=d['unix'],query_processes=d['query_processes'],
 installation_complete=d['installation'].get('complete') if d['installation'] else None,
 query_log=d['query_log'],workers=[dict(pid=x['pid'],phase=x['phase'],PSS_bytes=x['PSS_bytes'],
 artifacts=x['artifacts']) for x in d['workers']]),ensure_ascii=False))
