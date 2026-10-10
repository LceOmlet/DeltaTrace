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
preclip=out/'preclip-v1';preclip_files={}
if preclip.exists():
 for name in ['launch.json','query-result.json','query.log','launch-v2.json','query-result-v2.json','query-v2.log']:
  path=preclip/name
  if path.exists():preclip_files[name]=path.read_text(errors='replace')[-12000:]
 for p in psutil.process_iter(['pid','create_time','cmdline','name']):
  try:
   if any(str(preclip/name) in (p.info['cmdline'] or []) for name in ['install_preclip_query.py','install_preclip_query_v2.py']):processes.append(p.info)
  except psutil.Error:pass
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
 workers=workers,preclip_files=preclip_files,host_available_bytes=psutil.virtual_memory().available,production_changes=0)))
'''.replace('ROOT',repr(transport.ROOT),1)
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=45)
r.check_returncode();d=json.loads(r.stdout)
out=HERE/'direct-credit-records-20261009-v1'/('native-actor-observer-observation-'+str(int(d['unix']))+'.json')
out.write_bytes(r.stdout)
summary=[]
for worker in d['workers']:
 parsed={}
 for item in worker['artifacts']:
  if 'text' in item:
   path=Path(item['path'])
   parsed[path.name]=([json.loads(line) for line in item['text'].splitlines()]
       if path.suffix=='.jsonl' else json.loads(item['text']))
 summary.append(dict(pid=worker['pid'],phase=worker['phase'],PSS_bytes=worker['PSS_bytes'],
  native_events=parsed.get('native-events.jsonl',[])[-6:],
  preclip_events=parsed.get('preclip-events.jsonl',[])[-6:],
  preclip_installed='preclip-installation.json' in parsed,
  snapshots=[a for a in worker['artifacts'] if 'bytes' in a]))
preclip_result=json.loads(d['preclip_files']['query-result-v2.json']) if 'query-result-v2.json' in d['preclip_files'] else None
print(json.dumps(dict(saved=str(out),unix=d['unix'],query_processes=d['query_processes'],
 installation_complete=d['installation'].get('complete') if d['installation'] else None,
 preclip_installation_complete=preclip_result.get('complete') if preclip_result else None,
 workers=summary),ensure_ascii=False))
