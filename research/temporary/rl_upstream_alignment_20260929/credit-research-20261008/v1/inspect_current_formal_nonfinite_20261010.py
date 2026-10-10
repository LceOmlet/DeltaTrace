"""Read the current native worker incident and its existing capture facilities.

No model imports, worker RPC, changes to training, or remote writes.
"""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
remote = r'''
import hashlib,json,psutil,re,time
from pathlib import Path
root=Path(ROOT);formal=root/'runs/textcraft-formal-stable-20261009-v1'
driver=psutil.Process(982372);assert driver.create_time()==1791553809.84
source=json.loads((formal/'source.json').read_bytes())
ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs')
ansi=re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
trainer=next(ray.glob('*-985585.out'))
steps=[]
for line in ansi.sub('',trainer.read_text(errors='replace')).splitlines():
 if re.search(r'step:\d+ - ',line):
  text=line[line.index('step:'):]
  metrics={key:float(value) for key,value in re.findall(r'([\w/]+):(-?\d+(?:\.\d+)?|nan|inf|-inf)',text)}
  steps.append(dict(step=int(metrics['step']),original=text,metrics=metrics))
workers=[]
for pid,birth in [(987808,1791553850.00),(989860,1791553867.51)]:
 p=psutil.Process(pid);assert p.create_time()==birth
 m=p.memory_full_info();warnings=[]
 for suffix in ['out','err']:
  path=next(ray.glob('*-'+str(pid)+'.'+suffix))
  text=path.read_text(errors='replace');lines=text.splitlines()
  for i,line in enumerate(lines):
   if 'grad_norm is not finite' in line:
    previous=[x for x in lines[:i] if x.startswith('[DT direct joint record]')]
    after=[x for x in lines[i+1:] if x.startswith('[DT direct joint record]')]
    warnings.append(dict(path=str(path),line=i+1,text=line,
      previous_joint=json.loads(previous[-1].split('] ',1)[1]) if previous else None,
      next_joint=json.loads(after[0].split('] ',1)[1]) if after else None))
 workers.append(dict(pid=pid,birth=birth,phase=p.name(),PSS_bytes=m.pss,RSS_bytes=m.rss,
  warning_count=len(warnings),warnings=warnings,
  diagnostic_environment={k:v for k,v in p.environ().items()
    if k.startswith(('DT_ACTOR_INCIDENT','DT_NONFINITE','DT_STOP_AFTER','DT_HOLD'))}))
verldir=Path(source['verl_root']);entry=formal/'entry'
files=[]
for path in [verldir/'verl/workers/actor/dp_actor.py',verldir/'verl/workers/fsdp_workers.py',
             verldir/'verl/trainer/ppo/ray_trainer.py']+list(entry.glob('*.py')):
 text=path.read_text(errors='replace');lines=text.splitlines();hits=[]
 for i,line in enumerate(lines):
  if any(x in line.lower() for x in ['actor_incident','nonfinite','non_finite','not finite','torch.save','dump_debug','save_batch']):
   hits.append(dict(line=i+1,context=lines[max(0,i-3):i+6]))
 if hits:files.append(dict(path=str(path),resolved=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hits=hits))
capture_dirs=[]
for path in formal.iterdir():
 if path.is_dir() and path.name not in ['credit-records','textcraft-dt','entry']:
  capture_dirs.append(dict(path=str(path),entries=[dict(name=p.name,bytes=p.stat().st_size,is_file=p.is_file()) for p in path.iterdir()][:40]))
out=dict(unix=time.time(),driver=dict(pid=driver.pid,birth=driver.create_time()),
 latest_step=steps[-1]['step'] if steps else None,steps=steps,workers=workers,
 source_path=str(formal/'source.json'),source_sha256=hashlib.sha256((formal/'source.json').read_bytes()).hexdigest(),
 upstream_commit=source['upstream_commit'],verl_root=str(verldir),entry=str(entry),
 diagnostic_source_hits=files,capture_directories=capture_dirs,
 host_available_bytes=psutil.virtual_memory().available,
 production_changes=0,model_calls=0,worker_RPC=0)
print(json.dumps(out))
'''.replace('ROOT',repr(transport.ROOT),1)
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+remote+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=50)
out=HERE/'direct-credit-records-20261009-v1/current-formal-nonfinite-inspection-20261010.json'
out.with_suffix('.stderr').write_bytes(r.stderr)
r.check_returncode()
d=json.loads(r.stdout);out.write_bytes(r.stdout)
print(json.dumps(dict(saved=str(out),unix=d['unix'],latest_step=d['latest_step'],
 nonfinite_steps=[x['step'] for x in d['steps'] if 'actor/grad_norm:nan' in x['original']],
 workers=[{k:x[k] for k in ['pid','birth','phase','warning_count','diagnostic_environment']} for x in d['workers']],
 capture_source_files=[x['path'] for x in d['diagnostic_source_hits']],
 capture_directories=[x['path'] for x in d['capture_directories']],
 host_available_bytes=d['host_available_bytes'])))
