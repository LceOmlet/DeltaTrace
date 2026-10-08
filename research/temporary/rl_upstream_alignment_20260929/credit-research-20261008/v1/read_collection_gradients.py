"""Read this bounded diagnostic's PID/birth, native phases and measured resources."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import ROOT, ENTRY, SSH

REMOTE = ROOT + '/receipts/credit-collection-gradients-20261008-v1'
body = r'''
import json,os,psutil,subprocess,time
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes())
r={'unix':time.time(),'launch':launch,'host_available':psutil.virtual_memory().available}
p=psutil.Process(launch['pid']) if psutil.pid_exists(launch['pid']) else None
r['driver_same_birth']=p is not None and p.create_time()==launch['birth'];r['processes']=[];r['native_worker_logs']={}
if r['driver_same_birth']:
 for process in [p,*p.children(recursive=True)]:
  try:
   name=process.name();r['processes'].append({'pid':process.pid,'birth':process.create_time(),'name':name,'status':process.status(),'pss_bytes':process.memory_full_info().pss})
   if 'CollectionGradientWorker' in name:
    path=Path(os.readlink(f'/proc/{process.pid}/fd/1'))
    if path.is_file():r['native_worker_logs'][str(process.pid)]={'path':str(path),'tail':path.read_text(errors='replace')[-7000:]}
  except(psutil.NoSuchProcess,psutil.AccessDenied):pass
r['files']={}
for path in [out/'phase.json',out/'completed.json',out/'input-inspection.json',*sorted(out.glob('minibatch*-rank*.json'))]:
 if path.exists():r['files'][path.name]=json.loads(path.read_bytes())
log=out/'driver.log';r['log_tail']=log.read_text(errors='replace')[-15000:] if log.exists() else ''
r['physical']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
for path in [Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'),Path('/sys/fs/cgroup/memory/memory.stat')]:
 if path.exists():r[str(path)]=path.read_text()
root=Path(ROOT);cap=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
r['text_same_birth']=psutil.pid_exists(2833207) and psutil.Process(2833207).create_time()==1791370325.16
r['text_releases']=[(cap/f'rank{i}-release-update').exists() for i in (0,1)]
print(json.dumps(r))
'''.replace('OUT', repr(REMOTE)).replace('ROOT', repr(ROOT))
shell = 'set -eu\nsource ' + ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + body + '\nPY\n'
run = subprocess.run(SSH + ['bash', '-s'], input=shell.encode(), capture_output=True, timeout=45)
if run.returncode:
    print(run.stderr.decode(errors='replace')[-3000:]);run.check_returncode()
value = json.loads(run.stdout)
(HERE / f"collection-gradient-observation-{int(value['unix'])}.json").write_bytes(run.stdout)
print(json.dumps({k: value[k] for k in ('unix', 'driver_same_birth', 'host_available', 'text_same_birth', 'text_releases')}))
print(json.dumps({'phase':value['files'].get('phase.json'),
                  'completed_minibatch_rank_files':[k for k in value['files'] if k.startswith('minibatch')],
                  'process_tree_pss_bytes':sum(p['pss_bytes'] for p in value['processes'])}))
for pid,record in value['native_worker_logs'].items():
    print(json.dumps({'pid':pid,'native_progress':[line for line in record['tail'].splitlines() if 'native_minibatch_diagnostic' in line][-5:]}))
if '--physical' in sys.argv:
    print(value['physical'])
