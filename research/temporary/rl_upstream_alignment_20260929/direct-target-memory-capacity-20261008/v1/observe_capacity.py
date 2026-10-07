"""Read only exact process/phase/physical observations; no task advancement."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code = r'''
import json,psutil,time,subprocess
from pathlib import Path
root=Path(@ROOT@);out=root/'receipts/direct-target-memory-capacity-20261008-v1'
h=json.loads((out/'launch.json').read_bytes());p=psutil.Process(h['pid']) if psutil.pid_exists(h['pid']) else None
r=dict(unix=time.time(),pid=h['pid'],birth=h['birth'],driver_same_birth=p is not None and p.create_time()==h['birth'],completed=(out/'results/completed.json').exists(),ranks=[])
for i in (0,1):
 f=out/'results'/('rank'+str(i)+'-phases.jsonl')
 if f.exists():
  lines=[json.loads(s) for s in f.read_text().splitlines()]
  r['ranks'].append(dict(rank=i,last=lines[-1],boundaries=[s for s in lines if s['phase'] in ('DT_begin','DT_complete','repeat_comparison','real_regression_comparison','failed','complete')]))
r['physical']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
r['log_tail']=(out/'driver.log').read_text(errors='replace')[-5000:]
r['text_same_birth']=psutil.Process(2833207).create_time()==1791370325.16
r['text_released']=[(root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(i)+'-release-update')).exists() for i in (0,1)]
for name in ('memory.usage_in_bytes','memory.stat'):
 r[name]=(Path('/sys/fs/cgroup/memory')/name).read_text()
print(json.dumps(r))
'''.replace('@ROOT@', repr(transport.ROOT))
shell = 'set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
result = subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True)
if result.returncode:
    print(result.stderr.decode(errors='replace'))
result.check_returncode()
record = json.loads(result.stdout)
(HERE/('observation-'+str(int(record['unix']))+'.json')).write_text(json.dumps(record,indent=2)+'\n')
# Full source report is saved; keep direct output to work/phase and physical VRAM.
for rank in record['ranks']:
    rank['last'].pop('owners', None)
    rank['last'].pop('replay_cache', None)
    for phase in rank['boundaries']:
        if 'report' in phase:
            phase['report']={key:phase['report'][key] for key in ('finite_trace_calls','causal_context_lengths','actual_context_lengths')}
record['physical_mib']=[list(map(int, pair)) for pair in re.findall(r'(\d+)/([0-9]+) MiB',record.pop('physical'))]
if record['driver_same_birth'] and not any(rank['last']['phase']=='failed' for rank in record['ranks']):
    record.pop('log_tail',None)
record.pop('memory.stat',None)
print(json.dumps(record))
