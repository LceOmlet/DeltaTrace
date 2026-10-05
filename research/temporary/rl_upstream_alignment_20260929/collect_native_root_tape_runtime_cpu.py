"""Analyze existing real trace runtime events without another GPU run."""
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, SCP, SSH


source=REPO/'experiments/rl/results_native_root_tape_hot_20261004.json'
raw=json.loads(source.read_bytes())
assert raw['completed'] and raw['driver']['status']=='exited'
out=raw['remote_root']
analyzer=AUDIT/'analyze_native_hot_runtime_cpu.py'
subprocess.run(SCP+[str(analyzer),f'{SSH[-1]}:{out}/{analyzer.name}'],check=True)
script=r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import hashlib,importlib.util,json,pathlib
out=pathlib.Path(@OUT@);path=out/'analyze_native_hot_runtime_cpu.py'
spec=importlib.util.spec_from_file_location('original_runtime_intervals',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
prepared=json.loads((out/'prepared.json').read_bytes())
environment=prepared['configuration']['numerical_environment']
env_path=pathlib.Path(environment['path'])
assert hashlib.sha256(env_path.read_bytes()).hexdigest()==environment['sha256']
value=dict(scope='CPU-only reading of original real root-tape B4 traces; no new profile, GPU call or model',
 source=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
 numerical_environment=dict(**environment,content=json.loads(env_path.read_bytes())),traces=[])
for rank in (0,1):
 record=json.loads((out/f'rank{rank}.json').read_bytes())
 trace=pathlib.Path(record['trace']['path'])
 assert hashlib.sha256(trace.read_bytes()).hexdigest()==record['trace']['sha256']
 value['traces'].append(dict(rank=rank,analysis=module.analyze(trace)))
destination=out/'root-tape-runtime-cpu-ownership.json'
with destination.open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps(dict(value=value,remote_receipt=dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))))
PY
'''.replace('@ENTRY@',ENTRY).replace('@OUT@',repr(out))
run=subprocess.run(SSH+['bash','-s'],input=script.encode(),stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT,timeout=60)
if run.returncode:
    print(run.stdout.decode('utf8','replace'));raise SystemExit(run.returncode)
value=json.loads(run.stdout)
value['source_readout']=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
destination=REPO/'experiments/rl/results_native_root_tape_runtime_cpu_20261005.json'
with destination.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(local=str(destination),traces=[dict(rank=row['rank'],
 totals=row['analysis']['runtime_totals'],top_scopes=row['analysis']['runtime_by_scope'][:10])
 for row in value['value']['traces']]),indent=2))
