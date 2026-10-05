"""Read completed real traces; no model, GPU operation or new acceptance gate."""
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, SCP, SSH


source=REPO/'experiments/rl/results_native_root_tape_hot_20261004.json'
raw=json.loads(source.read_bytes())
assert raw['completed'] and raw['driver']['status']=='exited'
out=raw['remote_root']
analyzer=AUDIT/'analyze_native_hot_trace.py'
original=subprocess.check_output(['git','show','HEAD:research/temporary/rl_upstream_alignment_20260929/analyze_native_hot_trace.py'],cwd=REPO)
baseline=AUDIT/'root-tape-original-trace-analyzer.py'
with baseline.open('xb') as stream:stream.write(original)
for path in (analyzer,baseline):
    subprocess.run(SCP+[str(path),f'{SSH[-1]}:{out}/{path.name}'],check=True)
script=r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import hashlib,importlib.util,json,pathlib,xml.etree.ElementTree as ET
out=pathlib.Path(@OUT@)
def load(name):
 path=out/name
 spec=importlib.util.spec_from_file_location(name.removesuffix('.py').replace('-','_'),path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module
current=load('analyze_native_hot_trace.py')
original=load('root-tape-original-trace-analyzer.py')
results=[]
for rank in (0,1):
 record=json.loads((out/f'rank{rank}.json').read_bytes())
 trace=pathlib.Path(record['trace']['path'])
 assert hashlib.sha256(trace.read_bytes()).hexdigest()==record['trace']['sha256']
 before=original.analyze(trace)
 unchanged=current.analyze(trace)
 assert before==unchanged, 'Default parser changed on the actual saved trace'
 expanded=current.analyze(trace,parameter_ranges=True)
 assert {key:expanded[key] for key in before}==before, 'Original event partition changed'
 results.append(dict(rank=rank,default_analysis_unchanged=True,original_partition_unchanged=True,
  analysis=expanded))
value=dict(scope='One completed actual real B4 root-tape warm profile; source device sums can overlap and are not additive wall time',
 formal_deployment=False,model_or_GPU_calls_added=0,traces=results,
 cpu_tests=[node.attrib for node in ET.parse(out/'cpu-tests.xml').getroot().iter('testsuite')],
 analysis_sources=[dict(path=str(out/name),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest())
  for name in ('analyze_native_hot_trace.py','root-tape-original-trace-analyzer.py')])
path=out/'root-tape-original-parameter-phases.json'
with path.open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps(dict(value=value,remote_receipt=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))))
PY
'''.replace('@ENTRY@',ENTRY).replace('@OUT@',repr(out))
run=subprocess.run(SSH+['bash','-s'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
if run.returncode:
    print(run.stdout.decode('utf8','replace'));raise SystemExit(run.returncode)
value=json.loads(run.stdout)
value['source_readout']=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
destination=REPO/'experiments/rl/results_native_root_tape_parameter_phases_20261004.json'
with destination.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(local=str(destination),cpu_tests=value['value']['cpu_tests'],
 traces=[dict(rank=row['rank'],CPU=row['analysis']['parameter_cpu_by_stage'],
  device=row['analysis']['parameter_device_by_stage'],links=row['analysis']['parameter_link_counts'])
  for row in value['value']['traces']]),indent=2))
