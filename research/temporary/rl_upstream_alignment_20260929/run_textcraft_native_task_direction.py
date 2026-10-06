"""Run a CPU-only description of existing native gradient/parameter shards."""
import hashlib
import json
import subprocess

from stage_environment_entry import AUDIT, SSH, SCP
from stage_textcraft_native_adam import OUT, LOCAL

NAME = 'analyze_textcraft_native_task_direction.py'
SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import ast,hashlib,json,pathlib,psutil,subprocess,time
out=pathlib.Path('__OUT__'); source=out/'__NAME__'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='__SHA__'
ast.parse(source.read_bytes())
assert (out/'completed.json').is_file() and not (out/'native-task-direction.json').exists()
job=json.loads((out/'job.json').read_bytes())
parent=psutil.Process(job['reused_environment_pid'])
assert abs(parent.create_time()-job['reused_environment_pid_birth'])<.05
env=parent.environ(); env.pop('RAY_ADDRESS',None)
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
input_path=pathlib.Path(job['input']['path'])
assert hashlib.sha256(input_path.read_bytes()).hexdigest()==job['input']['sha256']
argv=[env['VENV_PYTHON'],str(source),'--input-dir',str(out),'--output',str(out/'native-task-direction.json')]
began=time.time()
with (out/'native-task-direction.stdout.txt').open('wb') as log:
 result=subprocess.run(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,'Inspect the saved CPU analysis failure; no model job is launched by this path'
data=json.loads((out/'native-task-direction.json').read_bytes())
assert all(value==0 for value in data['operations'].values())
assert not data['runtime']['resources_after']['cuda_initialized']
assert not data['runtime']['resources_after']['distributed_initialized']
receipt=dict(source=dict(path=str(source),sha256='__SHA__'),observed_unix=time.time(),seconds=time.time()-began,
 reused_environment_pid=parent.pid,reused_environment_pid_birth=parent.create_time(),argv=argv,
 output=dict(path=str(out/'native-task-direction.json'),sha256=hashlib.sha256((out/'native-task-direction.json').read_bytes()).hexdigest()),
 operations=data['operations'])
(out/'native-task-direction-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY
'''

if __name__ == '__main__':
    sha = hashlib.sha256((AUDIT / NAME).read_bytes()).hexdigest()
    subprocess.run(SCP + [str(AUDIT / NAME), f'{SSH[-1]}:{OUT}/{NAME}'], check=True)
    script = SCRIPT.replace('__OUT__', OUT).replace('__NAME__', NAME).replace('__SHA__', sha)
    (LOCAL / 'native-task-direction-launch.sh').write_bytes(script.encode())
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (LOCAL / 'native-task-direction-run.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace')); print(result.stderr.decode(errors='replace'))
    subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/native-task-direction.stdout.txt', str(LOCAL)+'/'], check=True)
    result.check_returncode()
    subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/native-task-direction*.json', str(LOCAL)+'/'], check=True)
    receipt = json.loads((LOCAL / 'native-task-direction-receipt.json').read_bytes())
    assert hashlib.sha256((LOCAL / 'native-task-direction.json').read_bytes()).hexdigest() == receipt['output']['sha256']
