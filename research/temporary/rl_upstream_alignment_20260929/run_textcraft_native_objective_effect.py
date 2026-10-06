"""CPU-only loss observation in the existing native environment; no new job."""
import hashlib
import json
import subprocess
from stage_environment_entry import AUDIT, ROOT, SSH, SCP
from stage_textcraft_native_adam import OUT, LOCAL

NAME = 'analyze_textcraft_native_objective_effect.py'
SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import hashlib,json,pathlib,psutil,subprocess,time
out=pathlib.Path('__OUT__'); source=out/'__NAME__'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='__SHA__'
assert (out/'completed.json').is_file() and not (out/'native-objective-effect.json').exists()
job=json.loads((out/'job.json').read_bytes())
parent=psutil.Process(job['reused_environment_pid']); assert abs(parent.create_time()-job['reused_environment_pid_birth'])<.05
env=parent.environ(); env.pop('RAY_ADDRESS',None)
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(out)+':'+job['runtime_environment']['PYTHONPATH'])
helper=out/'analyze_textcraft_native_adam.py'; assert helper.is_file()
input_path=job['input']['path']; assert hashlib.sha256(pathlib.Path(input_path).read_bytes()).hexdigest()==job['input']['sha256']
began=time.time()
with (out/'native-objective-effect.stdout.txt').open('wb') as log:
 result=subprocess.run([env['VENV_PYTHON'],str(source),'--input-dir',str(out),'--minibatch-path',input_path],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,'Inspect saved CPU observation log; never launch training on this path'
data=json.loads((out/'native-objective-effect.json').read_bytes()); assert data['operations']==dict(model_initializations=0,model_forwards=0,backward=0,optimizer=0,DT=0)
assert not data['runtime']['after']['cuda_initialized'] and not data['runtime']['after']['distributed_initialized']
receipt=dict(source=dict(path=str(source),sha256='__SHA__'),observed_unix=time.time(),seconds=time.time()-began,reused_environment_pid=parent.pid,reused_environment_pid_birth=parent.create_time(),output=dict(path=str(out/'native-objective-effect.json'),sha256=hashlib.sha256((out/'native-objective-effect.json').read_bytes()).hexdigest()),operations=data['operations'])
(out/'native-objective-effect-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY
'''

if __name__ == '__main__':
    sha = hashlib.sha256((AUDIT / NAME).read_bytes()).hexdigest()
    subprocess.run(SCP + [str(AUDIT / NAME), f'{SSH[-1]}:{OUT}/{NAME}'], check=True)
    result = subprocess.run(SSH + ['bash','-s'], input=SCRIPT.replace('__OUT__',OUT).replace('__NAME__',NAME).replace('__SHA__',sha).encode(), capture_output=True)
    (LOCAL / 'native-objective-effect-run.stdout.txt').write_bytes(result.stdout+result.stderr)
    print(result.stdout.decode(errors='replace')); print(result.stderr.decode(errors='replace'))
    if result.returncode:
        subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/native-objective-effect.stdout.txt',str(LOCAL)+'/'], check=True)
    result.check_returncode()
    subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/native-objective-effect*.json',f'{SSH[-1]}:{OUT}/native-objective-effect.stdout.txt',str(LOCAL)+'/'], check=True)
