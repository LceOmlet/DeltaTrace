"""Run only the saved-artifact CPU analysis in the existing recorded environment."""
import hashlib
import json
import subprocess

from stage_environment_entry import AUDIT, SSH, SCP
from stage_textcraft_native_adam import OUT, LOCAL
from stage_textcraft_native_readout import BASE

NAME = 'analyze_textcraft_native_adam.py'
SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import ast,hashlib,json,pathlib,psutil,subprocess,time
out=pathlib.Path('__OUT__'); base=pathlib.Path('__BASE__')
source=out/'__NAME__'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='__SHA__'
ast.parse(source.read_bytes())
assert (out/'completed.json').is_file(), 'Read the existing observation phase first'
job=json.loads((out/'job.json').read_bytes())
parent=psutil.Process(job['reused_environment_pid'])
assert abs(parent.create_time()-job['reused_environment_pid_birth'])<.05
env=parent.environ(); env.pop('RAY_ADDRESS',None)
for key,value in job['runtime_environment'].items():
 if value is None: env.pop(key,None)
 else: env[key]=value
env['CUDA_VISIBLE_DEVICES']=''
argv=[env['VENV_PYTHON'],'-u',str(source),'--input-dir',str(out),
 '--minibatch-path',str(base/'native-optimizer-minibatch.pkl'),
 '--output',str(out/'native-adam-analysis.json')]
started=time.time()
with (out/'analysis.stdout.txt').open('wb') as log:
 result=subprocess.run(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(source=dict(path=str(source),sha256='__SHA__'),
 argv=argv,returncode=result.returncode,started_unix=started,completed_unix=time.time(),
 original_environment_pid=parent.pid,original_environment_pid_birth=parent.create_time(),
 cuda_visible_devices='',model_initializations=0,DT_calls=0,optimizer_steps=0)
if result.returncode==0:
 p=out/'native-adam-analysis.json'; receipt['output']=dict(path=str(p),
  sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
(out/'analysis-cpu-run.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
raise SystemExit(result.returncode)
PY
'''

if __name__ == '__main__':
    sha = hashlib.sha256((AUDIT / NAME).read_bytes()).hexdigest()
    subprocess.run(SCP + [str(AUDIT / NAME), f'{SSH[-1]}:{OUT}/{NAME}'], check=True)
    script = (SCRIPT.replace('__OUT__', OUT).replace('__BASE__', BASE)
              .replace('__NAME__', NAME).replace('__SHA__', sha))
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'analysis-cpu-run.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    for name in ('analysis-cpu-run.json', 'analysis.stdout.txt'):
        subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}', str(LOCAL / name)], check=True)
    result.check_returncode()
    subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/native-adam-analysis.json',
                          str(LOCAL / 'native-adam-analysis.json')], check=True)
    receipt = json.loads((LOCAL / 'analysis-cpu-run.json').read_bytes())
    assert hashlib.sha256((LOCAL / 'native-adam-analysis.json').read_bytes()).hexdigest() == receipt['output']['sha256']
