"""Stage a prepared-only query repair and verify it on the reused CPU runtime."""
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ROOT, SSH, SCP, REPO

OUT = ROOT + '/receipts/reward-query-clock-repair-20261006-v1'
LOCAL = AUDIT / 'textcraft-degradation-20261005/query-clock-repair-20261006/v1'
FILES = [REPO/'experiments/rl/reward_readout.py', REPO/'experiments/rl/test_reward_readout.py',
         AUDIT/'verify_reward_query_clock_cpu.py']
SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import hashlib,json,pathlib,psutil,subprocess,time
out=pathlib.Path('__OUT__')
assert not (out/'query-clock-repair.json').exists()
job=json.loads(pathlib.Path('__ROOT__/receipts/textcraft-native-adam-20261006-v1/job.json').read_bytes())
parent=psutil.Process(job['reused_environment_pid'])
assert abs(parent.create_time()-job['reused_environment_pid_birth']) < .05
for name,expected in __HASHES__.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
env=parent.environ(); env.pop('RAY_ADDRESS',None)
env.update(job['runtime_environment'])
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DT_QUERY_REPAIR_ROOT=str(out))
env['PYTHONPATH']=str(out)+':'+env['PYTHONPATH']
start=time.time()
with (out/'cpu.stdout.txt').open('wb') as log:
 result=subprocess.run([env['VENV_PYTHON'],str(out/'verify_reward_query_clock_cpu.py')],
  cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
print((out/'cpu.stdout.txt').read_text())
receipt=dict(observed_unix=time.time(),seconds=time.time()-start,exit_code=result.returncode,
 reused_environment_pid=parent.pid,reused_environment_pid_birth=parent.create_time(),sources=__HASHES__)
(out/'launch-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt)); raise SystemExit(result.returncode)
PY
'''

if __name__ == '__main__':
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in FILES}
    LOCAL.mkdir(parents=True, exist_ok=True)
    subprocess.run(SSH+['mkdir', '-p', OUT], check=True, timeout=30)
    subprocess.run(SCP+[str(p) for p in FILES]+[f'{SSH[-1]}:{OUT}/'], check=True, timeout=40)
    script = SCRIPT.replace('__OUT__', OUT).replace('__ROOT__', ROOT).replace('__HASHES__', repr(hashes))
    (LOCAL/'launch.sh').write_bytes(script.encode())
    run = subprocess.run(SSH+['bash', '-s'], input=script.encode(), capture_output=True, timeout=150)
    (LOCAL/'transport.stdout.txt').write_bytes(run.stdout+run.stderr)
    print(run.stdout.decode(errors='replace')); print(run.stderr.decode(errors='replace'))
    subprocess.run(SCP+[f'{SSH[-1]}:{OUT}/*.json', f'{SSH[-1]}:{OUT}/*.stdout.txt', str(LOCAL)+'/'],
                   check=True, timeout=40)
    run.check_returncode()
