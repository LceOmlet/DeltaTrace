"""Analyze this diagnostic's saved original artifacts on CPU; never launch training."""
import argparse
import hashlib
import json
import subprocess

from stage_environment_entry import AUDIT, ROOT, SSH, SCP


OUT = ROOT + '/receipts/textcraft-native-minibatch-20261006-v4'
FILES = ('analyze_textcraft_native_minibatch.py', 'test_analyze_textcraft_native_minibatch.py')
SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import hashlib,json,pathlib,psutil,subprocess,time
out=pathlib.Path('__OUT__'); mode='__MODE__'; hashes=__HASHES__
job=json.loads((out/'job.json').read_bytes())
parent=psutil.Process(job['reused_environment_pid'])
assert abs(parent.create_time()-job['reused_environment_pid_birth'])<.05
env=parent.environ()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
env.pop('RAY_ADDRESS',None)
prepared=json.loads((out/'prepared-diagnostic.json').read_bytes())
source_dirs={}
for name,expected in [('textcraft_owner_rollout.py','entry'),('dp_actor.py','verl'),('qwen35_dense_finite_runner.py','dt')]:
 paths=[pathlib.Path(p) for p in prepared['sources'] if pathlib.Path(p).name==name]
 assert len(paths)==1
 p=paths[0]
 assert hashlib.sha256(p.read_bytes()).hexdigest()==prepared['sources'][str(p)]
 source_dirs[expected]=str(p.parent if expected=='entry' else p.parents[3] if expected=='verl' else p.parents[2])
# Read-only analysis imports the exact protocol owner used in the real job.
env['PYTHONPATH']=':'.join([str(out),source_dirs['entry'],source_dirs['verl'],source_dirs['dt']])
for name,h in hashes.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
started=time.time()
if mode=='test':
 argv=[env['VENV_PYTHON'],'-m','pytest','-q',str(out/FILES[1]),'--junitxml='+str(out/'analyzer-cpu-tests.xml')]
 log=out/'analyzer-cpu-tests.stdout.txt'
else:
 assert (out/'native-minibatch-completed.json').is_file()
 argv=[env['VENV_PYTHON'],str(out/FILES[0]),str(out)]
 log=out/'native-minibatch-analysis.stdout.txt'
with log.open('wb') as stream:
 result=subprocess.run(argv,cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT)
receipt=dict(mode=mode,cuda_visible_devices='',optimizer_steps=0,argv=argv,
 sources=hashes,import_prefixes=env['PYTHONPATH'],started_unix=started,
 completed_unix=time.time(),exit_code=result.returncode,stdout=str(log))
(out/('cpu-analysis-'+mode+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(log.read_text()[-5000:]); print(json.dumps(receipt))
raise SystemExit(result.returncode)
PY
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('test', 'analyze'))
    args = parser.parse_args()
    hashes = {name: hashlib.sha256((AUDIT/name).read_bytes()).hexdigest() for name in FILES}
    for name in FILES:
        subprocess.run(SCP+[str(AUDIT/name), f'{SSH[-1]}:{OUT}/{name}'], check=True)
    script = SCRIPT.replace('__OUT__', OUT).replace('__MODE__', args.mode).replace('__HASHES__', repr(hashes)).replace('FILES[1]', repr(FILES[1])).replace('FILES[0]', repr(FILES[0]))
    result = subprocess.run(SSH+['bash', '-s'], input=script.encode(), capture_output=True)
    local = AUDIT/'textcraft-degradation-20261005/native-minibatch-v4'
    (local/('cpu-analysis-'+args.mode+'.stdout.txt')).write_bytes(result.stdout+result.stderr)
    print(result.stdout.decode(errors='replace')); print(result.stderr.decode(errors='replace'))
    result.check_returncode()
