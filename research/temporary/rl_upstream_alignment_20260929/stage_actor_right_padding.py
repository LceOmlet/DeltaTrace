"""Prepare and CPU-check one minimal actor candidate; never touch formal jobs."""
import hashlib
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote


if __name__ == '__main__':
    out = ROOT+'/receipts/owner-b8-dispatch-20260930/actor-shared-right-padding-20261003'
    remote(f'test ! -e {out}/prepared.json && mkdir -p {out}\n')
    files = [REPO/'experiments/rl'/name for name in ('patch_actor_shared_right_padding.py', 'test_shared_padding.py')]
    files += [AUDIT/name for name in ('verify_owner_response_padding.py', 'verify_actor_right_padding.py')]
    for path in files:
        subprocess.run(SCP+[str(path), f'{SSH[-1]}:{out}/{path.name}'], check=True)
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,os,pathlib,psutil,shutil,subprocess,time,sys
root=pathlib.Path('@ROOT@');out=pathlib.Path('@OUT@')
job=next(x for x in json.loads((root/'active-training.json').read_bytes())['jobs'] if x['task']=='AppWorld')
driver=psutil.Process(job['pid'])
assert driver.create_time()==job['observed_process_created_unix']
base=pathlib.Path(job['verl_root']);candidate=out/'verl-root'
assert not candidate.exists()
shutil.copytree(base/'verl',candidate/'verl')
actor=pathlib.Path('verl/workers/actor/dp_actor.py')
before=(base/actor).read_text()
assert hashlib.sha256(before.encode()).hexdigest()=='1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd'
sys.path.insert(0,str(out));sys.path.insert(1,'@ENTRY@')
from patch_actor_shared_right_padding import patch
after=patch(before);assert patch(after)==after
def methods(text):
 c=next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef) and x.name=='DataParallelPPOActor')
 return {x.name:ast.dump(x) for x in c.body if isinstance(x,ast.FunctionDef)}
a,b=methods(before),methods(after)
assert [k for k in a if a[k]!=b[k]]==['_forward_micro_batch']
(candidate/actor).write_text(after)
host=root/'candidates/native-host-cache-phase-20261002/e5eb4afc42f1/fsdp_workers.py'
assert hashlib.sha256(host.read_bytes()).hexdigest()=='e5eb4afc42f10fb4608b3ac43046c906d6a2d21a5387ee02e176bc395f1c6f39'
shutil.copy2(host,candidate/'verl/workers/fsdp_workers.py')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VERL_ROOT=str(candidate))
env['PYTHONPATH']=':'.join([str(candidate),str(out),job['entry'],env['PYTHONPATH']])
with (out/'cpu-tests.log').open('wb') as log:
 test=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q',str(out/'test_shared_padding.py'),'--junitxml='+str(out/'cpu-tests.xml')],env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=dict(role='Prepared candidate; formal jobs unchanged; no model numerical or speed acceptance from CPU tests',observed_unix=time.time(),
 code_commit='@COMMIT@',stager_sha256='@SHA@',base=str(base),candidate=str(candidate),entry=job['entry'],formal_driver_pid=driver.pid,formal_driver_birth=driver.create_time(),
 before_actor_sha256=sha(base/actor),after_actor_sha256=sha(candidate/actor),changed_methods=['_forward_micro_batch'],test_returncode=test.returncode,
 files={str(p):sha(p) for p in [out/'patch_actor_shared_right_padding.py',out/'test_shared_padding.py',out/'verify_owner_response_padding.py',out/'verify_actor_right_padding.py',candidate/'verl/workers/fsdp_workers.py']})
(out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
source=dict(formal_launch=str(pathlib.Path(job['output'])/'launch.json'),
 original_request_artifacts=str(root/'receipts/owner-b8-dispatch-20260930/formal-dt-prefix-requests-1790974394'),
 official_padding_test=str(root/'receipts/upstream-alignment-20260929/official/tests/models/test_transformer.py'),reference_owner=str(base),candidate_owner=str(candidate))
# Use the unchanged launch budget, not a new diagnostic training schedule.
options=json.loads(pathlib.Path(source['formal_launch']).read_bytes())['options']
source['original_total_training_steps']=options['trainer.total_training_steps']
(out/'source.json').write_text(json.dumps(source,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
if test.returncode:print((out/'cpu-tests.log').read_text()[-8000:],flush=True)
test.check_returncode()
PY
'''.replace('@ENTRY@', ENTRY).replace('@ROOT@', ROOT).replace('@OUT@', out)
       .replace('@COMMIT@', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
       .replace('@SHA@', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
