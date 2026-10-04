"""Prepare and CPU-check one minimal actor candidate; never touch formal jobs."""
import hashlib
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote


if __name__ == '__main__':
    out = ROOT+'/receipts/owner-b8-dispatch-20260930/actor-shared-right-padding-20261004-v4'
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
try:
 driver=psutil.Process(job['pid'])
 assert driver.create_time()==job['observed_process_created_unix']
 driver_alive=True
except psutil.NoSuchProcess:
 driver_alive=False
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
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
previous=json.loads((out.parent/'actor-shared-right-padding-20261003/prepared.json').read_bytes())
assert sha(candidate/actor)==previous['after_actor_sha256'], 'Only fixture versions changed between CPU attempts'
cpu_previous=out.parent/'actor-shared-right-padding-20261003-v2'
previous_cpu=json.loads((cpu_previous/'prepared.json').read_bytes())
assert previous_cpu['test_returncode']==0 and sha(candidate/actor)==previous_cpu['after_actor_sha256']
assert sha(out/'test_shared_padding.py')==sha(cpu_previous/'test_shared_padding.py')
receipt=dict(role='Prepared candidate; formal jobs unchanged; no model numerical or speed acceptance from CPU tests',observed_unix=time.time(),
 previous_attempt=str(out.parent/'actor-shared-right-padding-20261003'),previous_actor_implementation_unchanged=True,
 code_commit='@COMMIT@',stager_sha256='@SHA@',base=str(base),candidate=str(candidate),entry=job['entry'],formal_driver_pid=job['pid'],formal_driver_birth=job['observed_process_created_unix'],formal_driver_alive_at_staging=driver_alive,
 before_actor_sha256=sha(base/actor),after_actor_sha256=sha(candidate/actor),changed_methods=['_forward_micro_batch'],test_returncode=previous_cpu['test_returncode'],
 reused_cpu_tests=dict(path=str(cpu_previous/'cpu-tests.xml'),sha256=sha(cpu_previous/'cpu-tests.xml'),scope='Same actor and test file hashes; 98 original CPU cases not rerun'),
 files={str(p):sha(p) for p in [out/'patch_actor_shared_right_padding.py',out/'test_shared_padding.py',out/'verify_owner_response_padding.py',out/'verify_actor_right_padding.py',candidate/'verl/workers/fsdp_workers.py']})
(out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
source=dict(formal_launch=str(pathlib.Path(job['output'])/'launch.json'),
 original_request_artifacts=str(root/'receipts/owner-b8-dispatch-20260930/formal-dt-prefix-requests-1790974394'),
 official_padding_test=str(root/'receipts/upstream-alignment-20260929/official/tests/models/test_transformer.py'),reference_owner=str(base),candidate_owner=str(candidate))
# Use the unchanged launch budget, not a new diagnostic training schedule.
options=json.loads(pathlib.Path(source['formal_launch']).read_bytes())['options']
source['original_total_training_steps']=options['trainer.total_training_steps']
(out/'source.json').write_text(json.dumps(source,indent=2)+'\n')
sys.path.insert(0,str(candidate))
os.environ['PADDING_DIAGNOSTIC_DIR']=str(out)
from verify_owner_response_padding import _OWNER_WORKER, ObservedWorker
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
assert _OWNER_WORKER is AsyncActorRolloutRefWorker
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from omegaconf import OmegaConf
factory_cfg=OmegaConf.load(candidate/'verl/trainer/config/ppo_trainer.yaml')
colocated=create_colocated_worker_cls(class_dict={'actor_rollout':RayClassWithInitArgs(ObservedWorker,config=factory_cfg.actor_rollout_ref,role='actor_rollout')})
assert colocated.cls.__ray_metadata__.modified_class.__name__=='WorkerDict'
receipt['factory_cpu_check']='Passed: original async worker selection and original colocated WorkerDict factory; no model or Ray cluster initialized'
(out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
'''.replace('@ENTRY@', ENTRY).replace('@ROOT@', ROOT).replace('@OUT@', out)
       .replace('@COMMIT@', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
       .replace('@SHA@', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
