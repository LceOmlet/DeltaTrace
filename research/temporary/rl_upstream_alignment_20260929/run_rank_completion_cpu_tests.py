"""Run the CPU owner-interface tests and record exact sources/resources."""
import json
import hashlib
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote

receipt = ROOT+'/receipts/owner-b8-dispatch-20260930/appworld-rank-completion'
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
subprocess.run(SCP+[str(AUDIT/'test_rank_completion_candidate.py'), f'{SSH[-1]}:{receipt}/'], check=True)
script = r'''set -e
source @ENTRY@/metax-entry.env.sh
export CUDA_VISIBLE_DEVICES=""
export MACA_VISIBLE_DEVICES=""
export ORIGINAL_LOOP_ENTRY=@ROOT@/candidates/appworld-rollout-scope-20261001/entry
export PYTHONPATH="@RECEIPT@:$ORIGINAL_LOOP_ENTRY:@ROOT@/candidates/appworld-rollout-scope-20261001/verl:$PYTHONPATH"
cd @RECEIPT@
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,inspect,json,os,psutil,subprocess,sys,time
import ray
from ray.util.actor_pool import ActorPool
from verl.single_controller.ray.base import RayWorkerGroup
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
artifacts={name:dict(path=str(path),sha256=sha(path)) for name,path in {
 'test':'test_rank_completion_candidate.py',
 'candidate':'appworld-rank-completion-20261002/loop_owner_rollout.py',
 'original':Path(os.environ['ORIGINAL_LOOP_ENTRY'])/'loop_owner_rollout.py',
 'ray_pool':inspect.getsourcefile(ActorPool),
 'verl_rpc':inspect.getsourcefile(RayWorkerGroup)}.items()}
start=time.monotonic();peak=0;observations=0
with open('final-cpu-tests.log','w') as log:
 process=subprocess.Popen([sys.executable,'-m','pytest','-q','test_rank_completion_candidate.py',
  '--junitxml=final-cpu-tests.xml'],stdout=log,stderr=subprocess.STDOUT)
 driver=psutil.Process(process.pid)
 while process.poll() is None:
  total=0
  try:
   for child in [driver]+driver.children(recursive=True):
    try:total+=child.memory_full_info().pss
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
  except psutil.NoSuchProcess:pass
  peak=max(peak,total);observations+=1;time.sleep(.5)
result=dict(test_code_commit='@COMMIT@',runner_source_sha256='@RUNNER_SHA@',
 python=sys.executable,ray_version=ray.__version__,
 exit_code=process.returncode,wall_seconds=time.monotonic()-start,
 sampled_process_tree_pss_peak_bytes=peak,resource_observations=observations,
 ray_resources=dict(num_cpus=2,num_gpus=0),sources=artifacts,
 scope='CPU transport/API tests; no weights, model, DT, PPO update or numerical tolerance test')
for name in ['final-cpu-tests.log','final-cpu-tests.xml']:
 p=Path(name)
 if p.is_file():result.setdefault('receipts',{})[name]=dict(path=str(p.resolve()),sha256=sha(p))
Path('final-cpu-tests.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));print(Path('final-cpu-tests.log').read_text()[-5000:])
raise SystemExit(process.returncode)
PY
'''
for key,value in {'ENTRY':ENTRY,'ROOT':ROOT,'RECEIPT':receipt,'COMMIT':revision,
                  'RUNNER_SHA':runner_sha}.items():
    script = script.replace('@'+key+'@',value)
result = subprocess.run(SSH+['bash','-s'],input=script.encode())
target = AUDIT/'appworld-rank-completion-20261002'
for name in ['final-cpu-tests.log','final-cpu-tests.xml','final-cpu-tests.json']:
    subprocess.run(SCP+[f'{SSH[-1]}:{receipt}/{name}', str(target/name)],check=True)
raise SystemExit(result.returncode)
