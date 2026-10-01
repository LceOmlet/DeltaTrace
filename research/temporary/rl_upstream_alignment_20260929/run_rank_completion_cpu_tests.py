"""Run the CPU owner-interface tests and record exact sources/resources."""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate-id', default='appworld-rank-completion-20261002')
parser.add_argument('--receipt-name', default='appworld-rank-completion')
parser.add_argument('--test', default='test_rank_completion_candidate.py')
parser.add_argument('--entry', default=ROOT+'/candidates/appworld-rollout-scope-20261001/entry')
args = parser.parse_args()
receipt = ROOT+'/receipts/owner-b8-dispatch-20260930/'+args.receipt_name
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
remote(f'mkdir -p {receipt}/{args.candidate_id}\n')
tests = {args.test, 'test_rank_completion_candidate.py'}
subprocess.run(SCP+[str(AUDIT/name) for name in sorted(tests)]+[f'{SSH[-1]}:{receipt}/'], check=True)
subprocess.run(SCP+[str(AUDIT/args.candidate_id/'loop_owner_rollout.py'),
                   f'{SSH[-1]}:{receipt}/{args.candidate_id}/'], check=True)
script = r'''set -e
source @ENTRY@/metax-entry.env.sh
export CUDA_VISIBLE_DEVICES=""
export MACA_VISIBLE_DEVICES=""
export ORIGINAL_LOOP_ENTRY=@ORIGINAL_ENTRY@
export DEPLOYED_LOOP_ENTRY="$ORIGINAL_LOOP_ENTRY"
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
 'test':'@TEST@',
 'support_test':'test_rank_completion_candidate.py',
 'candidate':'@CANDIDATE_ID@/loop_owner_rollout.py',
 'original':Path(os.environ['ORIGINAL_LOOP_ENTRY'])/'loop_owner_rollout.py',
 'ray_pool':inspect.getsourcefile(ActorPool),
 'verl_rpc':inspect.getsourcefile(RayWorkerGroup)}.items()}
start=time.monotonic();peak=0;observations=0
with open('final-cpu-tests.log','w') as log:
 process=subprocess.Popen([sys.executable,'-m','pytest','-q','@TEST@',
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
for name in ['final-cpu-tests.log','final-cpu-tests.xml','busy-owner-batches.json']:
 p=Path(name)
 if p.is_file():result.setdefault('receipts',{})[name]=dict(path=str(p.resolve()),sha256=sha(p))
Path('final-cpu-tests.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));print(Path('final-cpu-tests.log').read_text()[-5000:])
raise SystemExit(process.returncode)
PY
'''
for key,value in {'ENTRY':ENTRY,'ROOT':ROOT,'RECEIPT':receipt,'COMMIT':revision,
                  'RUNNER_SHA':runner_sha,'ORIGINAL_ENTRY':args.entry,
                  'TEST':args.test,'CANDIDATE_ID':args.candidate_id}.items():
    script = script.replace('@'+key+'@',value)
result = subprocess.run(SSH+['bash','-s'],input=script.encode())
target = AUDIT/args.candidate_id
for name in ['final-cpu-tests.log','final-cpu-tests.xml','final-cpu-tests.json']:
    subprocess.run(SCP+[f'{SSH[-1]}:{receipt}/{name}', str(target/name)],check=True)
raise SystemExit(result.returncode)
