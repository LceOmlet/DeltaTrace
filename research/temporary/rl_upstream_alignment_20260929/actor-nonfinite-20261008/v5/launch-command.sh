source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,os,psutil,re,subprocess,time,urllib.request
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-first-iteration-nonfinite-20261008-v5')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
assert sha(source_path)=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
source=json.loads(source_path.read_bytes())
assert not (out/'launch.json').exists(),'Do not duplicate this diagnostic'
worker=out/'observe_native_first_iteration.py'
assert sha(worker)=='078df501d0ec4d8d72a90ca32ca7032955bb16d43376a164870a9bf515d1d7db';ast.parse(worker.read_text())
files={}
for relative,expected in {
 'verl/workers/actor/dp_actor.py':'3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
 'verl/trainer/ppo/core_algos.py':'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
 'verl/workers/fsdp_workers.py':'e5eb4afc42f10fb4608b3ac43046c906d6a2d21a5387ee02e176bc395f1c6f39',
 'verl/workers/sharding_manager/fsdp_vllm.py':'364e69455cff871d3dd842063f216e882a3eccbdd38322fa2bd731b6e5f0bcc2',
}.items():
 path=Path(source['verl_root'])/relative
 assert sha(path)==expected,str(path)
 files[str(path)]=expected
env=dict(os.environ,**source['environment'])
env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_ACTOR_INCIDENT_OUT']=str(out)
dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(worker)]
prepared=subprocess.run(argv+['--inspect-only'],cwd=out,env=dict(env,CUDA_VISIBLE_DEVICES='-1'),capture_output=True,timeout=120)
(out/'prepare.stdout').write_bytes(prepared.stdout);(out/'prepare.stderr').write_bytes(prepared.stderr)
if prepared.returncode:
 print(prepared.stderr.decode(errors='replace')[-4000:]);prepared.check_returncode()
inspection=json.loads((out/'CPU-inspection.json').read_bytes())
assert not inspection['CUDA_initialized'] and inspection['microbatch']==4
assert (inspection['lora_rank'],inspection['lora_alpha'])==(8,16)
assert inspection['global_optimizer_minibatch']==64 and inspection['multi_turn']
assert inspection['epochs']==30 and inspection['total_training_steps'] is None
assert inspection['resume_mode']=='disable'
with urllib.request.urlopen('http://127.0.0.1:36005/docs',timeout=5) as response:assert response.status==200
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Physical4/5 occupied'
with (out/'driver.log').open('xb') as stream:
 p=subprocess.Popen(['timeout','--kill-after=10s','5400s',*argv],cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),devices=[4,5],
 argv=['timeout','--kill-after=10s','5400s',*argv],worker_sha256=sha(worker),base_commit='78dd0d97d132b19fc4ec58e6dd9d37d5fb109008',worker_committed_at_launch=False,
 source_path=str(source_path),source_sha256=sha(source_path),owner_files=files,effective_environment=env,
 scope='Original complete first rollout, old/ref, DT, whitening and full native update; stop after original step1 metrics.',
 configured_training_budget_unchanged=True,per_card_microbatch=4,global_optimizer_minibatch=64,lora_rank=8,lora_alpha=16,
 formal_deployment=False,diagnostic_only=True,checkpoint_restore=False,anomaly_detector=False,
 host_available_bytes=psutil.virtual_memory().available)
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'before-physical.txt').write_text(physical)
print(json.dumps(receipt))
PY
