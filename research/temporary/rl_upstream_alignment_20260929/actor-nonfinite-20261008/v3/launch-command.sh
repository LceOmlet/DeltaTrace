source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'); out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-rollout-DT-actor-nonfinite-20261008-v3')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
assert sha(source_path)=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
source=json.loads(source_path.read_bytes())
assert not (out/'launch.json').exists(), 'Diagnostic already launched'
worker=out/'diagnose_rollout_dt_to_actor.py'
assert sha(worker)=='ef2cf1f065b8bfe6bcec14559d626a403e51e14173c9d8b9cf506c3c21a4b41c'; ast.parse(worker.read_text())
previous=root/'receipts/textcraft-DT-to-actor-nonfinite-20261008-v2'
assert sha(previous/'diagnose_dt_to_actor.py')=='998698ecf191a4bf3202c265ae5130402fad1b343e58838f3cd2e13d1a32816c'
old=root/'receipts/textcraft-native-nonfinite-20261008-v1'
assert sha(old/'inspect_native_actor_nonfinite.py')=='9dab61c1881297c3414c1f801db6727941a83ad3e5a42eaad4abe01ffd7398bb'
for relative in ('verl/workers/actor/dp_actor.py','verl/workers/sharding_manager/fsdp_vllm.py','verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'):
 path=Path(source['verl_root'])/relative
 assert sha(path)==source['source_bindings'][str(path)]
env=dict(os.environ,**source['environment'])
env.pop('RAY_ADDRESS',None); env.pop('MACA_VISIBLE_DEVICES',None)
env.update(CUDA_VISIBLE_DEVICES='4,5',DT_ACTOR_INCIDENT_OUT=str(out))
dt=Path(env['DT_ROOT']); q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(previous),str(old),str(dt),env.get('DT_OFFICIAL_ROOT') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(worker)]
r=subprocess.run(argv+['--inspect-only'],cwd=out,env=dict(env,CUDA_VISIBLE_DEVICES='-1'),capture_output=True,timeout=120)
(out/'prepare.stdout').write_bytes(r.stdout); (out/'prepare.stderr').write_bytes(r.stderr)
if r.returncode:
 print(r.stderr.decode(errors='replace')[-5000:]); r.check_returncode()
inspection=json.loads((out/'handoff-input-inspection.json').read_bytes())
assert inspection['actor_rows']==256 and inspection['DT_rows']==176 and not inspection['CUDA_initialized']
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M), 'Physical4/5 occupied'
with (out/'driver.log').open('xb') as stream:
 p=subprocess.Popen(['timeout','--kill-after=10s','1800s',*argv],cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),devices=[4,5],argv=['timeout','--kill-after=10s','1800s',*argv],worker_sha256=sha(worker),base_commit='78dd0d97d132b19fc4ec58e6dd9d37d5fb109008',worker_committed_at_launch=False,source_path=str(source_path),source_sha256=sha(source_path),scope='Original vLLM initialization/LoRA sync/generation/sleep once on 32 original prompts, saved176-row DT and full256-row original actor update. Not an environment rollout or formal restart.',per_card_microbatch=4,global_optimizer_minibatch=64,lora_rank=8,lora_alpha=16,diagnostic_only=True,formal_deployment=False,host_available_bytes=psutil.virtual_memory().available)
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'before-physical.txt').write_text(physical)
print(json.dumps(receipt))
PY
