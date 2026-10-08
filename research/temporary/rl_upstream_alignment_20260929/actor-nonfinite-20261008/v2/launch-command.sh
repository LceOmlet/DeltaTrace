source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,os,re,psutil,subprocess,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'); out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-DT-to-actor-nonfinite-20261008-v2')
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source_path)=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
source=json.loads(source_path.read_bytes())
assert not (out/'launch.json').exists(), 'Do not duplicate this diagnostic'
worker=out/'diagnose_dt_to_actor.py'
assert sha(worker)=='998698ecf191a4bf3202c265ae5130402fad1b343e58838f3cd2e13d1a32816c'; ast.parse(worker.read_text())
old=root/'receipts/textcraft-native-nonfinite-20261008-v1'
original=old/'inspect_native_actor_nonfinite.py'
assert sha(original)=='9dab61c1881297c3414c1f801db6727941a83ad3e5a42eaad4abe01ffd7398bb'
actor=Path(source['verl_root'])/'verl/workers/actor/dp_actor.py'
assert sha(actor)=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
env=dict(os.environ,**source['environment'])
env.pop('RAY_ADDRESS',None); env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']='4,5'; env['DT_ACTOR_INCIDENT_OUT']=str(out)
dt=Path(env['DT_ROOT']); q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(old),str(dt),env.get('DT_OFFICIAL_ROOT') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(worker)]
prepared=subprocess.run(argv+['--inspect-only'],cwd=out,env=dict(env,CUDA_VISIBLE_DEVICES='-1'),capture_output=True,timeout=120)
(out/'prepare.stdout').write_bytes(prepared.stdout); (out/'prepare.stderr').write_bytes(prepared.stderr)
if prepared.returncode:
 print(prepared.stderr.decode(errors='replace')[-4000:]); prepared.check_returncode()
inspection=json.loads((out/'DT-input-inspection.json').read_bytes())
assert inspection['actor_rows']==256 and not inspection['CUDA_initialized']
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Physical4/5 occupied'
with (out/'driver.log').open('xb') as stream:
 p=subprocess.Popen(['timeout','--kill-after=10s','1800s',*argv],cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),devices=[4,5],argv=['timeout','--kill-after=10s','1800s',*argv],worker_sha256=sha(worker),reused_diagnostic_sha256=sha(original),base_commit='89776c4616ddfc3c0c8fa1f109a73ae10a618a30',worker_committed_at_launch=False,source_path=str(source_path),source_sha256=sha(source_path),actor_path=str(actor),actor_sha256=sha(actor),scope='Saved original DT carrier then native full actor update with original saved advantages. No rollout or checkpoint restore.',DT_calls='Original owner decides from exact saved prepared carrier',DT_rows=inspection['rows'],actor_rows=256,global_optimizer_minibatch=64,per_card_microbatch=4,lora_rank=8,lora_alpha=16,formal_deployment=False,diagnostic_only=True,host_available_bytes=psutil.virtual_memory().available)
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'before-physical.txt').write_text(physical)
print(json.dumps(receipt))
PY
