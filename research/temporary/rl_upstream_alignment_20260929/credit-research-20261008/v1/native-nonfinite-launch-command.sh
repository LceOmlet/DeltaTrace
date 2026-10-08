source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-nonfinite-20261008-v1'
HASHES={'inspect_native_actor_nonfinite.py': '9dab61c1881297c3414c1f801db6727941a83ad3e5a42eaad4abe01ffd7398bb', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'}
COMMIT='362635691454496cf0e7b95da6df3ff98c363845'

import ast,hashlib,json,os,re,psutil,subprocess,time
from pathlib import Path
root=Path(ROOT);out=Path(OUT)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
assert sha(source_path)=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
source=json.loads(source_path.read_bytes())
assert not (out/'launch.json').exists(),'Do not duplicate this bounded diagnostic'
for name,value in HASHES.items():
 assert sha(out/name)==value
 ast.parse((out/name).read_text())
env=dict(os.environ,**source['environment'])
env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_ACTOR_INCIDENT_OUT']=str(out)
dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'inspect_native_actor_nonfinite.py')]
prepared=subprocess.run(argv+['--inspect-only'],cwd=out,env=dict(env,CUDA_VISIBLE_DEVICES='-1'),capture_output=True,timeout=120)
(out/'prepare.stdout').write_bytes(prepared.stdout);(out/'prepare.stderr').write_bytes(prepared.stderr)
if prepared.returncode:
 print(prepared.stderr.decode(errors='replace')[-4000:]);prepared.check_returncode()
inspection=json.loads((out/'input-inspection.json').read_bytes())
assert inspection['rows']==256 and inspection['microbatch']==4
assert (inspection['lora_rank'],inspection['lora_alpha'])==(8,16) and not inspection['cuda_initialized']
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Physical4/5 occupied'
with (out/'driver.log').open('xb') as stream:
 p=subprocess.Popen(['timeout','--kill-after=10s','900s',*argv],cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),devices=[4,5],argv=['timeout','--kill-after=10s','900s',*argv],scripts=HASHES,diagnostic_code_commit=COMMIT,source_path=str(source_path),source_sha256=sha(source_path),scope=inspection['scope'],original_rows=256,global_optimizer_minibatch=64,per_card_microbatch=4,rank=8,alpha=16,formal_training_launched=False,DT=0,rollout=0,checkpoint_restore=0,initial_weight_scope='Fresh original native actor; original LoRA A and old/ref outputs were not captured. Not bitwise historical replay.',host_available_bytes=psutil.virtual_memory().available)
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'before-physical.txt').write_text(physical)
print(json.dumps(receipt))

PY
