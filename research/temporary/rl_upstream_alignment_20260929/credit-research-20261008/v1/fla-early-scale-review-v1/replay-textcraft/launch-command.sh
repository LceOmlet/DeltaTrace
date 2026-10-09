set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
tar -xf /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/fla-early-output-scale-20261009-v1/replay-textcraft/source.tar -C /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/fla-early-output-scale-20261009-v1/replay-textcraft
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/fla-early-output-scale-20261009-v1/replay-textcraft');root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');task='textcraft';hashes={'inspect_single_background_collection.py': '0e2ff8602046ec65d53bc47059479fc4e6b5bcbf8323fa6979a75ebefe63a3c4', 'layer-collection-inputs.json': 'e835680d4b527829fae2b2c116c58b198fb3eda7bf12da137e11871e411f87a5', 'passive_nonfinite.py': '3dc34ed90d7ad5d23e28c760fa3941171cb84b59f1f3bebe555c115754d2a8ce', 'inspect_action_curve.py': '7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5', 'diagnostic_gdn_owner.py': '7c06d5e0a4d6c00c13483dadd27d6389e25eee7868a05666d2d1faeaa2d62656', 'candidate-binding.json': '771c3ff566f69502e6b5c9e0dacd041c4e119b06ac0e43cd133298f7120fdd79'}
assert not (out/'launch.json').exists(),'Do not duplicate this replay'
for name,digest in hashes.items():
 p=out/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
 if name.endswith('.py'):ast.parse(p.read_bytes())
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Replay GPUs occupied'
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1'/task/(task+'-dt')/'source.json'
s=json.loads(source_path.read_bytes());env=dict(os.environ,**s['environment'])
env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None);env.pop('DT_CAPTURE_FLA_PRECAST',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=s['startup_options']['env.env_name']
env['DT_MAX_STEPS']=str(s['startup_options']['env.max_steps'])
env['DT_SINGLE_BACKGROUND_NONFINITE_REPLAY']='1'
env['DT_DIAGNOSTIC_GDN_OWNER_MANIFEST']=str(out/'candidate-binding.json')
dt=Path(s['dt_root']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
official=env.get('DT_OFFICIAL_ROOT') or qwen['official_root']
env['PYTHONPATH']=':'.join([str(out),str(dt),official,str(dt/'clean/qwen35'),s['pythonpath'],qwen['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'inspect_single_background_collection.py'),'--source',str(source_path),'--output',str(out/'results'),'--case',task]
if task=='textcraft':
 evidence=root/'receipts/direct-target-textcraft-author-curve-20261008-v1/textcraft-taskrunner-resolved-training-steps.json'
 assert hashlib.sha256(evidence.read_bytes()).hexdigest()=='57874a6f4491da68e5001fdf4f71a9d787b2dd54ec91a62b7216f1caa58da24d'
 argv+=['--owner-total-training-steps','330','--owner-total-steps-evidence',str(evidence)]
with (out/'driver.log').open('xb') as log:p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),unix=time.time(),source_commit='f020afd60f354f8d636055eec647a1ce05b5eece',files=hashes,argv=argv,devices=[4,5],physical_before=physical,DT_B4_calls_per_rank=1,optimizer=0,rollout=0,checkpoint_restore=0,production_changed=False)
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

PY
