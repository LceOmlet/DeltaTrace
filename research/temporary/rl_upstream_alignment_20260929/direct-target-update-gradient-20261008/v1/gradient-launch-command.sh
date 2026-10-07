set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,os,psutil,re,subprocess,time,hashlib
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-update-gradient-20261008-v1');assert not (out/'gradient-launch.json').exists()
assert psutil.Process(2833207).create_time()==1791370325.16
for rank in (0,1):assert not (root/f'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt/rank{rank}-release-update').exists()
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Selected cards4/5 occupied'
for name,h in {'inspect_actual_token_gradient_v2.py': '513229b4046cb91662e944e8201b6195616cfcd1a221b52f61e1333ff4e8f624', 'observe_native_optimizer_minibatch.py': '022466b2bac94617b8e627ea027fb3afdf4490dc4bc2bcaa11944ab444e36214', 'observe_native_actor_loss_gradients.py': '94219328ba644a5c0118f4a3bd2561b16f969643f2cd2915047202a7ff085047', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'}.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
p=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json';source=json.loads(p.read_bytes());assert hashlib.sha256(p.read_bytes()).hexdigest()=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
env=dict(os.environ,**source['environment']);env['CUDA_VISIBLE_DEVICES']='4,5';env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None);env['DT_TASK']=source['startup_options']['env.env_name'];env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps']);dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35'];env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'inspect_actual_token_gradient_v2.py')]
with (out/'gradient-driver.log').open('xb') as stream:process=subprocess.Popen(argv,cwd=str(out),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
r={'code_commit':'5fba8de4','pid':process.pid,'birth':psutil.Process(process.pid).create_time(),'launched_unix':time.time(),'devices':[4,5],'argv':argv,'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'script_hashes':{'inspect_actual_token_gradient_v2.py': '513229b4046cb91662e944e8201b6195616cfcd1a221b52f61e1333ff4e8f624', 'observe_native_optimizer_minibatch.py': '022466b2bac94617b8e627ea027fb3afdf4490dc4bc2bcaa11944ab444e36214', 'observe_native_actor_loss_gradients.py': '94219328ba644a5c0118f4a3bd2561b16f969643f2cd2915047202a7ff085047', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'},'scope':'Actual first global64/local32/B4 PPO minibatch; original full DT PG and one isolated Format-token contribution; no optimizer/scheduler write','planned_native_backward_passes_per_rank':2,'optimizer_steps':0,'checkpoint_restore':0,'DT_calls':0,'rollout_calls':0,'text_update_released':False,'formal_restart':False}
(out/'gradient-launch.json').write_text(json.dumps(r,indent=2)+'\n');(out/'gradient-before-physical.txt').write_text(physical);print(json.dumps(r))

PY
