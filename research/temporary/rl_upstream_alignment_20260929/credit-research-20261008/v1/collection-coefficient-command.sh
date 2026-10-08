set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-collection-coefficients-20261008-v1'
HASHES={'inspect_collection_coefficients.py': '9c2c8f31fb8a13304c96f3929fa6e5fbe41f995b1948f700a1ba6bdfce1f9986', 'inspect_collection_gradients.py': 'b20c6880472e8bd1399bdaf570bdcbb5523feb19ade5b0144fff9f82772aa7bd', 'observe_native_optimizer_minibatch.py': '6ed5cd292d8d34d9f701bdf859ec97c7100f9b5e9adc14ab798dbd597645a1c2', 'observe_native_actor_loss_gradients.py': '94219328ba644a5c0118f4a3bd2561b16f969643f2cd2915047202a7ff085047', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5', 'coefficient-inputs.json': '6707c6ad639655a2cff768189218f1fe1dda83d49d1c651d0265f1bd90764adb'}

import hashlib,json,os,psutil,subprocess,time
from pathlib import Path
root=Path(ROOT);out=Path(OUT)
assert not (out/'coefficients.json').exists(),'Read existing completed measurements instead of repeating'
assert not psutil.pid_exists(768614) or psutil.Process(768614).create_time()!=1791430928.98,'Prior diagnostic still live'
assert psutil.pid_exists(2833207) and psutil.Process(2833207).create_time()==1791370325.16
for name,h in HASHES.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json').read_bytes())
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='-1'
dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']])
result=subprocess.run([env['VENV_PYTHON'],str(out/'inspect_collection_coefficients.py')],cwd=out,env=env,capture_output=True,timeout=120)
(out/'stdout.txt').write_bytes(result.stdout);(out/'stderr.txt').write_bytes(result.stderr)
if result.returncode:print(result.stderr.decode(errors='replace'));result.check_returncode()
print(json.dumps({'status':json.loads(result.stdout),'physical':subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout,'host_available':psutil.virtual_memory().available,'unix':time.time(),'scripts':HASHES}))

PY
