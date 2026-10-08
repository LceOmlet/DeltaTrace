source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-layer-collection-inputs-20261008-v1'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
HASHES={'layer-collection-inputs.json': 'e835680d4b527829fae2b2c116c58b198fb3eda7bf12da137e11871e411f87a5', 'inspect_layer_collection_inputs.py': 'cf2585cdafde26cebfd38535f657dc6f810bc3b1e1029179f649689372f9de0d'}
import json,hashlib,os,subprocess,time
from pathlib import Path
out=Path(OUT);root=Path(ROOT)
result={'scope':'CPU exact-owner preparation; no model or GPU work','observed_unix':time.time(),'tasks':{}}
for name,h in HASHES.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
for task in ('textcraft','appworld'):
 p=root/'runs/direct-target-prefix-runtime-20261007-v1'/task/(task+'-dt')/'source.json'
 source=json.loads(p.read_bytes());env=dict(os.environ,**source['environment'])
 env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='-1'
 env['PYTHONPATH']=':'.join([str(out),source['pythonpath'],str(Path(source['dt_root'])/'clean/qwen35')])
 target=out/(task+'-cpu-inventory.json')
 command=[env['VENV_PYTHON'],str(out/'inspect_layer_collection_inputs.py'),str(out/'layer-collection-inputs.json'),task,str(p),str(target)]
 run=subprocess.run(command,env=env,cwd=out,capture_output=True,timeout=60)
 (out/(task+'-cpu.stderr.txt')).write_bytes(run.stderr)
 result['tasks'][task]={'returncode':run.returncode,'stdout':run.stdout.decode(errors='replace'),'stderr':run.stderr.decode(errors='replace')[-4000:],'command':command}
 if run.returncode==0:
  result['tasks'][task]['receipt']=json.loads(target.read_bytes())
  result['tasks'][task]['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
print(json.dumps(result))

PY
