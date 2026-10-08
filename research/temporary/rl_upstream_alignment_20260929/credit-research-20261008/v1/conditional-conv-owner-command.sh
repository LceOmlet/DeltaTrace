source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/conditional-conv-owner-20261009-v1'
import hashlib,json,os,re,subprocess
from pathlib import Path
source=Path(ROOT)/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
s=json.loads(source.read_bytes())
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
device=next(i for i in (4,5) if not re.search(r'^\|\s*'+str(i)+r'\s+\d+\s+\S',physical,re.M))
env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']=str(device)
env['PYTHONPATH']=':'.join([s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
out=Path(OUT)
(out/'inspection-environment.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 device=device,python=env['VENV_PYTHON'],pythonpath=env['PYTHONPATH'],physical_before=physical,
 model_calls=0,DT_calls=0,optimizer=0),indent=2)+'\n')
subprocess.run([env['VENV_PYTHON'],str(out/'inspect_conditional_conv_owner.py'),
 '--root',ROOT,'--output',str(out/'owner.json')],env=env,check=True)

PY
