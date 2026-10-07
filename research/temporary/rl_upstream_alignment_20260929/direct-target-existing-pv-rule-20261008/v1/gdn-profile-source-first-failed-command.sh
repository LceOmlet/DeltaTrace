set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import os,json,inspect,importlib,sys,hashlib,subprocess
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json').read_bytes());c=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes())
env=dict(os.environ,**source['environment']);env['CUDA_VISIBLE_DEVICES']='';env.pop('MACA_VISIBLE_DEVICES',None)
dt=Path(c['candidate_dt_root']);q=json.loads(Path(c['candidate_environment']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(dt),env.get('DT_OFFICIAL_'/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']])
script="""import importlib,inspect,json,hashlib,torch
from pathlib import Path
m=importlib.import_module('deltatrace.profiles.qwen35_gdn_symmetric')
p=Path(inspect.getfile(m));print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text(),cuda_initialized=torch.cuda.is_initialized(),scope='CPU import/source audit only; no actor/model/DT/forward/update')))
"""
r=subprocess.run([env['VENV_PYTHON'],'-c',script],env=env,capture_output=True);r.check_returncode();print(r.stdout.decode())

PY
