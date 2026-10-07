set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,os,sys,hashlib,inspect,importlib,psutil,subprocess,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json').read_bytes());c=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes());dt=Path(c['candidate_dt_root']);env=json.loads(Path(c['candidate_environment']).read_bytes())['qwen35']
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.pop('MACA_VISIBLE_DEVICES',None)
sys.path[:0]=[str(dt),os.environ.get('DT_OFFICIAL_ROOT') or env['official_root'],str(dt/'clean/qwen35'),*source['pythonpath'].split(':'),env['ft_extension_root']]
import torch
from finite_fla_gpu import verify_native_sources
chunk=verify_native_sources(env['native_stage_source_sha256'])
record=dict(unix=time.time(),cuda_initialized=torch.cuda.is_initialized(),expected_source_sha256=env['native_stage_source_sha256'],owners={},functions={})
for name,path in [('chunk',inspect.getfile(chunk)),('state',inspect.getfile(importlib.import_module('fla.ops.common.chunk_delta_h'))),('wy',inspect.getfile(importlib.import_module('fla.ops.gated_delta_rule.wy_fast')))]:
 p=Path(path);record['owners'][name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for name in ('chunk_gated_delta_rule','chunk_gated_delta_rule_fwd'):
 f=inspect.unwrap(getattr(chunk,name));record['functions'][name]=dict(signature=str(inspect.signature(f)),source=inspect.getsource(f))
record['physical_mx_smi']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
record['textcraft_same_birth']=psutil.Process(2833207).create_time()==1791370325.16
record['textcraft_release_present']=[(root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(i)+'-release-update')).exists() for i in (0,1)]
print(json.dumps(record))

PY
