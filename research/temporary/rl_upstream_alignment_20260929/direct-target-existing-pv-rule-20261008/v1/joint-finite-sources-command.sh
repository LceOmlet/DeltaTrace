set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import psutil
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
source=json.loads(source_path.read_bytes())
candidate=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes())
env=json.loads(Path(candidate['candidate_environment']).read_bytes())['qwen35']
os.environ.update(source['environment'])
os.environ['CUDA_VISIBLE_DEVICES']=''
dt=Path(candidate['candidate_dt_root'])
sys.path[:0]=[str(dt),os.environ.get('DT_OFFICIAL_ROOT') or env['official_root'],str(dt/'clean/qwen35'),*source['pythonpath'].split(':'),env['ft_extension_root']]
files=[]
for name in ('qwen35_answer_finite','qwen35_gdn_finite','profiles.qwen35_gdn_symmetric','compiled_logprob_seed','finite_fla_gpu','qwen35_dense_finite_runner','reward_readout'):
 spec=importlib.util.find_spec(name)
 assert spec is not None,name
 p=Path(spec.origin)
 files.append(dict(module=name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text()))
import torch
assert not torch.cuda.is_initialized()
held=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
print(json.dumps(dict(unix=time.time(),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),files=files,cuda_initialized=False,scope='CPU source resolution only; no model, DT, update, profile deployment or restart',textcraft_same_birth=psutil.Process(2833207).create_time()==1791370325.16,textcraft_release_present=[(held/('rank'+str(i)+'-release-update')).exists() for i in (0,1)])))

PY
