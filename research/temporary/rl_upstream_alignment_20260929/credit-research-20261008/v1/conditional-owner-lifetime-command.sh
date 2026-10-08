set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
BINDINGS={'textcraft': {'source': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json', 'sha': '2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52', 'runner': {'path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/deltatrace/clean/qwen35/qwen35_dense_finite_runner.py', 'resolved_path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1/combined-capacity-b8-v1/qwen35_dense_finite_runner_row_candidate.py', 'sha256': '5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555'}}, 'appworld': {'source': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json', 'sha': '58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0', 'runner': {'path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace/clean/qwen35/qwen35_dense_finite_runner.py', 'resolved_path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace/clean/qwen35/qwen35_dense_finite_runner.py', 'sha256': '628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'}}}
MEMORY={'path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-consumed-cache-release-20261007-v1/deltatrace/clean/qwen35/qwen35_dense_finite_runner.py', 'sha256': '7d6f57f61ecde7ce3506b8ef58d61268859fc04357c35de99a1892928c4ba7a8'}

import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import psutil
import torch
def file(p):
 p=Path(p);b=p.read_bytes()
 return dict(path=str(p),resolved_path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),text=b.decode())
out=dict(unix=time.time(),tasks={},scope='Source/lifetime inspection only; no model, capture, kernel or candidate')
base=sys.path.copy()
for task,b in BINDINGS.items():
 source_path=Path(b['source']);src=file(source_path)
 assert src['sha256']==b['sha']
 source=json.loads(src['text']);os.environ.update(source['environment'])
 os.environ['CUDA_VISIBLE_DEVICES']=''
 root=Path(os.environ['DT_ROOT']).resolve()
 envp=Path(os.environ.get('DT_ENVIRONMENT_JSON',root/'environment.json'))
 env=json.loads(envp.read_bytes())['qwen35']
 sys.path[:]=[str(root),os.environ.get('DT_OFFICIAL_ROOT') or env['official_root'],
  str(root/'clean/qwen35'),*source['pythonpath'].split(':'),env['ft_extension_root'],*base]
 files=[]
 for name in ['qwen35_dense_finite_runner','native_dense_attention_capture','native_attention_capture']:
  spec=importlib.util.find_spec(name);assert spec is not None
  record=file(spec.origin);record['module']=name;files.append(record)
 assert files[0]['sha256']==b['runner']['sha256']
 out['tasks'][task]=dict(source_path=str(source_path),source_sha256=src['sha256'],files=files,
  environment_path=str(envp),environment_sha256=hashlib.sha256(envp.read_bytes()).hexdigest(),
  resource_settings={k:v for k,v in env.items() if k.startswith('dt_')})
out['memory_candidate_runner']=file(MEMORY['path'])
assert out['memory_candidate_runner']['sha256']==MEMORY['sha256']
hold=Path(BINDINGS['textcraft']['source']).parents[4]/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
out['formal_textcraft']=dict(pid=2833207,expected_birth=1791370325.16,
 actual_birth=psutil.Process(2833207).create_time(),
 release_exists=[(hold/('rank'+str(r)+'-release-update')).exists() for r in (0,1)])
out['cuda_initialized']=torch.cuda.is_initialized()
assert not out['cuda_initialized']
out['operations']=dict(model=0,DT=0,GPU=0,update=0)
print(json.dumps(out,ensure_ascii=False))

PY
