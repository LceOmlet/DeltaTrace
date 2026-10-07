source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,sys,inspect,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');launch=json.loads((root/'receipts/direct-target-existing-pv-rule-20261008-v1-gdn-v2/launch.json').read_bytes());sp=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json';s=json.loads(sp.read_bytes());assert hashlib.sha256(sp.read_bytes()).hexdigest()==launch['source_sha256'];os.environ.update(s['environment']);r=Path(launch['memory_candidate']['candidate_dt_root']);env=json.loads(Path(launch['memory_candidate']['candidate_environment']).read_bytes())['qwen35'];sys.path[:0]=[str(r),os.environ.get('DT_OFFICIAL_ROOT') or env['official_root'],str(r/'clean/qwen35'),*s['pythonpath'].split(':'),env['ft_extension_root']]
import torch,finite_fla_gpu,qwen35_dense_finite_runner
from profiles import qwen35_gdn_symmetric
owners={name:dict(path=inspect.getfile(mod),sha256=hashlib.sha256(Path(inspect.getfile(mod)).read_bytes()).hexdigest()) for name,mod in [('finite_fla_gpu',finite_fla_gpu),('symmetric',qwen35_gdn_symmetric),('runner',qwen35_dense_finite_runner)]}
assert owners['finite_fla_gpu']['sha256']=='f1555736d32974668676eff4e040a500f4c2b907e6ac1b298f01dc2ba87f7db6';assert owners['symmetric']['sha256']=='dd6bbfff9aae4c679439af113a160e0bd8c4cf1b19bb17cab47a04ff7b0831d0'
print(json.dumps(dict(unix=time.time(),scope='Post-completion CPU resolution using the exact launch source/environment and sys.path; not a missing live-process import dump',owners=owners,cuda_initialized=torch.cuda.is_initialized(),source_sha256=launch['source_sha256']),indent=2))
PY
