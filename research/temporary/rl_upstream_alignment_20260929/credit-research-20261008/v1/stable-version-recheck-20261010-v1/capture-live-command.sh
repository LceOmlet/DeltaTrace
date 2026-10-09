source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import json,hashlib,psutil,time,subprocess
from pathlib import Path
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
f=r/'runs/textcraft-formal-stable-20261009-v1'
p=psutil.Process(982372);assert p.create_time()==1791553809.84
source_sha=hashlib.sha256((f/'source.json').read_bytes()).hexdigest()
assert source_sha=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs')
markers=['Traceback (most','OutOfMemoryError','FloatingPointError','[Skip the step]','Non-finite grad','grad_norm is not finite']
errors={}
for pid in [987808,989860,985585]:
 for suffix in ['out','err']:
  path=next(ray.glob('*-'+str(pid)+'.'+suffix))
  errors[str(path)]=[x for x in path.read_text(errors='replace').splitlines() if any(k in x for k in markers)]
lines=(f/'train.log').read_text(errors='replace').splitlines()
record=dict(unix=time.time(),pid=p.pid,birth=p.create_time(),source_sha256=source_sha,
 phase=[dict(pid=c.pid,name=c.name()) for c in p.children(recursive=True) if c.name().startswith('ray::WorkerDict')],
 progress=[x for x in lines if any(k in x for k in ['Rounds ','Training Progress:','step:'])][-8:],
 errors=errors,physical_mx_smi=subprocess.check_output(['mx-smi'],text=True),
 worker_PSS_bytes={str(pid):psutil.Process(pid).memory_full_info().pss for pid in [987808,989860]},
 host_available_bytes=psutil.virtual_memory().available,
 resource_scope='physical VRAM and PSS are phase samples, not peaks',
 model_calls_added=0,optimizer_calls_added=0,runtime_patches_added=0)
out=r/'receipts/stable-version-recheck-20261010-v1/live.json'
out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
