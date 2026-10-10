source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
REMOTE='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/paper-role-implementation-check-20261010-v3'

import hashlib,json,psutil,subprocess,time,torch
from pathlib import Path
p=Path(REMOTE);l=json.loads((p/'launch.json').read_bytes());r=json.loads((p/'result.json').read_bytes())
assert r['phase']=='complete' and r['DT_calls']==2
files={x.name:dict(bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest())
       for x in [p/'result.json',p/'gdn-symmetric-v1.pt',p/'clean-v1.pt',p/'driver.log']}
peaks={}
for profile in ('gdn-symmetric-v1','clean-v1'):
 data=torch.load(p/(profile+'.pt'),map_location='cpu',weights_only=False)
 assert data['signed'].shape==(4,869) and torch.isfinite(data['signed']).all()
 peaks[profile]={k:data['details'].get(k) for k in ('peak_allocated','peak_reserved','complete_attribution_seconds_with_diagnostics')}
print(json.dumps(dict(unix=time.time(),files=files,owner_details=peaks,
 process_same_birth=bool(psutil.pid_exists(l['pid']) and psutil.Process(l['pid']).create_time()==l['birth']),
 physical_after=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,
 formal_same_birth=psutil.Process(982372).create_time()==1791553809.84,
 formal_source_sha256=hashlib.sha256((p.parents[1]/'runs/textcraft-formal-stable-20261009-v1/source.json').read_bytes()).hexdigest(),
 model_DT_optimizer_calls=0)))

PY
