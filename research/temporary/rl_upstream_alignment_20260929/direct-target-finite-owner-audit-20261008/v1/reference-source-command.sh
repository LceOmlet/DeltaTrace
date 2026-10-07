set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,hashlib,re,ast,time
from pathlib import Path
root=Path("/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922")
source=json.loads((root/"runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json").read_bytes());dt=Path(source['environment']['DT_ROOT'])
dirs=[dt/'clean/qwen3',root/'candidates/appworld-row-cuts-finite-20261007-v1',root/'receipts/direct-target-numerics-20261007-v4']
records=[]
for directory in dirs:
 for p in directory.rglob('*'):
  if not p.is_file() or p.suffix not in ('.py','.cpp','.cu','.md','.json') or p.stat().st_size>500000:continue
  if directory!=dirs[0] and not any(s in p.name.lower() for s in ('test','reference','finite','verify','receipt')):continue
  text=p.read_text(errors='replace')
  if not any(s in text for s in ('logarithmic_mean','finite_attention','attention_ref','P1','content_P1')):continue
  lines=text.splitlines();hits=[{'line':i+1,'text':line} for i,line in enumerate(lines) if any(s in line for s in ('def ','assert ','tolerance','attention_ref','finite_attention','content_P1'))]
  records.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'relevant_lines':hits[:65]})
print(json.dumps({'unix':time.time(),'scope':'Read installed finite-FA reference/test owners only','directories':list(map(str,dirs)),'records':records}))

PY
