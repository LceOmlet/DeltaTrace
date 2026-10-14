"""Read the already queued DT version client; never submit another RPC."""
import json
from pathlib import Path
import psutil
import time

OUT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
job=json.loads((OUT/'actual-DT-worker-bindings-v49-client.json').read_bytes())
alive=False
try:
    p=psutil.Process(job['pid']);alive=p.create_time()==job['birth'] and p.status()!=psutil.STATUS_ZOMBIE
except psutil.NoSuchProcess:pass
path=OUT/'actual-DT-worker-bindings-v49.json'
report=dict(unix=time.time(),client=job,alive=alive,complete=path.exists(),
    result=json.loads(path.read_bytes()) if path.exists() else None,
    stderr_tail=Path(job['stderr']).read_text(errors='replace')[-2000:])
print(json.dumps(dict(receipt=str(path),report=report)))
