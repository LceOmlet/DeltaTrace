"""Inspect native Qwen derivative boundaries and current resources; no model."""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
import psutil
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
rows = {}
for name in ('Qwen3_5RMSNorm','Qwen3_5DecoderLayer','Qwen3_5TextModel'):
    cls = getattr(qwen,name,None)
    if cls is not None:
        path = Path(inspect.getsourcefile(cls))
        rows[name] = {'file':str(path),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'source':inspect.getsource(cls)}
job = json.loads((OUT/'historical-target-derivatives-v1/job.json').read_bytes())
try:
    p=psutil.Process(job['pid'])
    identity={'pid':p.pid,'birth':p.create_time(),'expected_birth':job['birth'],
              'alive':p.create_time()==job['birth'] and p.status()!=psutil.STATUS_ZOMBIE}
except psutil.NoSuchProcess:
    identity={'pid':job['pid'],'expected_birth':job['birth'],'alive':False}
report={'unix':time.time(),'scope':__doc__,'classes':rows,'prior_diagnostic':identity,
        'prior_diagnostic_state':json.loads((OUT/'historical-target-derivatives-v1/result.json').read_bytes())['phase'],
        'mx_smi':subprocess.check_output(['mx-smi'],text=True)}
path=OUT/'native-gradient-chain-entry-v4.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
