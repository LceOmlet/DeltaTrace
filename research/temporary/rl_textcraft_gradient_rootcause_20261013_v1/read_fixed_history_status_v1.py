"""Read the single gradient localization diagnostic; never submit or mutate it."""
import json
from pathlib import Path
import psutil
import time
OUT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
directory=OUT/'fixed-batch-historical-checkpoints-v1'
job=json.loads((directory/'job.json').read_bytes())
report={'unix':time.time(),'job':job,'result':json.loads((directory/'result.json').read_bytes()) if (directory/'result.json').exists() else None}
try:
 p=psutil.Process(job['pid']);report.update(alive=p.create_time()==job['birth'] and p.status()!=psutil.STATUS_ZOMBIE,
 observed_birth=p.create_time(),workers=[{'pid':c.pid,'birth':c.create_time(),'name':c.name()} for c in p.children(recursive=True) if 'ObservedWorker' in c.name()])
except psutil.NoSuchProcess:report['alive']=False
report['stderr_tail']=(directory/'stderr.log').read_text(errors='replace').splitlines()[-10:]
report['stdout_tail']=(directory/'stdout.log').read_text(errors='replace').splitlines()[-8:]
report['parameter_statistics']={p.name:json.loads(p.read_bytes()) for p in directory.glob('*-parameter-statistics.json')}
dest=OUT/f'fixed-history-observation-{int(report["unix"])}.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
