"""Read the exact bounded chain diagnostic, preserving its process and files."""
import json
from pathlib import Path
import time
import sys
import psutil

OUT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
DEST=OUT/(sys.argv[1] if len(sys.argv)>1 else 'native-gradient-chain-v4')
assert DEST.parent==OUT
job=json.loads((DEST/'job.json').read_bytes())
state=json.loads((DEST/'result.json').read_bytes()) if (DEST/'result.json').exists() else None
report={'unix':time.time(),'job':job,'state':state,'chain_reports':[]}
try:
    p=psutil.Process(job['pid'])
    report.update(alive=p.create_time()==job['birth'] and p.status()!=psutil.STATUS_ZOMBIE,
                  birth=p.create_time())
except psutil.NoSuchProcess:report['alive']=False
for path in sorted(DEST.glob('iteration*-rank[01]-chain.json')):
    report['chain_reports'].append({'path':str(path),'value':json.loads(path.read_bytes())})
for filename in ('stdout.log','stderr.log'):
    report[filename+'_tail']=(DEST/filename).read_text(errors='replace').splitlines()[-8:]
path=OUT/f'{DEST.name}-status-{int(report["unix"])}.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
