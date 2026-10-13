"""Read the exact candidate identity, owner logs and passive receipts only."""
import json
from pathlib import Path
import psutil
import time
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-global-token-mean-temporal-20261014-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
report={'unix':time.time(),'pid':launch['pid'],'expected_birth':launch['birth'],'run':str(RUN)}
try:
 p=psutil.Process(launch['pid']);report.update(alive=p.create_time()==launch['birth'] and p.status()!=psutil.STATUS_ZOMBIE,
   birth=p.create_time(),workers=[{'pid':c.pid,'birth':c.create_time(),'name':c.name()} for c in p.children(recursive=True) if 'WorkerDict' in c.name() or 'TaskRunner' in c.name()])
except psutil.NoSuchProcess:report['alive']=False
for name in ('driver.log','driver.stderr.log'):
 path=RUN/name
 with path.open('rb') as f:f.seek(max(0,path.stat().st_size-12000));tail=f.read().decode(errors='replace')
 report[name+'_tail']=tail.splitlines()[-12:]
receipt=RUN/'runtime-overrides/initial-state-and-temporal-capture-v2.json'
report['initial_capture_receipt']=json.loads(receipt.read_bytes()) if receipt.exists() else None
records=sorted((RUN/'temporal-update-records').glob('*/*-record.json'))
report['temporal_record_count']=len(records)
report['latest_temporal_records']=[json.loads(p.read_bytes()) for p in records[-2:]]
marker=RUN/'textcraft-dt/checkpoints/latest_checkpointed_iteration.txt'
report['native_checkpoint_marker']=marker.read_text() if marker.exists() else None
dest=OUT/f'temporal-candidate-status-{int(report["unix"])}.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
