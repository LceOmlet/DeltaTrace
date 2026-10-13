"""Read the exact candidate identity, owner logs and passive receipts only."""
import hashlib
import json
import runpy
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
parser_path=OUT/'original_plot_training_progress.py'
if parser_path.exists() and report.get('alive'):
    parser=runpy.run_path(str(parser_path),run_name='original_native_log_parser')
    runners=[w for w in report.get('workers',[]) if 'TaskRunner' in w['name']]
    paths=[]
    for runner in runners:
        for session in Path('/tmp/ray-tc-temporal-v1').glob('session_*'):
            if not session.is_symlink():paths.extend((session/'logs').glob(f'worker-*-{runner["pid"]}.out'))
    records=[];errors=[]
    for path in paths:
        r,e=parser['read_job_logs']({'log':str(path)});records.extend(r);errors.extend(e)
    metrics,phases=parser['parse_logs'](records)
    report.update(native_log_paths=[str(p) for p in paths],native_completed_iterations=max((m['step'] for m in metrics),default=0),
                  native_metrics=metrics[-5:],native_errors=errors[-10:],native_parser_SHA256=hashlib.sha256(parser_path.read_bytes()).hexdigest())
report['passive_credit_record_file_count']=len(list((RUN/'credit-records').glob('*/*.pt')))
dest=OUT/f'temporal-candidate-status-v2-{int(report["unix"])}.json' ;dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
