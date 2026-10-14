"""Read the exact new DTPO job, native phases and deployment receipts."""
import hashlib
import json
from pathlib import Path
import psutil
import runpy
import subprocess
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
job = json.loads((OUT/'DT-group-multiplier-v40-job.json').read_bytes())
run = Path(job['run'])
p = psutil.Process(job['pid'])
assert p.create_time() == job['birth']
report = dict(unix=time.time(),job=job,alive=p.is_running() and p.status()!=psutil.STATUS_ZOMBIE,
              children=[],logs={},native_metrics=[],native_phases=[],runtime_overrides=[],manifests={})
for c in p.children(recursive=True):
    try:
        report['children'].append(dict(pid=c.pid,birth=c.create_time(),name=c.name(),status=c.status()))
    except psutil.NoSuchProcess: pass
for name in ('driver.stdout.log','driver.stderr.log'):
    path=run/name
    if path.exists():
        lines=path.read_text(errors='replace').splitlines()
        report['logs'][name]=dict(path=str(path),lines=len(lines),tail='\n'.join(lines[-25:])[-11000:])
plot = runpy.run_path(str(OUT/'original_plot_training_progress.py'),run_name='original_plot_parser_v40')
for session in Path('/tmp/ray').glob('session_*_'+str(job['pid'])):
    records,errors=plot['read_job_logs']({'ray_session':str(session)})
    metrics,phases=plot['parse_logs'](records)
    report['native_metrics']+=metrics;report['native_phases']+=phases
    report['ray_session']=str(session)
    report.setdefault('read_errors',[]).extend(errors)
for directory in ('worker-runtime-overrides','runtime-overrides'):
    for path in sorted((run/directory).glob('*.json')):
        report['runtime_overrides'].append(dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                value=json.loads(path.read_bytes())))
for name in ('formal-training.json','active-training.json','active-source.json'):
    value=json.loads((ROOT/name).read_bytes())
    report['manifests'][name]=[dict(keys=list(j),identity={k:v for k,v in j.items()
        if k in ('pid','birth','method','task','output','run','stdout','stderr','status')}) for j in value['jobs']]
report['physical_mx_smi']=subprocess.check_output(['mx-smi'],text=True,timeout=30)
for pid,birth in [(3174448,1791793629.73),(2671393,1791949551.27)]:
    assert psutil.Process(pid).create_time()==birth
dest=OUT/'DT-group-multiplier-observation-v40.json'
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
