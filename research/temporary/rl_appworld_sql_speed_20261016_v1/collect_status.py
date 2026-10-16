"""Read trial-owned original logs with the existing parser and physical mx-smi."""
import json
from pathlib import Path
import runpy
import subprocess
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-sql-native-speed-20261016-v1'
owner=runpy.run_path(str(ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'),run_name='speed_trial_original_log_reader')
status=AUDIT/'trial-status.json'
state=json.loads(status.read_bytes()) if status.exists() else {}
out=dict(collected_unix=time.time(),queue=state,jobs=[])
for job in state.get('tasks',[]):
    p=Path(job['run'])/'job.json';current=json.loads(p.read_bytes()) if p.exists() else job
    records,errors=owner['read_job_logs'](current)
    metrics,phases=owner['parse_logs'](records)
    try:
        process=psutil.Process(current['pid'])
        alive=process.create_time()==current['birth'] and process.status()!=psutil.STATUS_ZOMBIE
        workers=[dict(pid=c.pid,birth=c.create_time(),phase=c.name(),PSS_bytes=c.memory_full_info().pss)
                 for c in process.children(recursive=True) if 'WorkerDict' in c.name()] if alive else []
    except psutil.NoSuchProcess:alive=False;workers=[]
    path=Path(current['log'])
    tail=path.read_text(errors='replace').splitlines()[-14:] if path.exists() else []
    out['jobs'].append(dict(task=current['task'],pid=current['pid'],birth=current['birth'],alive=alive,
        status=current['status'],exit_code=current.get('exit_code'),workers=workers,
        completed_iterations=max((m['step'] for m in metrics),default=0),metrics=metrics,phases=phases,
        errors=errors,tail=[s[:1000] for s in tail],source_log=str(path)))
out['mx_smi']=subprocess.check_output(['mx-smi'],text=True)
memory=Path('/sys/fs/cgroup/memory')
out['memory_usage_bytes']=int((memory/'memory.usage_in_bytes').read_text())
out['memory_stat']={k:int(v) for k,v in (line.split() for line in (memory/'memory.stat').read_text().splitlines())}
out['protected_current_processes']=[]
for pid,birth in ((878971,1792091830.03),(3946004,1792041120.19)):
    try:
        p=psutil.Process(pid)
        out['protected_current_processes'].append(dict(pid=pid,birth=p.create_time(),identity_matches=p.create_time()==birth))
    except psutil.NoSuchProcess:out['protected_current_processes'].append(dict(pid=pid,alive=False))
(AUDIT/'latest-trial-snapshot.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(collected_unix=out['collected_unix'],queue_status=state.get('status'),
    jobs=[{k:v for k,v in j.items() if k not in ('metrics','phases')} for j in out['jobs']],
    physical_gpu=out['mx_smi'],memory_usage_GiB=out['memory_usage_bytes']/2**30),indent=2))
