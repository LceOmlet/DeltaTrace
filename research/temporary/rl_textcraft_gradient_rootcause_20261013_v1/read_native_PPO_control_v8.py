"""Read the exact native PPO job, original logs and completed owner receipts."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import re
import time

import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
version=sys.argv[1] if len(sys.argv)>1 else 'v8'
assert re.fullmatch(r'v[0-9]+',version)
job=json.loads((OUT/f'native-PPO-control-{version}-job.json').read_bytes())
run=Path(job['run'])
alive=False;birth=None;children=[]
if psutil.pid_exists(job['pid']):
    p=psutil.Process(job['pid']);birth=p.create_time()
    alive=birth==job['birth'] and p.status()!=psutil.STATUS_ZOMBIE
    if alive:
        children=[{'pid':c.pid,'birth':c.create_time(),'name':c.name(),'status':c.status(),
                   'phase_cmdline':c.cmdline() if c.name().startswith('ray::WorkerDict') else None}
                  for c in p.children(recursive=True) if c.is_running()]
report={'unix':time.time(),'scope':__doc__,'job':job,'alive':alive,'observed_birth':birth,
    'children':children,'new_model_calls':0,'logs':{},'ray_sessions':[],
    'completed_runtime_overrides':[],'native_metrics':[],'native_phases':[]}
for stream in ('stdout','stderr'):
    # A native resume has its own driver logs in the same run directory.
    path=Path(job.get(stream,run/f'driver.{stream}.log'))
    name=path.name
    if path.is_file():
        lines=path.read_text(errors='replace').splitlines()
        report['logs'][name]={'path':str(path),'lines':len(lines),'tail':'\n'.join(lines[-65:])[-18000:]}
sessions=sorted(Path('/tmp/ray').glob('session_*_'+str(job['pid'])))
for session in sessions:
    report['ray_sessions'].append(str(session))
    plot=runpy.run_path(str(OUT/'original_plot_training_progress.py'),run_name='native_PPO_original_log_parser')
    records,errors=plot['read_job_logs']({'ray_session':str(session)})
    metrics,phases=plot['parse_logs'](records)
    report['native_metrics']+=metrics;report['native_phases']+=phases
    report.setdefault('read_errors',[]).extend(errors)
for path in sorted((run/'worker-runtime-overrides').glob('*.json')):
    try:value=json.loads(path.read_bytes())
    except (ValueError,OSError):continue
    report['completed_runtime_overrides'].append({'path':str(path),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'value':value})
grpo=psutil.Process(3174448)
report['protected_GRPO_same_identity']=grpo.create_time()==1791793629.73
report['physical_mx_smi']=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=30).stdout
report['cgroup_memory']={str(p):p.read_text() for p in [Path('/sys/fs/cgroup/memory.current'),
    Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory.stat'),
    Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'),Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'),
    Path('/sys/fs/cgroup/memory/memory.stat'),Path('/sys/fs/cgroup/memory/memory.failcnt')]
    if p.is_file()}
report['worker_PSS']=[]
for c in children:
    if c['name'].startswith('ray::WorkerDict'):
        path=Path(f"/proc/{c['pid']}/smaps_rollup")
        if path.is_file():
            report['worker_PSS'].append({'pid':c['pid'],'birth':c['birth'],'phase':c['name'],
                'source':str(path),'values':[line for line in path.read_text().splitlines()
                                           if line.startswith(('Rss:','Pss:','Pss_Anon:','Pss_File:','Pss_Shmem:'))]})
report['disk']=subprocess.run(['df','-h',str(ROOT),'/tmp'],capture_output=True,text=True,timeout=30).stdout
path=OUT/f'native-PPO-control-observation-{version}.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
