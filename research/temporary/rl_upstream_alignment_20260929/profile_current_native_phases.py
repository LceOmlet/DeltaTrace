"""Observe live owner workers without changing training code or GPU state.

Nonblocking py-spy samples identify Python call sites, not CUDA kernel time.
Native logs supply completed phase timings. Read-only mx-smi samples supply
physical utilization/bandwidth, not an attribution of time to individual ops.
"""
from stage_environment_entry import remote, ROOT, ENTRY


remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,psutil,re,subprocess,time

root=Path('@ROOT@')
active=json.loads((root/'active-training.json').read_text())
out=root/'receipts/owner-b8-dispatch-20260930'/f'phase-cost-profile-{int(time.time())}'
out.mkdir()
read=lambda p:Path(p).read_text(errors='replace')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
jobs=[]; profilers=[]
for job in active['jobs']:
    p=psutil.Process(job['pid'])
    assert p.create_time()==job['observed_process_created_unix']
    assert p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
    workers=[]; task_logs=set()
    for child in p.children(recursive=True):
        try:
            files=set(f.path for f in child.open_files()
                      if '/worker-' in f.path and f.path.endswith(('.out','.err')))
            if 'TaskRunner' in child.name():task_logs.update(files)
            if 'WorkerDict' not in child.name():continue
            key=f'{job["task"].lower()}-{child.pid}'
            worker=dict(pid=child.pid,created_unix=child.create_time(),logs=sorted(files),
                        process_title=child.name(),raw=str(out/(key+'.raw')))
            log=(out/(key+'-pyspy.log')).open('wb')
            argv=['/opt/conda/bin/py-spy','record','--pid',str(child.pid),
                  '--duration','20','--rate','20','--format','raw','--threads',
                  '--idle','--nonblocking','--output',worker['raw']]
            sampler=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
            profilers.append((sampler,log,worker,argv))
            workers.append(worker)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    source=job.get('source_receipt',str(Path(job['output'])/'source.json'))
    jobs.append(dict(task=job['task'],driver=p.pid,created_unix=p.create_time(),
                     devices=job['devices'],source_receipt=source,source_sha256=sha(source),
                     output=job['output'],workers=workers,task_logs=sorted(task_logs)))
physical=[]
for _ in range(5):
    argv=['mx-smi','--show-usage','--show-pcie-bandwidth','--show-hbm-bandwidth','-j']
    r=subprocess.run(argv,capture_output=True,text=True,timeout=10)
    physical.append(dict(unix=time.time(),argv=argv,returncode=r.returncode,
                         stdout=r.stdout,stderr=r.stderr))
    time.sleep(4)
for sampler,log,worker,argv in profilers:
    sampler.wait(timeout=15);log.close()
    worker.update(profiler_returncode=sampler.returncode,profiler_argv=argv)
    p=psutil.Process(worker['pid'])
    worker['live_after']=p.create_time()==worker['created_unix'] and p.is_running()
    if Path(worker['raw']).exists():worker['raw_sha256']=sha(worker['raw'])
for job in jobs:
    metrics=[]; groups=[]; calls=[]
    for f in job['task_logs']:
        for line in read(f).splitlines():
            if line.startswith('step:'):
                metrics.append(dict(log=f,line=line))
            elif line.startswith(('[loop_transport]','[loop_trajectory]','[owner_trajectory]')):
                calls.append(dict(log=f,line=line))
    for worker in job['workers']:
        for f in worker['logs']:
            if not f.endswith('.out'):continue
            current=None
            for line in read(f).splitlines():
                if line.startswith('[DT EOS plan]'):
                    current=dict(worker=worker['pid'],log=f,plan=line,batches=[])
                    groups.append(current)
                elif current is not None and line.startswith('[DT EOS minibatch]'):
                    current['batches'].append(line)
                elif current is not None and line.startswith('[DeltaTrace readout]'):
                    data=json.loads(line.split('] ',1)[1])
                    current['report']={k:v for k,v in data.items()
                                       if k not in ('traces','minimum_log_ratio_batch')}
    job.update(native_metrics=metrics,dt_groups=groups,transport=calls)
record=dict(unix=time.time(),manifest=active['manifest'],jobs=jobs,physical=physical,
            scope='Read-only original logs plus 20-second nonblocking Python sampling; not GPU kernel profiling or a controlled speed comparison.')
(out/'observed.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(receipt=str(out/'observed.json'),sha256=sha(out/'observed.json'),
                     workers=[dict(pid=w['pid'],title=w['process_title'],live_after=w['live_after'])
                              for j in jobs for w in j['workers']])),flush=True)
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY))
