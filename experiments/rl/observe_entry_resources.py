"""Passive physical GPU/cgroup/PSS samples for explicitly named test jobs."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import psutil


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    jobs=[]
    for path in args.job:
        record=json.loads(path.read_text())
        proc=psutil.Process(record['pid'])
        jobs.append((path.parent,proc,proc.create_time()))
    with args.output.open('x') as stream:
        while True:
            states=[]
            for directory,driver,created in jobs:
                try:
                    live=driver.is_running() and driver.create_time()==created and driver.status()!=psutil.STATUS_ZOMBIE
                    if not live:continue
                    processes=[driver]+driver.children(recursive=True)
                    pss=[]
                    workers=[]
                    for process in processes:
                        try:
                            pss.append(process.memory_full_info().pss)
                            name=process.name()
                            if any(owner in name for owner in ('WorkerDict','TaskRunner','EngineCore','AsyncvLLM')):
                                workers.append(dict(pid=process.pid,name=name,pss_bytes=pss[-1]))
                        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                    states.append(dict(job=directory.name,pid=driver.pid,processes=len(pss),
                        pss_bytes=sum(pss),native_worker_names=workers))
                except psutil.NoSuchProcess:pass
            if not states:break
            output=subprocess.check_output(['mx-smi'],text=True)
            cg=Path('/sys/fs/cgroup/memory')
            stats=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
            # Host-wide pressure can kill a job even below its cgroup limit.
            host={key:int(value.split()[0])*1024 for key,value in
                (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                if key in ('MemTotal','MemFree','MemAvailable','AnonPages',
                    'Shmem','Unevictable','SUnreclaim','PageTables')}
            oom=dict(line.split() for line in (cg/'memory.oom_control').read_text().splitlines())
            metax={}
            for device in sorted(Path('/sys/class/drm').glob('renderD*/device')):
                values={}
                for name in ('mem_info_xtt_used','mem_info_xtt_total','mem_vram_used','mem_vram_total'):
                    field=device/name
                    if field.is_file():values[name]=int(field.read_text().split()[0])*1024
                if values:metax[device.parent.name]=values
            stream.write(json.dumps(dict(unix=time.time(),jobs=states,
                physical_gpu_mib=[int(x) for x in re.findall(r'(\d+)/65536 MiB',output)],
                host_memory_bytes=host,cgroup_oom_control={key:int(value) for key,value in oom.items()},
                # XTT may be a shared driver-wide counter; retain raw per-device values.
                metax_memory_bytes=metax,
                cgroup_failcnt=int((cg/'memory.failcnt').read_text()),
                cgroup_usage_bytes=int((cg/'memory.usage_in_bytes').read_text()),
                cgroup_rss_bytes=int(stats['total_rss']),cgroup_cache_bytes=int(stats['total_cache'])))+'\n')
            stream.flush()
            time.sleep(5)
