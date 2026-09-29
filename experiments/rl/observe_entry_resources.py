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
                    for process in processes:
                        try:pss.append(process.memory_full_info().pss)
                        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                    states.append(dict(job=directory.name,pid=driver.pid,processes=len(pss),pss_bytes=sum(pss)))
                except psutil.NoSuchProcess:pass
            if not states:break
            output=subprocess.check_output(['mx-smi'],text=True)
            cg=Path('/sys/fs/cgroup/memory')
            stats=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
            stream.write(json.dumps(dict(unix=time.time(),jobs=states,
                physical_gpu_mib=[int(x) for x in re.findall(r'(\d+)/65536 MiB',output)],
                cgroup_usage_bytes=int((cg/'memory.usage_in_bytes').read_text()),
                cgroup_rss_bytes=int(stats['total_rss']),cgroup_cache_bytes=int(stats['total_cache'])))+'\n')
            stream.flush()
            time.sleep(5)
