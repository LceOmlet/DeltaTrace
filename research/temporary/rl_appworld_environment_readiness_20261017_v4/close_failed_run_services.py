"""Close only proven orphan servers of the two recorded, exited formal runs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import site
import sys
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-environment-readiness-20261017-v4'
versions=['appworld-dtpo-reward-target-exponential-formal-20261017-v3',
          'appworld-native-PPO-formal-20261017-v1']
sources=[json.loads((ROOT/'runs'/v/'source.json').read_bytes()) for v in versions]
os.environ.update(sources[0]['effective_environment'])
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.pop('MACA_VISIBLE_DEVICES',None)
sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep)
site.addsitedir(os.environ['LOOP_EXTRAS'])
import psutil
from phi_agents.appworld.server import stop_appworld_environment_server, force_stop_appworld_environment_server

for source in sources:
    try:
        process=psutil.Process(source['pid'])
        assert process.create_time()!=source['birth'] or process.status()==psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
targets={str(ROOT/'runs'/v/'appworld-dt/worker-runtime-overrides') for v in versions}
selected=[]
for process in psutil.process_iter(['pid','create_time','cmdline','ppid']):
    try:
        cmd=process.info['cmdline'] or []
        if not(any('appworld' in arg for arg in cmd[:2]) and 'serve' in cmd and 'environment' in cmd):continue
        run=process.environ().get('DT_FSDP_EVENT_WAIT_RECEIPTS')
        if run in targets:
            selected.append(dict(pid=process.pid,birth=process.create_time(),run=run,
                                 ppid=process.ppid(),PSS_bytes=process.memory_full_info().pss))
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
record=dict(started=time.time(),status='closing_proven_orphans',failed_versions=versions,selected=selected,
            original_owner_stop_api=stop_appworld_environment_server.__module__,
            active_training_signals=0,environment_or_training_configuration_changes=0)
path=AUDIT/'failed-run-server-cleanup.json'
path.write_text(json.dumps(record,indent=2)+'\n')


def close(identity):
    try:
        process=psutil.Process(identity['pid'])
        assert process.create_time()==identity['birth']
        assert process.environ().get('DT_FSDP_EVENT_WAIT_RECEIPTS')==identity['run']
        forced=False
        try:stop_appworld_environment_server(process)
        except psutil.TimeoutExpired:
            forced=True
            force_stop_appworld_environment_server(process)
        return dict(**identity,alive=psutil.pid_exists(identity['pid']),forced=forced)
    except psutil.NoSuchProcess:
        return dict(**identity,alive=False,already_exited=True)


with ThreadPoolExecutor(max_workers=16) as executor:
    record['results']=list(executor.map(close,selected))
record.update(finished=time.time(),status='passed' if not any(x['alive'] for x in record['results']) else 'remaining_orphans')
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],selected=len(selected),alive=sum(x['alive'] for x in record['results']),
                     orphan_PSS_GiB=sum(x['PSS_bytes'] for x in selected)/2**30,active_training_signals=0)))
