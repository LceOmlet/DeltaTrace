"""Run two bounded original VERL jobs sequentially on reserved physical GPU2/3."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-native-PPO-32k-speed-20261016-v3'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
performance=read(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/status.json')
assert performance['status']=='completed' and performance['exact_transport']=='passed'
assert next(item for item in performance['comparisons'] if item['context']==32768)['median_ratio']<=1.0
short=read(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v4/performance-exec3/status.json')
assert short['status']=='completed' and short['native_short_dispatch']=='passed'
assert short['comparison']['median_ratio']<=1.0, 'No short-input efficiency regression accepted'

index=read(AUDIT/'prepared-index.json')
assert (AUDIT/'CPU-verification.json').exists()
import xml.etree.ElementTree as ET
xml=ET.parse(AUDIT/'SQL-boundary-tests.xml').getroot()
assert sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in xml.iter('testsuite'))==0
state=dict(version=index['version'],pid=os.getpid(),birth=psutil.Process().create_time(),
    started_unix=time.time(),source_commit=sys.argv[1],status='starting',tasks=[],
    physical_gpus=[2,3],existing_formal_training_mutations=0)
status=AUDIT/'trial-status.json';write(status,state)
for item in index['tasks']:
    record=read(item['path']);run=Path(record['run']);run.mkdir(parents=True,exist_ok=True)
    for path,expected in record['source_bindings'].items():assert sha(path)==expected,(path,'source changed')
    smi=subprocess.check_output(['mx-smi'],text=True)
    occupied=[line for line in smi.splitlines() if re.match(r'\|\s*(?:2|3)\s+\d+\s+\S+',line)]
    if occupied:
        state.update(status='reserved_GPU_busy_no_training_stopped',blocked_task=record['task'],physical_processes=occupied)
        write(status,state);raise RuntimeError('Physical GPU2/3 occupied; no job was signalled')
    env=dict(record['effective_environment']);env['CUDA_VISIBLE_DEVICES']='2,3';env.pop('MACA_VISIBLE_DEVICES',None)
    log=run/'driver.log';stream=log.open('wb')
    process=subprocess.Popen(record['argv'],cwd=run,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    job=dict(task=record['task'],pid=process.pid,birth=psutil.Process(process.pid).create_time(),
        started_unix=time.time(),run=str(run),log=str(log),physical_gpus=[2,3],
        source_commit=sys.argv[1],baseline_v5_commit=record['baseline_v5_commit'],
        status='running_initialization_not_completed_iteration',budget=record['budget'],formal=False,
        prepared=item['path'],source_bindings=record['source_bindings'])
    write(run/'job.json',job);write(run/'launch.json',record)
    state['tasks'].append(job);state['status']='running_'+record['task'];write(status,state)
    code=process.wait();stream.close()
    job.update(exit_code=code,ended_unix=time.time(),status='native_driver_completed' if code==0 else 'native_driver_failed')
    write(run/'job.json',job);write(status,state)
    # Ray/VERL own worker shutdown. Do not send global stop/kill signals.
    for _ in range(12):
        smi=subprocess.check_output(['mx-smi'],text=True)
        if not any(re.match(r'\|\s*(?:2|3)\s+\d+\s+\S+',line) for line in smi.splitlines()):break
        time.sleep(5)
state.update(status='all_native_drivers_finished',completed_unix=time.time())
write(status,state)
