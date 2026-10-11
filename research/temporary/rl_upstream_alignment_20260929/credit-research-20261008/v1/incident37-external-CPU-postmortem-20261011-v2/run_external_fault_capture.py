"""Preserve original workers only after saved real log-prob gradients are NaN.

No child-side hook/import/probe and no pre-fault signal. The existing native
incident recorder owns the loss artifact; this process only reads it. Original
GEMM captures are allowed to finish after the external CPU read and SIGCONT.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

import psutil
import torch

from host_mapping import readable_regions


ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
SHARED = ROOT / 'receipts/incident37-external-CPU-postmortem-source-20261011-v1'
assert hashlib.sha256((SHARED / 'capture_cpu_shard.py').read_bytes()).hexdigest() == 'f28dbbf2951d15d59f2435f61d4ad12b3fc540b67f28d643eb9c8a4f73c16fec'
sys.path.insert(0, str(SHARED))
import capture_cpu_shard as reader
reader.anonymous_regions = lambda pid: readable_regions(pid, reader.SHARD_BYTES)


def load(path):
    try:
        return json.loads(path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    source_path = ROOT / 'runs/textcraft-formal-stable-20261009-v1/source.json'
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == '1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
    source = load(source_path)
    original = ROOT / 'receipts/textcraft-native-incident37-competing-weight-capture-20261011-v1'
    entry = original / 'source/capture_native_incident37_GEMM_rows_20261011.py'
    assert hashlib.sha256(entry.read_bytes()).hexdigest() == '35262c5946fe86989425a68bbdba4414fd22ee13e000f404ef6c45a89889d3f8'
    paths = [original / 'source', ROOT / 'receipts/textcraft-native-incident37-async-head-capture-20261011-v1/source',
        ROOT / 'receipts/textcraft-native-incident37-original-exception-capture-20261011-v2/source',
        ROOT / 'receipts/textcraft-native-incident37-old-ref-DT-20261011-v1/source']
    job = ROOT / 'receipts/textcraft-native-incident37-external-fault-CPU-20261011-v2'
    mx = subprocess.run(['mx-smi'], capture_output=True, text=True, timeout=15, check=True).stdout
    assert 'no process found' in mx
    job.mkdir(exist_ok=False)
    (job / 'selection.json').write_bytes((original / 'selection.json').read_bytes())
    env = os.environ.copy()
    env.update({key:str(value) for key,value in source['environment'].items()})
    env.update(PYTHONPATH=':'.join([*(str(path) for path in paths), source['pythonpath']]), CUDA_VISIBLE_DEVICES='4,5', TORCH_TRACE=str(job / 'torch-trace'))
    env.pop('RAY_ADDRESS', None)
    assert not any(env.get(key) for key in ['LD_PRELOAD','TORCH_CUDA_SANITIZER','CUDA_LAUNCH_BLOCKING','MACA_LAUNCH_BLOCKING'])
    command = [sys.executable, str(entry), '--source', str(source_path), '--output', str(job / 'result')]
    with (job / 'driver.log').open('ab',buffering=0) as log:
        child = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    driver_birth = psutil.Process(child.pid).create_time()
    record = dict(unix=time.time(),supervisor_pid=os.getpid(),supervisor_birth=psutil.Process().create_time(),
        driver=dict(pid=child.pid,birth=driver_birth),command=command,phase='unchanged_original_incident_running_no_pre_fault_signals',
        workers=[],signals=[],CPU_captures=[],loss_read_errors=[],production_changes=0,formal_restarted=False,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('host_mapping.py'),SHARED/'capture_cpu_shard.py',entry]})
    def save():
        record['observed_unix']=time.time()
        (job/'job.json').write_text(json.dumps(record,indent=2)+'\n')
    def send(pid,birth,value):
        try:
            if abs(psutil.Process(pid).create_time()-birth)>=.001:return False
            os.kill(pid,value);record['signals'].append(dict(pid=pid,birth=birth,signal=value,unix=time.time()));return True
        except (psutil.NoSuchProcess,ProcessLookupError):return False
    stopped=[]
    inspected=set()
    try:
        save()
        while time.monotonic()-started<900:
            for path in (job/'result').glob('rank*-loss-inputs-microbatch*.pt'):
                if str(path)in inspected:continue
                try:
                    packet=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                    bad=not bool(torch.isfinite(packet['log_prob_gradient']).all())
                except Exception as error:
                    if len(record['loss_read_errors'])<10:record['loss_read_errors'].append(dict(path=str(path),error=repr(error)))
                    continue
                inspected.add(str(path))
                if not bad:continue
                record['failure_trigger']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),nonfinite_count=int((~torch.isfinite(packet['log_prob_gradient'])).sum()),unix=time.time())
                assert send(child.pid,driver_birth,signal.SIGSTOP);stopped.append((child.pid,driver_birth))
                for rank in (0,1):
                    state=load(job/'result'/f'rank{rank}.json');worker=dict(rank=rank,pid=state['pid'],birth=state['birth']);record['workers'].append(worker)
                    assert send(worker['pid'],worker['birth'],signal.SIGSTOP);stopped.append((worker['pid'],worker['birth']))
                record['phase']='actual_nonfinite_gradient_saved_workers_stopped_external_CPU_read';save()
                for worker in record['workers']:
                    begin=time.monotonic()
                    while psutil.Process(worker['pid']).status()!=psutil.STATUS_STOPPED:
                        if time.monotonic()-begin>5:raise TimeoutError('Worker did not acknowledge SIGSTOP')
                        time.sleep(.01)
                baseline=torch.load(ROOT/'receipts/textcraft-native-incident37-native-head-live-capture-20261011-v1/result/rank1-native-head-live-microbatch1.pt',map_location='cpu',weights_only=False,mmap=True)['tensors']['vocab_weights']
                anchors={row:baseline[124160+row,:32].contiguous().view(torch.uint8).numpy().tobytes() for row in reader.ANCHOR_ROWS}
                rank1=next(w for w in record['workers']if w['rank']==1)
                record['CPU_captures'].append(reader.capture(rank1['pid'],anchors,job/'external-rank1-CPU'))
                record['phase']='actual_failure_external_CPU_capture_completed';save();return
            if child.poll()is not None:
                record.update(phase='original_incident_ended_without_observed_nonfinite_gradient_no_capture',driver_returncode=child.returncode);return
            time.sleep(.02)
        raise TimeoutError('Diagnostic observation budget ended; not a production repair')
    except BaseException:
        record.update(phase='external_fault_supervisor_error',error=traceback.format_exc())
    finally:
        for pid,birth in reversed(stopped):send(pid,birth,signal.SIGCONT)
        save()
        try:record['driver_returncode']=child.wait(timeout=90)
        except subprocess.TimeoutExpired:record['driver_still_running']=True
        record.update(seconds=time.monotonic()-started,supervisor_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized());save()


if __name__=='__main__':main()
