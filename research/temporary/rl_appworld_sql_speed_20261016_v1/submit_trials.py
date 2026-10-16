"""Start the isolated native trial queue once; existing formal jobs stay intact."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-sql-native-speed-20261016-v1'
path=AUDIT/'queue-job.json'
assert not path.exists(),'Read the existing queue receipt instead of submitting twice.'
assert (AUDIT/'CPU-verification.json').exists()
env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';env.pop('MACA_VISIBLE_DEVICES',None)
log=(AUDIT/'queue.log').open('wb')
process=subprocess.Popen([env['VENV_PYTHON'],str(AUDIT/'source/run_trials.py'),sys.argv[1]],
    env=env,cwd=AUDIT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
log.close()
job=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),started_unix=time.time(),
    source_commit=sys.argv[1],physical_gpus=[2,3],scope='Two native limited speed trials; not formal training',
    log=str(AUDIT/'queue.log'),status='submitted')
path.write_text(json.dumps(job,indent=2)+'\n');print(json.dumps(job))
