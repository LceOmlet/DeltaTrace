"""Start the bounded native interface/capacity check on currently empty GPU2/3."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'
output=AUDIT/'native-behavior-job.json'
assert not output.exists(),'Read the existing bounded check instead of submitting it twice'
physical=subprocess.check_output(['mx-smi'],text=True,timeout=30)
assert not re.search(r'\|\s*[23]\s+\d+\s+',physical.split('| Process:')[-1]),'GPU2/3 occupied'
(AUDIT/'before-native-check-mx-smi.txt').write_text(physical)
env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='2,3'
env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(AUDIT/'native-worker-runtime-overrides')
for key in ('RAY_ADDRESS','RAY_TMPDIR','MACA_VISIBLE_DEVICES'):env.pop(key,None)
with (AUDIT/'native-behavior.stdout.log').open('wb') as stdout,(AUDIT/'native-behavior.stderr.log').open('wb') as stderr:
    p=subprocess.Popen([env['VENV_PYTHON'],'-u',str(AUDIT/'source/run_native_behavior_v5.py')],
        cwd=env['VERL_ROOT'],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),unix=time.time(),physical_gpus=[2,3],
    optimizer_updates=0,generated_rollouts=0,quality_evaluations=0,status='submitted_native_interface_and_capacity')
output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
