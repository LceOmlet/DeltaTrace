"""Launch one isolated native validation case using the original VENV."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1'
case,commit=sys.argv[1:]
assert case in ('baseline-fit','baseline-repeat-fit','candidate-fit','reference-32k','candidate-32k')
assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
dest=AUDIT/case;dest.mkdir(exist_ok=True)
assert not (dest/'launched.json').exists(),'Preserve an already submitted case'
env=dict(os.environ);env['PPO_32K_SOURCE_COMMIT']=commit
log=dest/'driver.log'
with log.open('xb') as stream:
    child=subprocess.Popen([env['VENV_PYTHON'],str(AUDIT/'source/run_native_critic.py'),'--case',case],
        env=env,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(case=case,source_commit=commit,pid=child.pid,birth=psutil.Process(child.pid).create_time(),
        launched=time.time(),log=str(log),physical_gpus=[2,3],status='submitted_not_yet_validated',
        scope='Native numerical/capacity/efficiency validation only; not a formal training run')
    (dest/'launched.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
