"""One isolated native same-session numerical/capacity replay, not formal training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2'
assert not (AUDIT/'same-session-launch.json').exists(),'Keep any existing submission'
env=dict(os.environ);env['PPO_32K_SOURCE_COMMIT']=sys.argv[1]
with (AUDIT/'same-session-driver.log').open('xb') as log:
    p=subprocess.Popen([env['VENV_PYTHON'],str(AUDIT/'source/run_same_session.py')],
        env=env,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),source_commit=sys.argv[1],
    physical_gpus=[2,3],log=str(AUDIT/'same-session-driver.log'),scope=__doc__,status='submitted_not_validated')
(AUDIT/'same-session-launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
