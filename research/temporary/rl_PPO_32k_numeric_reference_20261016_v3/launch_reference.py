"""Launch the bounded numerical reference once using the recorded VENV."""
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-32k-numeric-reference-20261016-v3'
assert not (AUDIT/'same-session-launch.json').exists()
env=dict(os.environ);env['PPO_32K_SOURCE_COMMIT']=sys.argv[1]
with (AUDIT/'same-session-driver.log').open('xb') as log:
    p=subprocess.Popen([env['VENV_PYTHON'],str(AUDIT/'source/run_reference.py')],env=env,cwd=str(ROOT),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),source_commit=sys.argv[1],
    physical_gpus=[2,3],test_only_FA_deterministic=True,production_flags_unchanged=True,
    scope=__doc__,status='submitted_not_validated',log=str(AUDIT/'same-session-driver.log'))
(AUDIT/'same-session-launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
