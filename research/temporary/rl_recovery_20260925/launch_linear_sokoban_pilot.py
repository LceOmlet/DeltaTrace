"""Bounded full-horizon Sokoban training through the existing VERL launcher."""
import json
from pathlib import Path
import shlex
import subprocess
import time

root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
candidate=root/'candidates/linear-return-20260925'
original=next(j for j in json.loads((root/'formal-training.json').read_text())['jobs'] if j['task']=='Sokoban')
assert original['status']=='stopped_for_linear_return_readout'
assert not Path('/proc/3615368/cmdline').exists()
directory=root/'runs/linear-return-pilot/Sokoban'
directory.mkdir(parents=True,exist_ok=False)
settings={**original['settings'],
    'DT_ROOT':str(candidate),'DT_ENVIRONMENT_JSON':str(candidate/'environment.json'),
    'TRAIN_SIZE':'1','GROUP_SIZE':'4','TOTAL_EPOCHS':'2','VAL_SIZE':'1','VAL_DATA_SIZE':'1',
    'DT_RAY_NUM_CPUS':'8','TEST_FREQ':'-1','RESUME_MODE':'disable',
    'DATA_ROOT':str(directory/'data'),'ROLLOUT_DATA_DIR':str(directory/'rollouts'),
    'CHECKPOINT_DIR':str(directory/'checkpoints')}
script=['#!/bin/bash']+['export '+k+'='+shlex.quote(v) for k,v in settings.items()]
script+=['source "$DT_ROOT/experiments/rl/environments/metax.env.sh"',
         'bash "$DT_ROOT/experiments/rl/run_verl_agent.sh" '+shlex.join(original['native_hydra_overrides']),
         'rc=$?',"printf '%s\\n' \"$rc\" > "+shlex.quote(str(directory/'exit-code')), 'exit "$rc"']
(directory/'run.sh').write_text('\n'.join(script)+'\n')
subprocess.run(['bash','-n',str(directory/'run.sh')],check=True)
with (directory/'train.log').open('w') as log:
    proc=subprocess.Popen(['bash',str(directory/'run.sh')],stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(task='Sokoban',pid=proc.pid,started=time.time(),gpu=4,settings=settings,
         run_dir=str(directory),log=str(directory/'train.log'),
         scope='Two original trainer iterations, four actual rollouts each, unchanged horizon15/response1024/cap32768/actorDT B4. Not formal budget or task-performance evaluation.')
(directory/'job.json').write_text(json.dumps(job,indent=2)+'\n')
print(json.dumps({k:v for k,v in job.items() if k!='settings'}))
