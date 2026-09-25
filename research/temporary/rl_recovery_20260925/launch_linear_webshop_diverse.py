"""Use the existing author launcher on four task groups after a zero-reward pilot.

No task is selected by its reward. The native fixed seed/task sampler and group8
are preserved; actor/DT minibatch4 and the 32768 budget stay unchanged.
"""
import json
from pathlib import Path
import shlex
import subprocess
import time

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
previous = root/'runs/linear-return-pilot/Webshop'
original = json.loads((previous/'job.json').read_text())
assert (previous/'exit-code').read_text().strip() == '0'
assert not Path(f"/proc/{original['worker_pid']}").exists()
assert (previous/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip() == '2'
formal = next(j for j in json.loads((root/'formal-training.json').read_text())['jobs'] if j['task']=='Webshop')
directory = root/'runs/linear-return-diverse/Webshop'
directory.mkdir(parents=True, exist_ok=False)
settings = {**original['settings'], 'TRAIN_SIZE':'4', 'GROUP_SIZE':'8',
    'TOTAL_EPOCHS':'2', 'RESUME_MODE':'disable',
    'DATA_ROOT':str(directory/'data'), 'ROLLOUT_DATA_DIR':str(directory/'rollouts'),
    'CHECKPOINT_DIR':str(directory/'checkpoints')}
script = ['#!/bin/bash']+['export '+k+'='+shlex.quote(v) for k,v in settings.items()]
script += ['source "$DT_ROOT/experiments/rl/environments/metax.env.sh"',
    'bash "$DT_ROOT/experiments/rl/run_verl_agent.sh" '+shlex.join(formal['native_hydra_overrides']),
    'rc=$?', "printf '%s\\n' \"$rc\" > "+shlex.quote(str(directory/'exit-code')), 'exit "$rc"']
(directory/'run.sh').write_text('\n'.join(script)+'\n')
subprocess.run(['bash','-n',str(directory/'run.sh')], check=True)
with (directory/'train.log').open('w') as log:
    proc = subprocess.Popen(['bash',str(directory/'run.sh')],stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
job = dict(task='Webshop', pid=proc.pid, started=time.time(), gpu=5,
    run_dir=str(directory), log=str(directory/'train.log'), checkpoint_dir=settings['CHECKPOINT_DIR'],
    settings=settings, status='initializing', supersedes=str(previous/'job.json'),
    scope='Two bounded native iterations, 4 task groups x group8. No positive-case filtering; not formal150x128 or task performance evaluation.')
(directory/'job.json').write_text(json.dumps(job,indent=2)+'\n')
original.update(status='completed_zero_reward_validation', completed_iterations=2,
    nonzero_credit_path_validated=False, next_validation=str(directory/'job.json'))
(previous/'job.json').write_text(json.dumps(original,indent=2)+'\n')
for name in ['active-training.json','active-source.json']:
    path=root/name;manifest=json.loads(path.read_text())
    manifest['bounded_validation']['manifests']=[str(directory/'job.json') if p==str(previous/'job.json') else p
        for p in manifest['bounded_validation']['manifests']]
    path.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in job.items() if k!='settings'}))
