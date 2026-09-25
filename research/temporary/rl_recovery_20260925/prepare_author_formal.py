"""Prepare or explicitly launch the recorded paper budgets via the owner entry point."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--launch', action='store_true')
parser.add_argument('--release', default='64e5876')
args = parser.parse_args()
root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
release = root / 'releases' / args.release
receipt = json.loads((release / 'deployment-receipt.json').read_text())
scale = json.loads((release / 'experiments/rl/paper_scale.json').read_text())
owner = root / 'candidates/official-verl-20bd331'
# Validate the actual published files and current linear-return receipts, not
# the earlier per-event pilot. The trainer is still the existing owner entry.
manifest = json.loads((release / 'source-manifest.json').read_text())
for name, metadata in manifest['files'].items():
    assert hashlib.sha256((release/name).read_bytes()).hexdigest() == metadata['sha256'], name
capacity_path = root/'receipts/rollout-major-cost/linear-return-capacity-compact.json'
capacity = json.loads(capacity_path.read_text())
assert capacity['status'] == 'passed' and capacity['dt_input_tokens'] == 32768
assert capacity['minibatch'] == 4 and len(capacity['updates']) == 2
assert capacity['ppo_core_unchanged_from_previous_capacity']
assert hashlib.sha256((owner/'verl/trainer/ppo/core_algos.py').read_bytes()).hexdigest() == capacity['ppo_source_sha256']
tests_path = root/'receipts/rollout-major-cost/linear-return-cpu-tests.json'
assert json.loads(tests_path.read_text())['returncode'] == 0
runtime_verification = [str(capacity_path), str(tests_path)]
pilots = [json.loads((root/'runs'/run/task/'job.json').read_text()) for task,run in
          [('Sokoban','linear-return-pilot'), ('Webshop','linear-return-diverse'),
           ('AppWorld','linear-return-pilot')]]
ray_tmpdir = subprocess.check_output(
    ['bash', '-c', 'source "$DT_ROOT/experiments/rl/environments/metax.env.sh"\nprintf %s "$RAY_TMPDIR"'],
    env={**os.environ, 'DT_ROOT': str(release)}, text=True)
base = root / 'runs' / ('author-' + args.release + '-paper')
base.mkdir(exist_ok=True)
assert not (base / 'jobs.json').exists(), 'This paper run has already been launched.'
jobs = []
for task, gpu in [('Sokoban', 4), ('Webshop', 5), ('AppWorld', 6)]:
    folder = base / task
    folder.mkdir(exist_ok=True)
    budget = scale['tasks'][task]
    settings = {**scale['fixed'], **budget['env'],
        'DT_ROOT': str(release), 'DT_ENVIRONMENT_JSON': str(release / 'environment.json'),
        'VERL_ROOT': str(owner),
        'WEBSHOP_ROOT': str(owner / 'agent_system/environments/env_package/webshop/webshop'),
        'ENV_NAME': task, 'PARAM_OFFLOAD': 'True',
        'CUDA_VISIBLE_DEVICES': str(gpu), 'MACA_VISIBLE_DEVICES': str(gpu),
        'PYTHONUNBUFFERED': '1', 'DATA_ROOT': str(folder / 'data'),
        'ROLLOUT_DATA_DIR': str(folder / 'rollouts'), 'CHECKPOINT_DIR': str(folder / 'checkpoints'),
        'APPWORLD_PORT_FILE': str(root / 'third_party/appworld-42b5bcf3cd334fee33f0c37c02070a9f5807add5/appworld_ports_formal.ports'),
    }
    if task == 'AppWorld':
        settings['MAX_PROMPT'] = '32768'  # Existing launcher reserves response/readout.
    assert int(settings['TRAIN_SIZE']) * int(settings['GROUP_SIZE']) == budget['episodes_per_iteration']
    assert int(settings['TOTAL_EPOCHS']) == budget['iterations']
    script = ['#!/bin/bash']
    script += ['export ' + key + '=' + shlex.quote(value) for key, value in settings.items()]
    script += ['source "$DT_ROOT/experiments/rl/environments/metax.env.sh"',
        'bash "$DT_ROOT/experiments/rl/run_verl_agent.sh" ' + shlex.join(scale['native_hydra_overrides']),
        'rc=$?', "printf '%s\\n' \"$rc\" > " + shlex.quote(str(folder / 'exit-code')), 'exit "$rc"']
    (folder / 'run.sh').write_text('\n'.join(script) + '\n')
    subprocess.run(['bash', '-n', str(folder / 'run.sh')], check=True)
    jobs.append(dict(task=task, gpu=gpu, run_dir=str(folder), ray_tmpdir=ray_tmpdir,
        checkpoint_dir=settings['CHECKPOINT_DIR'], log=str(folder / 'train.log'),
        exit_file=str(folder / 'exit-code'), settings=settings, paper_budget=budget,
        native_hydra_overrides=scale['native_hydra_overrides']))
prepared = dict(status='prepared_not_started', source_commit=receipt['source_commit'],
    dt_root=str(release), environment_json=str(release / 'environment.json'),
    scope=scale['scope'], continuous_pilot_manifests=[str(Path(j['run_dir'])/'job.json') for j in pilots], jobs=jobs)
prepared['runtime_changes_since_pilot'] = []
prepared['runtime_change_verification'] = runtime_verification
prepared['method'] = 'Current PLAN complete-return EOS DT: at most one request per active response, actor/DT minibatch4, original PPO.'
(base / 'prepared.json').write_text(json.dumps(prepared, indent=2) + '\n')
if not args.launch:
    print(json.dumps({'prepared': str(base / 'prepared.json'), 'status': prepared['status'],
        'budgets': {j['task']: [j['paper_budget']['iterations'], j['paper_budget']['episodes_per_iteration']] for j in jobs}}))
    raise SystemExit(0)

# The already-running continuous checks must finish; do not restart or kill them.
for job in pilots:
    assert (Path(job['run_dir'])/'exit-code').read_text().strip() == '0', job['task']
    assert int((Path(job['checkpoint_dir']) / 'latest_checkpointed_iteration.txt').read_text()) == 2, job['task']
    assert not Path('/proc', str(job['pid'])).exists(), job['task']
    log_text = Path(job['log']).read_text(errors='replace')
    assert any(float(x)>0 for x in re.findall(r'actor/grad_norm:([0-9.]+)', log_text)), job['task']
    assert any(int(x)>0 for x in re.findall(r'nonzero_reward_events[^0-9]+([0-9]+)', log_text)), job['task']
for gpu in (4, 5, 6):
    output = subprocess.check_output(['mx-smi', '-i', str(gpu)], text=True)
    memory = re.search(r'(\d+)\s*/\s*(\d+)\s*MiB', output)
    assert memory and int(memory[1]) < 2048, (gpu, output)
    assert not re.search(r'\|\s*' + str(gpu) + r'\s+\d+\s+\S', output), (gpu, output)
for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
    path = root / name
    if path.exists():
        (base / ('previous-' + name)).write_bytes(path.read_bytes())
training = {k:v for k,v in prepared.items() if k != 'jobs'}
training.update(started=time.time(), status='starting', jobs=[])
for job in jobs:
    with Path(job['log']).open('w') as log:
        process = subprocess.Popen(['bash', str(Path(job['run_dir']) / 'run.sh')],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    job.update(pid=process.pid, started=time.time(), status='starting_not_yet_validated')
    training['jobs'].append(job)
    (Path(job['run_dir']) / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
    for path in (base / 'jobs.json', root / 'formal-training.json', root / 'active-training.json'):
        path.write_text(json.dumps(training, indent=2) + '\n')
source = {k:v for k,v in training.items() if k != 'jobs'}
source.update(files=manifest['files'], verl_root=str(owner))
(root / 'active-source.json').write_text(json.dumps(source, indent=2) + '\n')
print(json.dumps({'status':'launched_not_yet_validated', 'jobs':[
    {k:j[k] for k in ('task','gpu','pid','run_dir')} for j in training['jobs']]}))
