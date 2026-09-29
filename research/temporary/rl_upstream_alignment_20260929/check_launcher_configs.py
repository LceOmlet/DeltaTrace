"""Resolve the real launcher through native Hydra without starting training."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import time
import yaml

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
audit = root / 'receipts/upstream-alignment-20260929'
started = time.time()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tasks', nargs='+', default=['Sokoban', 'Webshop', 'AppWorld'])
args = parser.parse_args()
result = {'scope': __doc__, 'tasks': {}}
receipt = audit / 'native-reward-config-parity.json'
if args.tasks != ['Sokoban', 'Webshop', 'AppWorld'] and receipt.exists():
    result = json.loads(receipt.read_text())

def flatten(value, prefix=''):
    if not isinstance(value, dict):
        return {prefix: value}
    return {k: v for key, item in value.items()
            for k, v in flatten(item, f'{prefix}.{key}' if prefix else key).items()}

for task, cap in [('Sokoban', '1024'), ('Webshop', '4096'), ('AppWorld', '32256')]:
    if task not in args.tasks:
        continue
    configs = {}
    for method in ['ppo', 'dt']:
        env = dict(os.environ, METHOD=method, ENV_NAME=task, MAX_PROMPT=cap,
                   MAX_STEPS='40' if task == 'AppWorld' else '15',
                   DATA_ROOT=str(audit / 'config-data' / task),
                   CHECKPOINT_DIR=str(audit / 'unused-checkpoints' / task),
                   CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='')
        proc = subprocess.run(['bash', str(audit / 'dt-candidate/experiments/rl/run_verl_agent.sh'), '--cfg', 'job', '--resolve'],
                              env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (audit / f'config-{task}-{method}.log').write_text(proc.stdout)
        if proc.returncode:
            raise RuntimeError(f'{task}/{method} native config failed: {proc.returncode}')
        offset = proc.stdout.index('data:\n')
        configs[method] = yaml.safe_load(proc.stdout[offset:])
    left, right = (flatten(configs[name]) for name in ('ppo', 'dt'))
    differences = {key: [left.get(key), right.get(key)] for key in left.keys() | right.keys()
                   if left.get(key) != right.get(key)}
    assert set(differences) == {'algorithm.adv_estimator', 'trainer.experiment_name'}, differences
    assert left['actor_rollout_ref.actor.entropy_coeff'] == .001
    assert left['actor_rollout_ref.actor.clip_ratio_c'] == 3.0
    assert left['actor_rollout_ref.actor.use_invalid_action_penalty'] is True
    assert left['env.history_length'] == 2
    assert 'env.appworld_history_char_limit' not in left
    assert left['data.max_response_length'] == 512
    assert left['actor_rollout_ref.actor.ppo_mini_batch_size'] == 64
    assert left['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu'] == 4
    result['tasks'][task] = {'differences': differences, 'passed': True,
                             'shared_config': configs['dt']}
    print(task, 'native config parity passed', flush=True)
    (audit / 'native-reward-config-parity.json').write_text(json.dumps(result, indent=2))
result['seconds'] = time.time() - started
(audit / 'native-reward-config-parity.json').write_text(json.dumps(result, indent=2))
