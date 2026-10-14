"""Check a saved actual native update input with the original GRPO component.

CPU only. No model, DT, backward, optimizer, rollout or validation call.
This verifies deployment of the approved integration, not a long-term trend.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN = ROOT/'runs/textcraft-dt-grpo-multiplier-no-whiten-20261014-v1'
DEST = OUT/'live-native-update-input-v61'
env = json.loads((RUN/'launch.json').read_bytes())['effective_environment']
sys.path[:0] = env['PYTHONPATH'].split(':')
os.environ.update(env)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

result_path = DEST/'actual-native-input-verification-v62.json'
client_path = DEST/'verifier-client.json'
if '--submit' in sys.argv or '--read' in sys.argv:
    if '--submit' in sys.argv and not client_path.exists() and not result_path.exists():
        job = json.loads((OUT/'DT-group-multiplier-v40-job.json').read_bytes())
        assert psutil.Process(job['pid']).create_time() == job['birth']
        stdout, stderr = DEST/'verifier.stdout.log', DEST/'verifier.stderr.log'
        with stdout.open('wb') as out, stderr.open('wb') as err:
            p = subprocess.Popen([env['VENV_PYTHON'], '-u', __file__, '--wait-for-capture'],
                stdout=out, stderr=err, env=dict(os.environ), start_new_session=True)
        client_path.write_text(json.dumps(dict(pid=p.pid, birth=psutil.Process(p.pid).create_time(),
            stdout=str(stdout), stderr=str(stderr), driver_pid=job['pid'], driver_birth=job['birth'],
            source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    client = json.loads(client_path.read_bytes()) if client_path.exists() else None
    alive = False
    if client:
        try:
            p = psutil.Process(client['pid'])
            alive = p.create_time() == client['birth'] and p.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    report = dict(unix=time.time(), client=client, client_alive=alive,
        result=json.loads(result_path.read_bytes()) if result_path.exists() else None,
        stderr_tail=Path(client['stderr']).read_text(errors='replace')[-2000:] if client else '')
    print(json.dumps(dict(receipt=str(result_path), report=report)))
    raise SystemExit(0)

if '--wait-for-capture' in sys.argv:
    job = json.loads((OUT/'DT-group-multiplier-v40-job.json').read_bytes())
    deadline = time.monotonic()+7200
    while not all((DEST/f'approved_DTPO_actual_update/rank{r}-capture.json').exists() for r in (0, 1)):
        assert psutil.Process(job['pid']).create_time() == job['birth']
        assert time.monotonic() < deadline, 'Specific one-shot input capture did not arrive within two hours'
        time.sleep(20)

import torch
import dt_training_batch
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
from dt_training_batch import CREDIT_KEYS

torch.set_num_threads(4)
binding = {}
for obj, expected in [
    (core_algos, 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'),
    (compute_advantage, '63b0d623380bab3173eaf2c7148bddf21b557827625bd5eff25c5b4faa05daa4'),
    (dt_training_batch, 'a4e1f70b42082972324199b328975fb87ec6600ab9a572d5ca2b2040bebc54f7'),
]:
    path = Path(inspect.getsourcefile(obj))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected
    binding[str(path)] = digest

captures = [json.loads((DEST/f'approved_DTPO_actual_update/rank{rank}-capture.json').read_bytes())
            for rank in (0, 1)]
assert all(r['status'] == 'saved' and r['original_update_restored'] for r in captures)
assert len({r['completed_step_before_update'] for r in captures}) == 1
assert {r['pid']: r['birth'] for r in captures} == {1426979: 1791977192.37, 1429032: 1791977210.63}
for r in captures:
    assert hashlib.sha256(Path(r['input']).read_bytes()).hexdigest() == r['input_sha256']
data = DataProto.concat([DataProto.load_from_disk(r['input']) for r in captures])
assert len(data) == 256
original_advantages = data.batch['advantages'].clone()
original_rewards = data.batch['token_level_rewards'].clone()
stored_multiplier = data.batch['dt_grpo_multipliers'].clone()
group_view = data.select(batch_keys=['responses', 'token_level_rewards', 'response_mask', 'loss_mask'],
                         non_tensor_batch_keys=['uid', 'traj_uid'], deepcopy=True)
group_view.batch = group_view.batch.clone(recurse=False)
group = compute_advantage(group_view, adv_estimator=AdvantageEstimator.GRPO,
                          norm_adv_by_std_in_grpo=True, multi_turn=True)
mask = data.batch['response_mask'].bool()
first_policy = mask.long().argmax(-1, keepdim=True)
native_multiplier = group.batch['advantages'].gather(1, first_policy).squeeze(1)
assert torch.equal(group.batch['advantages'][mask], native_multiplier[:, None].expand_as(mask)[mask])
zero = native_multiplier.eq(0)
positive_zero = zero & original_rewards.sum(-1).gt(0)
checks = dict(
    actual_multiplier_equals_native_GRPO=torch.equal(stored_multiplier, native_multiplier),
    actual_advantages_equal_DT_output=torch.equal(original_advantages, data.batch['dt_token_advantages']),
    actual_zero_group_advantages_zero=not bool(original_advantages[zero].any()),
    actual_zero_group_credit_fields_zero={k: not bool(data.batch[k][zero].any()) for k in CREDIT_KEYS},
    actual_observation_padding_advantages_zero=not bool(original_advantages[~mask].any()),
    readonly_native_GRPO_did_not_modify_raw_rewards=torch.equal(data.batch['token_level_rewards'], original_rewards),
    readonly_native_GRPO_did_not_modify_actual_advantages=torch.equal(data.batch['advantages'], original_advantages),
)
if 'token_level_scores' in data.batch:
    checks['actual_rewards_equal_native_scores'] = torch.equal(original_rewards, data.batch['token_level_scores'])

report = dict(unix=time.time(), scope=__doc__, status='checked_actual_native_PPO_input',
    completed_step_before_update=captures[0]['completed_step_before_update'], rows=len(data),
    zero_group_rows=int(zero.sum()), positive_reward_zero_group_rows=int(positive_zero.sum()),
    zero_reward_nonzero_group_rows=int((original_rewards.sum(-1).eq(0) & ~zero).sum()),
    actual_advantage_norm=float(original_advantages.double().norm()),
    multiplier_max_abs_difference=float((stored_multiplier.double()-native_multiplier.double()).abs().max()),
    checks=checks, source_bindings=binding, captures=captures,
    native_GRPO_entry=str(inspect.getsourcefile(compute_advantage))+':compute_advantage(GRPO)',
    no_shadow_GRPO=True, model_calls=0, DT_calls=0, optimizer_calls=0,
    rootcause_repair_verified=False,
    limits=['Actual zero-group transport and no-whitening regression does not establish long-term gradient stability.',
            'The already completed late-state update intervention remains conditional on old inputs, weights and optimizer history.'])
report['integration_regression_passed'] = all(
    all(value.values()) if isinstance(value, dict) else value for value in checks.values())
assert not torch.cuda.is_initialized()
path = result_path
assert not path.exists()
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(receipt=str(path), report=report)))
