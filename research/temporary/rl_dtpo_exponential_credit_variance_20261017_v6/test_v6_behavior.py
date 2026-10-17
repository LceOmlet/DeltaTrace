"""CPU contracts using the pinned VERL statistic and its real DP transport.

Sentinel DT values test composition and transport, not DT prediction accuracy.
No model, generation, optimizer or training checkpoint is invoked here.
"""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6'
sys.path.append(str(ROOT / 'receipts/textcraft-reward-target-exponential-20261016-v5/source'))
sys.path.append(str(ROOT / 'receipts/textcraft-weighted-reward-targets-20261015-v4/source'))
from test_multi_target_v5 import EventTransportWorker, events
from test_direct_target_transport import artifact, native_training_data
from dt_training_batch import CREDIT_KEYS, compute_direct_target_credit
from verl.utils.torch_functional import masked_var

MODE = 'reward_target_exponential_variance_v6'


def native_mask(data):
    rows = []
    for item in data.non_tensor_batch['dt_direct_target_artifact']:
        rows.append([False if position < 0 else item['policy_mask'][position]
                     for position in item['retained_response_positions']])
    return torch.tensor(rows, dtype=torch.bool)


@pytest.fixture(autouse=True)
def version_mode(monkeypatch):
    monkeypatch.setenv('DT_CREDIT_MODE', MODE)
    monkeypatch.delenv('DT_NATIVE_HOST_CACHE_BATCH', raising=False)
    monkeypatch.delenv('DT_NATIVE_HOST_MEMORY_LEASE', raising=False)


@pytest.mark.parametrize('multipliers', [[.25, -1.5, 0.], [.5, -3., 0.]])
def test_unit_credit_variance_precedes_multiplier_and_traces_zero_coefficient(multipliers):
    data = native_training_data([artifact(i) for i in range(3)], multipliers=multipliers)
    original = data.batch['dt_grpo_multipliers'].clone()
    worker = EventTransportWorker()
    output = compute_direct_target_credit(data, worker, eos_token_id=99, pad_token_id=0)
    assert torch.equal(data.batch['dt_grpo_multipliers'], original)
    assert set(worker.returned) == {'uid-0', 'uid-1', 'uid-2'}
    raw = {key: torch.zeros_like(output.batch[key]) for key in CREDIT_KEYS}
    for row, item in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
        for slot, position in enumerate(item['retained_response_positions']):
            if position >= 0:
                for key in CREDIT_KEYS:
                    raw[key][row, slot] = worker.returned[f'uid-{row}'][key][position]
    mask = native_mask(data)
    scale = torch.rsqrt(masked_var(raw['dt_token_advantages'], mask) + 1e-8)
    for key in CREDIT_KEYS:
        expected = torch.where(mask, raw[key] * scale * original.float()[:, None], 0.)
        assert torch.equal(output.batch[key], expected)
    positive = mask[0] & (raw['dt_token_advantages'][0] > 0)
    assert bool((output.batch['dt_token_advantages'][0, positive] > 0).all())
    # The trainer must not standardize the already multiplied v6 advantage.
    from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
    data.batch['response_mask'] = mask
    data = data.union(output)
    computed = compute_advantage(data, AdvantageEstimator.DELTATRACE)
    assert torch.equal(computed.batch['advantages'], output.batch['dt_token_advantages'])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('credits', [(-.1, 1.), (-1., 1.)])
def test_multitarget_one_shared_pre_coefficient_scale_at_original_retained_slots(credits):
    items = [events(artifact(i), credits) for i in range(3)]
    repeated = deepcopy(items[0])
    repeated['retained_response_positions'] = [3, 4, 5, -1]
    data = native_training_data(items + [repeated],
        identities=['uid-0', 'uid-1', 'uid-2', 'uid-0'], multipliers=[sum(credits)] * 4)
    worker = EventTransportWorker()
    output = compute_direct_target_credit(data, worker, eos_token_id=99, pad_token_id=0)
    mask = native_mask(data)
    population = []
    for row, item in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
        uid = str(data.non_tensor_batch['traj_uid'][row])
        for event in item['dt_reward_target_events']:
            value = worker.returned[str((uid, 'str', event['event_id']))]['dt_token_advantages']
            population += [value[position] for slot, position in enumerate(item['retained_response_positions'])
                           if position >= 0 and mask[row, slot]]
    values = torch.stack(population)
    scale = torch.rsqrt(masked_var(values, torch.ones_like(values, dtype=torch.bool)) + 1e-8)
    for row, item in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
        uid = str(data.non_tensor_batch['traj_uid'][row])
        for key in CREDIT_KEYS:
            expected = torch.zeros_like(output.batch[key][row])
            for event in item['dt_reward_target_events']:
                value = worker.returned[str((uid, 'str', event['event_id']))][key]
                weighted = value * scale * torch.tensor(event['credit'], dtype=value.dtype)
                for slot, position in enumerate(item['retained_response_positions']):
                    if position >= 0:
                        expected[slot] += weighted[position]
            assert torch.equal(output.batch[key][row], expected)
    if sum(credits) == 0:
        assert bool(output.batch['dt_token_advantages'].any())


def test_zero_event_still_contributes_its_unit_credit_to_shared_statistic():
    item = events(artifact(0))
    item['dt_reward_target_events'].append(dict(event_id='zero', target_offsets=[1, 4], credit=0.,
                                               future_policy_mask=[True] * 6))
    worker = EventTransportWorker()
    result = compute_direct_target_credit(native_training_data([item]), worker,
                                         eos_token_id=99, pad_token_id=0)
    assert any("'zero'" in uid for uid in worker.returned)
    assert all(bool(torch.isfinite(result.batch[key]).all()) for key in CREDIT_KEYS)


class StripResourceBoundary(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == 'verl.utils.native_host_memory' else node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == 'release_unused_host_cache':
                return None
        return self.generic_visit(node)

    def visit_With(self, node):
        node = self.generic_visit(node)
        node.items = [item for item in node.items if not (
            isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == 'host_memory_lease')]
        return node if node.items else node.body


@pytest.mark.parametrize('key', ['TextCraft_DTPO', 'AppWorld_DTPO', 'TextCraft_PPO', 'AppWorld_PPO'])
def test_native_resource_patch_preserves_every_original_nonresource_statement(key):
    record = json.loads((AUDIT / (key + '-source-prepared.json')).read_bytes())
    parent = json.loads((Path(record['parent_run']) / 'launch.json').read_bytes())['effective_environment']['VERL_ROOT']
    spec = importlib.util.spec_from_file_location('original_DT_boundary_patch', AUDIT / 'source/patch_verl_agent2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for changed in record['host_memory_changes']:
        path = Path(changed)
        if path.name == 'native_host_memory.py':
            continue
        relative = path.relative_to(record['verl_root'])
        expected = (Path(parent) / relative).read_text()
        if key.endswith('DTPO') and path.name == 'ray_trainer.py':
            expected = module.patch_dt_advantage_preprocessing(expected)
        stripped = StripResourceBoundary().visit(ast.parse(path.read_text()))
        assert ast.dump(stripped, include_attributes=False) == ast.dump(ast.parse(expected), include_attributes=False)
    assert not torch.cuda.is_initialized()


def test_default_host_release_is_inert_without_opt_in():
    from verl.utils.native_host_memory import release_unused_host_cache
    release_unused_host_cache('CPU_default_path')
    assert not torch.cuda.is_initialized()
