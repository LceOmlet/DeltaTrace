"""Real owner/tokenizer boundary tests; RPC doubles do not claim task success."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto
from agent_system.multi_turn_rollout import rollout_loop as owner
from agent_system.environments.env_package.appworld import envs as app_owner
from agent_system.environments import env_manager as manager_owner
from agent_system.memory import memory as memory_owner
from patch_verl_agent2 import patch_context_budget, patch_appworld_active_steps, patch_memory_active_steps


def patched_class(module, source, name):
    namespace = dict(vars(module))
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<patched-owner>', 'exec'), namespace)
    return namespace[name]


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)


@pytest.fixture
def collector(tokenizer):
    source = Path(os.environ.get('CONTEXT_COLLECTOR_SOURCE', owner.__file__)).read_text()
    cls = patched_class(owner, patch_context_budget(source), 'TrajectoryCollector')
    config = OmegaConf.create(dict(
        data=dict(max_prompt_length=31598, truncation='error', return_raw_chat=True,
                  apply_chat_template_kwargs=dict(enable_thinking=False)),
        algorithm=dict(adv_estimator='deltatrace'),
        env=dict(context_budget_action='end_episode', max_steps=4, rollout=dict(n=1))))
    result = cls(config, tokenizer)
    reference = Path(os.environ.get('CONTEXT_COLLECTOR_REFERENCE', owner.__file__)).read_text()
    result.reference_type = patched_class(owner, reference, 'TrajectoryCollector')
    return result


def batch(n=1):
    return DataProto.from_single_dict(dict(
        input_ids=torch.zeros((n, 1), dtype=torch.long),
        raw_prompt=np.array([[dict(role='user', content='Task')]] * n, dtype=object),
        data_source=np.array(['AppWorld'] * n, dtype=object)))


def messages_at_length(tokenizer, length):
    chat = [dict(role='user', content=' x')]
    def count():
        rendered = tokenizer.apply_chat_template(chat, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return len(tokenizer.encode(rendered, add_special_tokens=False))
    chat[0]['content'] += ' x' * (length - count())
    assert count() == length
    return chat


def preprocess_args(collector, chat):
    kwargs = dict(item=0, gen_batch=batch(), obs={'text': [chat[0]['content']]})
    if hasattr(collector, '_copy_messages'):
        kwargs['messages'] = chat
    return kwargs


@pytest.mark.parametrize('length', [31597, 31598, 31599, 31748])
def test_actual_qwen_tokenizer_reserved_32k_boundary(collector, length):
    chat = messages_at_length(collector.tokenizer, length)
    before = copy.deepcopy(chat)
    kwargs = preprocess_args(collector, chat)
    result = collector.preprocess_single_sample(**kwargs)
    assert chat == before and result['raw_prompt'] == before
    if length > 31598:
        assert result['context_budget_tokens'] == length
        assert result['attention_mask'].sum() == 0
        assert result['raw_prompt_ids'] == []
    else:
        assert result['context_budget_tokens'] == 0
        collector.config.env.context_budget_action = 'error'
        original = collector.reference_type(collector.config, collector.tokenizer)
        expected = original.preprocess_single_sample(**kwargs)
        for key in ['input_ids', 'attention_mask', 'position_ids']:
            assert torch.equal(result[key], expected[key])
        assert result['raw_prompt_ids'] == expected['raw_prompt_ids']


def test_default_still_rejects_overflow(collector):
    collector.config.env.context_budget_action = 'error'
    with pytest.raises((RuntimeError, NotImplementedError)):
        collector.preprocess_single_sample(**preprocess_args(collector,
            messages_at_length(collector.tokenizer, 31748)))


def test_owner_patch_idempotent():
    for module, patch in [(owner, patch_context_budget), (app_owner, patch_appworld_active_steps),
                          (memory_owner, patch_memory_active_steps)]:
        source = Path(module.__file__).read_text()
        modified = patch(source)
        ast.parse(modified)
        assert patch(modified) == modified
    source = Path(manager_owner.__file__).read_text()
    modified = patch_appworld_active_steps(source, manager=True)
    ast.parse(modified)
    assert patch_appworld_active_steps(modified, manager=True) == modified


@pytest.mark.parametrize('mask', [None, [False, True]])
def test_actual_manager_forwards_mask_to_environment_owner(mask):
    cls = patched_class(manager_owner, patch_appworld_active_steps(
        Path(manager_owner.__file__).read_text(), manager=True), 'AppWorldEnvironmentManager')
    env = cls.__new__(cls)
    env.config = OmegaConf.create({'env': {'history_length': 2}})
    env.tasks = ['task0', 'task1']
    env.supervisors = [dict(first_name='Test', last_name='Person', email='test@example.com', phone_number='0')] * 2
    env.projection_f = lambda actions: (actions, [True, True])
    env.envs = SimpleNamespace(step=Mock(return_value=(
        ['observed0', 'observed1'], [0., 10.], [True, True], [{'won': False}, {'won': True}])))
    env.memory = patched_class(memory_owner, patch_memory_active_steps(
        Path(memory_owner.__file__).read_text()), 'SimpleMemory')()
    env.memory.reset(batch_size=2)
    env.step(['a', 'b'], active_masks=mask)
    assert env.envs.step.call_args.args == (['a', 'b'],)
    assert env.envs.step.call_args.kwargs == ({} if mask is None else {'active_masks': mask})
    assert [len(env.memory[i]) for i in range(2)] == ([1, 1] if mask is None else [0, 1])


@pytest.mark.parametrize('mask', [None, [True, False], [False, False]])
def test_appworld_rpc_only_executes_active_actions(monkeypatch, mask):
    cls = patched_class(app_owner, patch_appworld_active_steps(Path(app_owner.__file__).read_text()), 'AppWorldEnvs')
    env = cls.__new__(cls)
    env.num_processes = 2
    env._last_observations = [('last0', {'won': False, 'step_count': 1}),
                              ('last1', {'won': True, 'step_count': 2})]
    env.workers = [SimpleNamespace(step=SimpleNamespace(remote=Mock(return_value=(
        f'new{i}', 10.0, True, {'won': True, 'step_count': 3})))) for i in range(2)]
    monkeypatch.setattr(app_owner.ray, 'get', lambda values: values)
    observations, rewards, dones, infos = env.step(['a', 'b'], active_masks=mask)
    for i, worker in enumerate(env.workers):
        active = mask is None or mask[i]
        assert worker.step.remote.call_count == int(active)
        if active:
            assert observations[i] == f'new{i}' and rewards[i] == 10.0
        else:
            assert observations[i] == f'last{i}' and rewards[i] == 0
            assert infos[i]['step_count'] == i + 1


@pytest.mark.parametrize('all_overflow', [False, True])
def test_real_collector_stops_only_overflow_episode_and_preserves_rows(collector, all_overflow):
    tokenizer = collector.tokenizer
    response = torch.tensor(tokenizer.encode('pass', add_special_tokens=False)).repeat(2, 1)
    calls = []
    def generate(data):
        active = data.non_tensor_batch['rollout_active_mask'].copy()
        calls.append(active)
        rmask = torch.tensor(active.astype(np.int64)).unsqueeze(1).expand_as(response)
        attn = torch.cat((data.batch['attention_mask'], rmask), dim=1)
        return DataProto.from_single_dict(dict(
            input_ids=torch.cat((data.batch['input_ids'], response), dim=1),
            responses=response.clone(), attention_mask=attn,
            position_ids=owner.compute_position_id_with_mask(attn)))
    actor = SimpleNamespace(world_size=1, generate_sequences=generate)
    class EnvironmentRPCDouble:
        def reset(self, kwargs):
            self.steps = 0
            self.masks = []
            return {'text': ['initial', 'initial'], 'image': None}, [{}, {}]
        def step(self, actions, active_masks=None):
            self.steps += 1
            self.masks.append(active_masks.copy())
            long = ' x' * 31748
            return ({'text': [long, long if all_overflow else 'short'], 'image': None},
                    np.array([0., 10. if self.steps == 3 else 0.]),
                    np.array([False, self.steps == 3]),
                    [{'won': False}, {'won': self.steps == 3}])
        def success_evaluator(self, **kwargs):
            self.final = kwargs
            return {'success_rate': np.array([0., float(self.steps == 3)])}
    env = EnvironmentRPCDouble()
    rows, rewards, lengths, *_ = collector.vanilla_multi_turn_loop(batch(2), actor, env)
    assert lengths.tolist() == ([1., 1.] if all_overflow else [1., 3.])
    assert rewards.tolist() == ([0., 0.] if all_overflow else [0., 10.])
    assert [r['rewards'] for r in rows[0] if r['active_masks']] == [0.]
    assert all(mask.tolist() == [False, True] for mask in calls[1:])
    assert len(calls) == (1 if all_overflow else 3)
    assert all(np.array_equal(a, b) for a, b in zip(calls, env.masks))
